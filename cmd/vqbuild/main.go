package main

// vqbuild: writes a copy of a BF16 checkpoint whose weights have been through
// the compression pipeline and back.
//
// The copy is still BF16 on disk, and still the same size — what it carries is
// the *values* a kernel reading the compressed form would reconstruct. That is
// what lets the loss be measured end to end, with the model golem already has,
// before a single line of Vulkan is written. The bits actually spent are
// reported here rather than shown by the file.

import (
	"encoding/binary"
	"flag"
	"fmt"
	"io"
	"math"
	"os"
	"sort"
	"strings"
	"time"

	"github.com/ThiraSoft/golem/compress"
	"github.com/ThiraSoft/golem/qwen"
	"github.com/ThiraSoft/golem/token/bytebpe"
)

// which site's activations feed each matrix.
var feeds = map[string]string{
	"attn_q": "qkv", "attn_k": "qkv", "attn_v": "qkv",
	"attn_output": "o",
	"ffn_gate":    "gateup", "ffn_up": "gateup",
	"ffn_down": "down",
}

func main() {
	src := flag.String("model", "", "BF16 GGUF to compress")
	dst := flag.String("out", "", "where to write the reconstructed copy")
	plan := flag.String("plan", "default:3", "bit levels per role, e.g. default:3,token_embd:6,ffn_down:4")
	lattice := flag.String("lattice", "E8", "E8 or D4; D4 is the one whose decode table fits a workgroup")
	scaleBlk := flag.Int("scale", 64, "weights per fp16 scale")
	alpha := flag.Float64("alpha", 0.5, "salience exponent")
	outliers := flag.Int("outliers", 32, "columns held at 8 bits")
	search := flag.Bool("search", false, "search each block's scale instead of taking its RMS")
	ntok := flag.Int("tokens", 256, "calibration tokens")
	roles := flag.String("roles", "all", "which matrices to compress; the rest stay BF16")
	flag.Parse()

	// One pass over a calibration text, keeping only the per-column power of
	// the activations that meet each matrix.
	sal := map[string][]float64{}
	cnt := map[string]int{}
	m, err := qwen.Open(*src, 4096)
	must(err)
	v, err := bytebpe.Load(m.File())
	must(err)
	ids := v.Encode(calibText, true, false)
	if len(ids) > *ntok {
		ids = ids[:*ntok]
	}
	qwen.Calib = func(block int, site string, rows [][]float32) {
		key := fmt.Sprintf("%d/%s", block, site)
		s := sal[key]
		if s == nil {
			s = make([]float64, len(rows[0]))
			sal[key] = s
		}
		for _, r := range rows {
			for j, x := range r {
				s[j] += float64(x) * float64(x)
			}
		}
		cnt[key] += len(rows)
	}
	t0 := time.Now()
	m.ForwardBatch(ids, 0)
	qwen.Calib = nil
	fmt.Printf("calibrated on %d tokens in %s, %d sites\n",
		len(ids), time.Since(t0).Round(time.Millisecond), len(sal))
	salience := map[string][]float32{}
	for k, s := range sal {
		out := make([]float32, len(s))
		for j := range s {
			out[j] = float32(math.Sqrt(s[j] / float64(cnt[k])))
		}
		salience[k] = out
	}
	file := m.File()

	// The copy, patched tensor by tensor.
	must(copyFile(*src, *dst))
	out, err := os.OpenFile(*dst, os.O_WRONLY, 0)
	must(err)

	levelOf := parsePlan(*plan)

	names := make([]string, 0, len(file.Tensors))
	for n := range file.Tensors {
		names = append(names, n)
	}
	sort.Strings(names)

	keep := func(mat string) bool {
		if *roles == "all" {
			return true
		}
		for _, r := range strings.Split(*roles, ",") {
			if r == mat {
				return true
			}
		}
		return false
	}

	var bits, weights float64
	t0 = time.Now()
	for _, name := range names {
		t := file.Tensors[name]
		if t.DType != "BF16" || len(t.Shape) != 2 {
			continue
		}
		cols := t.Shape[0]
		rows := t.Elems() / cols
		blkN, mat, isBlk := parse(name)
		role := mat
		if !isBlk {
			role = "token_embd"
		}
		if !keep(role) {
			bits += 16 * float64(t.Elems())
			weights += float64(t.Elems())
			continue
		}
		w, err := t.F32()
		must(err)

		lat, table := compress.LatE8, e8Levels
		if *lattice == "D4" {
			lat, table = compress.LatD4, d4Levels
		}
		lv := table[levelOf(role)]
		opts := compress.Opts{UseLattice: true, Lat: lat,
			MaxNorm2: float32(lv.r), Beta: lv.beta, ScaleBlock: *scaleBlk, HadGroup: 128,
			SearchScale: *search}
		sc := compress.Scheme{Alpha: *alpha, Outliers: *outliers, VQ: opts}
		var s []float32
		if isBlk {
			s = salience[fmt.Sprintf("%d/%s", blkN, feeds[mat])]
		} else {
			// the embedding, which is also the logit head: no site feeds it, so
			// it goes through the quantizer alone
			sc.Alpha, sc.Outliers = 0, 0
		}
		if cols%opts.HadGroup != 0 {
			sc.VQ.HadGroup = 0
		}
		rec := sc.Apply(w, rows, cols, s, 1234)
		if rec == nil {
			fmt.Printf("  %-32s skipped\n", name)
			continue
		}
		buf := make([]byte, len(rec)*2)
		for i, x := range rec {
			binary.LittleEndian.PutUint16(buf[2*i:], bf16(x))
		}
		_, err = out.WriteAt(buf, int64(t.Offset))
		must(err)
		b := sc.BPW(cols)
		bits += b * float64(len(rec))
		weights += float64(len(rec))
		fmt.Printf("  %-32s %5dx%-5d %.2f bpw\n", name, rows, cols, b)
	}
	must(out.Close())
	m.Close()
	fmt.Printf("\n%.0f M weights at %.3f bits per weight — %.2f GiB against %.2f GiB in BF16, %.2f in Q4_0\n",
		weights/1e6, bits/weights, bits/8/(1<<30), weights*2/(1<<30), weights*4.5/8/(1<<30))
	fmt.Printf("rewritten in %s\n", time.Since(t0).Round(time.Second))
}

// The (shell, resolution) pairs that trace each lattice's rate-distortion
// curve. A step up costs roughly a third of a bit. The shell is set so that it
// holds what the resolution produces: a normalised subvector has ‖βx‖² ≈ dβ²,
// and the shell is about 1.6 times that.
type level struct{ r, beta float64 }

// The resolution is deliberately past the point where it still matters. With
// the block scale searched rather than taken, β and that scale do the same job,
// and the error stops moving once β is large enough to use the whole shell —
// what sets the rate is the shell, and nothing else.
var e8Levels = []level{
	{10, 2.4}, {16, 3.1}, {26, 3.9}, {42, 5.0}, {62, 6.1},
	{100, 7.7}, {156, 9.6}, {260, 12.4}, {460, 16.5}, {820, 22.1},
}

// D4 at the same budget: half the dimension, so the same rate needs a quarter
// of the squared radius, and the shell stays small enough to enumerate into a
// table a shader can hold — 3961 points at r²=40, thirty-one kibibytes.
var d4Levels = []level{
	{6, 3.1}, {10, 4.0}, {20, 5.7}, {40, 8.0}, {60, 9.8},
	{100, 12.6}, {160, 16.0}, {260, 20.4}, {460, 27.1}, {820, 36.2},
}

var levels = e8Levels

// parsePlan reads "default:3,token_embd:6" into a lookup from role to level.
func parsePlan(spec string) func(string) int {
	byRole := map[string]int{}
	def := 3
	for _, part := range strings.Split(spec, ",") {
		kv := strings.SplitN(part, ":", 2)
		if len(kv) != 2 {
			continue
		}
		var n int
		fmt.Sscan(kv[1], &n)
		if n < 0 {
			n = 0
		}
		if n >= len(levels) {
			n = len(levels) - 1
		}
		if kv[0] == "default" {
			def = n
		} else {
			byRole[kv[0]] = n
		}
	}
	return func(role string) int {
		if n, ok := byRole[role]; ok {
			return n
		}
		return def
	}
}

func parse(name string) (int, string, bool) {
	if !strings.HasPrefix(name, "blk.") {
		return 0, "", false
	}
	parts := strings.Split(name, ".")
	if len(parts) != 4 {
		return 0, "", false
	}
	var n int
	fmt.Sscan(parts[1], &n)
	return n, parts[2], true
}

func bf16(x float32) uint16 {
	u := math.Float32bits(x)
	return uint16((u + 0x7fff + ((u >> 16) & 1)) >> 16)
}

func copyFile(src, dst string) error {
	in, err := os.Open(src)
	if err != nil {
		return err
	}
	defer in.Close()
	out, err := os.Create(dst)
	if err != nil {
		return err
	}
	if _, err := io.Copy(out, in); err != nil {
		out.Close()
		return err
	}
	return out.Close()
}

func must(err error) {
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

const calibText = `The quick brown fox jumps over the lazy dog. ` +
	`In computing, quantization is the process of constraining values from a ` +
	`continuous set to a relatively small discrete set. Neural network weights ` +
	`are commonly stored as 16-bit floating point numbers, and inference is ` +
	`limited by memory bandwidth rather than arithmetic throughput. ` +
	`La compression des poids d'un modèle de langage repose sur l'idée que la ` +
	`distribution des coefficients est fortement redondante. ` +
	`def fibonacci(n):\n    if n < 2:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)\n` +
	`The capital of France is Paris, and the capital of Japan is Tokyo. ` +
	`Water boils at 100 degrees Celsius at sea level, and freezes at zero. ` +
	`Un modèle de langage prédit le prochain jeton à partir de ceux qui précèdent.`
