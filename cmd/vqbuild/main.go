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
	"github.com/ThiraSoft/golem/qwen35"
	"github.com/ThiraSoft/golem/token/bytebpe"
	"github.com/ThiraSoft/golem/vk"
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
	codec := flag.String("codec", "lattice", "lattice or trellis; the trellis needs no decode table at all")
	trK := flag.Int("k", 3, "trellis: bits emitted per weight")
	trL := flag.Int("L", 12, "trellis: state bits; the Viterbi costs 2^L per weight")
	trSeq := flag.Int("seq", 1024, "trellis: weights coded as one sequence")
	trGain := flag.Float64("gain", 0.94, "trellis: how far the codebook is narrowed against the source")
	onCard := flag.Bool("vulkan", false, "run the trellis Viterbi on a Vulkan device; it is five hundred times the lattice's work and the only half that cares where it runs")
	scaleBlk := flag.Int("scale", 64, "weights per fp16 scale")
	step8 := flag.Bool("step8", false, "store each block's scale as the format's eight-bit step code rather than an fp16, which is half a bit a block cheaper at -scale 64")
	alpha := flag.Float64("alpha", 0.5, "salience exponent")
	outliers := flag.Int("outliers", 32, "columns held at 8 bits")
	search := flag.Bool("search", false, "search each block's scale instead of taking its RMS")
	hadGroup := flag.Int("hadamard", 128, "rotation group; 0 leaves the weights unrotated")
	ntok := flag.Int("tokens", 256, "calibration tokens")
	window := flag.Int("window", 0, "measure the sites in windows of this many tokens; 0 is one pass over the lot, which measures better")
	calibCard := flag.Bool("calib-vulkan", false, "measure the sites by running the model on a Vulkan device; on the processor this is most of a conversion's wall clock")
	calibFile := flag.String("calib", "", "text to calibrate on; the built-in paragraph is 173 tokens, which is few enough to overfit")
	roles := flag.String("roles", "all", "which matrices to compress; the rest stay BF16")
	flag.Parse()

	if *onCard && *codec == "trellis" {
		dev, err := vk.Open()
		must(err)
		defer dev.Close()
		enc, err := vk.NewTrellisEncoder(dev, 1<<24)
		must(err)
		defer enc.Close()
		var onCardWeights, offCardWeights int64
		compress.TrellisAccel = func(norm []float32, o compress.TrellisOpts) bool {
			// The kernel is compiled for one shape. Anything else falls back
			// rather than quietly answering a different question.
			if o.K != vk.TrellisGPUK || o.L != vk.TrellisGPUL || o.Seq != vk.TrellisGPUSeq {
				offCardWeights += int64(len(norm))
				return false
			}
			if err := enc.Quantize(norm, float32(o.Gain)); err != nil {
				fmt.Printf("  the card refused a matrix (%v); the processor takes it\n", err)
				offCardWeights += int64(len(norm))
				return false
			}
			onCardWeights += int64(len(norm))
			return true
		}
		defer func() {
			fmt.Printf("%d M weights through the card, %d M through the processor\n",
				onCardWeights/1e6, offCardWeights/1e6)
		}()
	}

	// One pass over a calibration text, keeping only the per-column power of
	// the activations that meet each matrix.
	sal := map[string][]float64{}
	cnt := map[string]int{}
	// The context has to hold what one pass reads, or the forward pass walks
	// off the end of the cache — a panic in the attention, a long way from the
	// flag that caused it.
	//
	// Windows are offered and are not the default, which is the opposite of
	// what this file assumed. Running eight thousand tokens as one pass makes
	// the attention quadratic and costs thirty-five minutes against ten, so
	// sweeping in windows the way cmd/golemquant does looked like the free
	// correction — and the guess behind it, that the sites should be measured
	// in the regime the evaluation reads them in, was wrong. On Qwen3-4B, at
	// the same eight thousand tokens: one pass reads 19.7004 and 0.0550,
	// sixteen windows of 512 read 19.7808 and 0.0566. Worse on both.
	//
	// What the same three runs say about the token count is stranger and is
	// left as an observation rather than a rule, being two points on one model:
	// 2048 tokens in one pass reads 19.7285 and **0.0510**, which is worse
	// perplexity than 8192 and better divergence. More calibration appears to
	// buy average likelihood and sell per-token agreement — which would be the
	// salience fitting its proxy, activation power, better and better while the
	// thing that proxy stands for drifts.
	ctx := 4096
	for ctx < *ntok && *window == 0 {
		ctx *= 2
	}
	for ctx < *window {
		ctx *= 2
	}
	m, err := qwen.Open(*src, ctx)
	must(err)
	v, err := bytebpe.Load(m.File())
	must(err)
	text := calibText
	if *calibFile != "" {
		b, err := os.ReadFile(*calibFile)
		must(err)
		text = string(b)
	}
	ids := v.Encode(text, true, false)
	if len(ids) > *ntok {
		ids = ids[:*ntok]
	}
	// The corpus a file is judged on must not be the corpus it was fitted to,
	// and this is the place that mistake gets made.
	fmt.Printf("calibrating on %d tokens\n", len(ids))

	// The card first, when asked. It measures the same thing and it is the
	// difference between seven minutes and seconds; the processor's tap below
	// stays as the fallback and as what the card is checked against.
	var salience map[string][]float32
	if *calibCard {
		win := *window
		if win == 0 {
			win = *ntok
		}
		if s, ok := calibrateOnCard(*src, text, *ntok, win); ok {
			salience = s
		}
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
	if salience == nil {
		if *window == 0 {
			m.ForwardBatch(ids, 0)
		} else {
			// Whole windows only, which is what cmd/golemquant's sweep does: a
			// short tail would measure the sites in a third regime.
			for at := 0; at+*window <= len(ids); at += *window {
				m.Reset()
				m.ForwardBatch(ids[at:at+*window], 0)
			}
		}
		qwen.Calib = nil
		fmt.Printf("calibrated on %d tokens in %s, %d sites\n",
			len(ids), time.Since(t0).Round(time.Millisecond), len(sal))
		salience = make(map[string][]float32, len(sal))
		for k, s := range sal {
			out := make([]float32, len(s))
			for j := range s {
				out[j] = float32(math.Sqrt(s[j] / float64(cnt[k])))
			}
			salience[k] = out
		}
	}
	qwen.Calib = nil
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

		var opts compress.Opts
		if *codec == "trellis" {
			opts = compress.Opts{UseTrellis: true, Step8: *step8,
				ScaleBlock: *scaleBlk, HadGroup: *hadGroup,
				Tr: compress.TrellisOpts{K: *trK, L: *trL, Seq: *trSeq,
					Gain: *trGain, Code: compress.Code1MAD}}
		} else {
			lat, table := compress.LatE8, e8Levels
			if *lattice == "D4" {
				lat, table = compress.LatD4, d4Levels
			}
			lv := table[levelOf(role)]
			opts = compress.Opts{UseLattice: true, Lat: lat, Step8: *step8,
				MaxNorm2: float32(lv.r), Beta: lv.beta, ScaleBlock: *scaleBlk, HadGroup: *hadGroup,
				SearchScale: *search}
		}
		sc := compress.Scheme{Alpha: *alpha, Outliers: *outliers, VQ: opts}
		var s []float32
		if isBlk {
			s = salience[fmt.Sprintf("%d/%s", blkN, feeds[mat])]
		} else {
			// the embedding, which is also the logit head: no site feeds it, so
			// it goes through the quantizer alone
			sc.Alpha, sc.Outliers = 0, 0
		}
		if opts.HadGroup > 0 && cols%opts.HadGroup != 0 {
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
	if n := compress.StepClipped(); n > 0 {
		fmt.Printf("%d blocks of the %.0f M weights landed on an end of the step grid\n", n, weights/1e6)
	}
	fmt.Printf("rewritten in %s\n", time.Since(t0).Round(time.Second))
}

// The (shell, resolution) pairs that trace each lattice's rate-distortion
// curve. A step up costs roughly a third of a bit. The shell is set so that it
// holds what the resolution produces: a normalised subvector has ‖βx‖² ≈ dβ²,
// and the shell is about 1.6 times that.
type level struct{ r, beta float64 }

// β is not free to be large. It used to be set so a subvector filled the whole
// shell, on the reasoning that overflow costs nothing — which held only because
// the bench reconstructed an overflowing subvector by the factor it had been
// pulled by, and no decoder has that factor. Under a reconstruction a file can
// perform, overflow is clipping, and the measured optimum is four times lower:
// β ≈ 0.63·√(r²/d), which reads 2.0 at D4's r²=40 and 4.0 at its r²=160.
var e8Levels = []level{
	{10, 0.70}, {16, 0.89}, {26, 1.14}, {42, 1.44}, {62, 1.75},
	{100, 2.23}, {156, 2.78}, {260, 3.59}, {460, 4.78}, {820, 6.38},
}

// D4 at the same budget: half the dimension, so the same rate needs a quarter
// of the squared radius, and the shell stays small enough to enumerate into a
// table a shader can hold — 3961 points at r²=40, thirty-one kibibytes.
var d4Levels = []level{
	{6, 0.77}, {10, 1.00}, {20, 1.41}, {40, 1.99}, {60, 2.44},
	{100, 3.15}, {160, 3.98}, {260, 5.08}, {460, 6.76}, {820, 9.02},
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

// calibrateOnCard is the salience measured with the model on a card, which is
// the same measurement cmd/golemquant makes and for the same reason: on the
// processor it is most of a conversion's wall clock. Once the Viterbi moved to
// the card, the eight minutes a six-hundred-million-weight model took were
// seven of calibration and one of quantizing — the wrong half was on the card.
//
// The site keys are "block/site" with the same four names the processor's tap
// uses — vk/qwen_calib.go names them qkv, o, gateup and down — so what comes
// back drops straight into the map the rest of this command already reads.
func calibrateOnCard(path, text string, ntok, win int) (map[string][]float32, bool) {
	give := func(err error) (map[string][]float32, bool) {
		fmt.Printf("no calibration on the card (%v); the processor then\n", err)
		return nil, false
	}
	m, err := qwen35.Open(path, win)
	if err != nil {
		return give(err)
	}
	defer m.Close()
	if err := m.UseVulkanStack(); err != nil {
		return give(err)
	}
	if err := m.StartVulkanCalibration(); err != nil {
		return give(err)
	}
	v, err := bytebpe.Load(m.File())
	if err != nil {
		return give(err)
	}
	ids := v.Encode(text, true, false)
	if len(ids) > ntok {
		ids = ids[:ntok]
	}
	t0 := time.Now()
	for at := 0; at+win <= len(ids); at += win {
		m.Reset()
		m.ForwardBatch(ids[at:at+win], 0)
		m.CountVulkanCalibration(win)
	}
	sums, rows, err := m.VulkanCalibrationSums()
	if err != nil {
		return give(err)
	}
	if rows == 0 {
		return give(fmt.Errorf("the card measured no rows"))
	}
	out := make(map[string][]float32, len(sums))
	for k, sv := range sums {
		u := make([]float32, len(sv))
		for j, x := range sv {
			u[j] = float32(math.Sqrt(float64(x) / float64(rows)))
		}
		out[k] = u
	}
	fmt.Printf("calibrated on %d tokens in %s on the card, %d sites\n",
		len(ids), time.Since(t0).Round(time.Millisecond), len(out))
	return out, true
}
