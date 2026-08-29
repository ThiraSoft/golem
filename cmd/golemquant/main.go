package main

// golemquant: a checkpoint in, a .golem out.
//
// The file is a GGUF — the same container, so the vocabulary, the rope base and
// the chat template travel unchanged — carrying tensors of a type llama.cpp
// does not know. What is new besides the weights is one vector a block: the
// per-column scale the activations of each site must meet, with the rotation's
// sign flips folded into it.

import (
	"encoding/binary"
	"flag"
	"fmt"
	"math"
	"os"
	"sort"
	"strings"
	"time"

	"github.com/ThiraSoft/golem/compress"
	"github.com/ThiraSoft/golem/nn"
	"github.com/ThiraSoft/golem/qwen"
	"github.com/ThiraSoft/golem/tensors"
	"github.com/ThiraSoft/golem/token/bytebpe"
)

// feeds says which site's activations reach each matrix, and so which vector it
// shares.
var feeds = map[string]string{
	"attn_q": "qkv", "attn_k": "qkv", "attn_v": "qkv",
	"attn_output": "o",
	"ffn_gate":    "gateup", "ffn_up": "gateup",
	"ffn_down": "down",
}

func main() {
	src := flag.String("model", "", "the BF16 checkpoint to convert")
	dst := flag.String("out", "", "the .golem file to write")
	alpha := flag.Float64("alpha", 0.5, "salience exponent; 0 leaves the columns alone")
	hadGroup := flag.Int("hadamard", 128, "rotation group; 0 leaves the weights unrotated")
	beta := flag.Float64("beta", 2, "how far a block is scaled up before rounding")
	scaleBlk := flag.Int("scale", 64, "weights sharing one fp16 step")
	ntok := flag.Int("tokens", 256, "calibration tokens")
	flag.Parse()

	salience := calibrate(*src, *ntok)

	g, err := tensors.OpenGGUF(*src)
	must(err)
	defer g.Close()

	names := make([]string, 0, len(g.Tensors))
	for n := range g.Tensors {
		names = append(names, n)
	}
	sort.Strings(names)

	params := compress.D4Params{Beta: *beta, ScaleBlock: *scaleBlk,
		HadGroup: *hadGroup, SearchScale: true}

	// One vector a site: the sign flips of the rotation over the salience
	// scale. The weights are multiplied by it, the activations by its
	// reciprocal, and nn.PrepareD4G is both.
	signs := map[int][]float32{}
	pre := map[string][]float32{}    // what the activations meet
	weight := map[string][]float32{} // its reciprocal, what the weights meet
	for key, sal := range salience {
		cols := len(sal)
		if signs[cols] == nil {
			signs[cols] = compress.RandomSigns(cols, int64(cols)*7919)
		}
		s := saliencyScale(sal, *alpha)
		p := make([]float32, cols)
		q := make([]float32, cols)
		for j := range s {
			q[j] = signs[cols][j] * s[j]
			p[j] = 1 / q[j]
		}
		pre[key], weight[key] = p, q
	}

	var out []tensors.OutTensor
	var bits, count float64
	t0 := time.Now()
	for _, name := range names {
		t := g.Tensors[name]
		if t.DType != "BF16" || len(t.Shape) != 2 {
			out = append(out, tensors.OutTensor{Name: name, Shape: t.Shape,
				DType: t.DType, Data: append([]byte(nil), t.Raw...)})
			continue
		}
		cols := t.Shape[0]
		rows := t.Elems() / cols
		w, err := t.F32()
		must(err)

		var q []float32
		key := ""
		if blk, mat, ok := parse(name); ok {
			key = fmt.Sprintf("%d/%s", blk, feeds[mat])
			q = weight[key]
		}
		p := params
		if q == nil {
			// The embedding table, which is also the logit head. It is read a
			// row at a time for the input, and a rotated row is not the row
			// anybody wants, so it is stored plain.
			p.HadGroup = 0
		}
		data := compress.EncodeD4G(w, rows, cols, q, p)
		out = append(out, tensors.OutTensor{Name: name, Shape: t.Shape,
			DType: "D4G", Data: data})
		bits += float64(len(data)) * 8
		count += float64(len(w))
		fmt.Printf("  %-32s %6dx%-6d %s\n", name, rows, cols, sizeOf(len(data)))
	}

	// The vectors, one a site, as plain F32 tensors the loader binds by name.
	preNames := make([]string, 0, len(pre))
	for k := range pre {
		preNames = append(preNames, k)
	}
	sort.Strings(preNames)
	for _, k := range preNames {
		parts := strings.SplitN(k, "/", 2)
		name := fmt.Sprintf("blk.%s.%s.pre", parts[0], parts[1])
		v := pre[k]
		raw := make([]byte, len(v)*4)
		for i, x := range v {
			binary.LittleEndian.PutUint32(raw[4*i:], math.Float32bits(x))
		}
		out = append(out, tensors.OutTensor{Name: name, Shape: []int{len(v)},
			DType: "F32", Data: raw})
		bits += float64(len(raw)) * 8
	}

	meta := map[string]any{}
	for k, v := range g.Meta {
		meta[k] = v
	}
	meta["golem.d4.hadamard_group"] = uint32(*hadGroup)
	meta["golem.d4.radius"] = uint32(nn.D4Radius)
	meta["golem.d4.scale_block"] = uint32(*scaleBlk)
	meta["general.file_type"] = uint32(1000)

	must(tensors.WriteGGUF(*dst, meta, out))
	fmt.Printf("\n%.0f M weights at %.3f bits each — %s\n",
		count/1e6, bits/count, sizeOf(int(bits/8)))
	fmt.Printf("written to %s in %s\n", *dst, time.Since(t0).Round(time.Second))
}

// calibrate runs a text through the model and keeps, for each site, the
// per-column power of the activations that reach it.
func calibrate(path string, ntok int) map[string][]float32 {
	m, err := qwen.Open(path, 4096)
	must(err)
	defer m.Close()
	v, err := bytebpe.Load(m.File())
	must(err)
	ids := v.Encode(calibText, true, false)
	if len(ids) > ntok {
		ids = ids[:ntok]
	}
	sums := map[string][]float64{}
	seen := map[string]int{}
	qwen.Calib = func(block int, site string, rows [][]float32) {
		key := fmt.Sprintf("%d/%s", block, site)
		s := sums[key]
		if s == nil {
			s = make([]float64, len(rows[0]))
			sums[key] = s
		}
		for _, r := range rows {
			for j, x := range r {
				s[j] += float64(x) * float64(x)
			}
		}
		seen[key] += len(rows)
	}
	t0 := time.Now()
	m.ForwardBatch(ids, 0)
	qwen.Calib = nil
	out := map[string][]float32{}
	for k, s := range sums {
		v := make([]float32, len(s))
		for j := range s {
			v[j] = float32(math.Sqrt(s[j] / float64(seen[k])))
		}
		out[k] = v
	}
	fmt.Printf("calibrated on %d tokens in %s, %d sites\n",
		len(ids), time.Since(t0).Round(time.Millisecond), len(out))
	return out
}

// saliencyScale turns per-column activation power into the factor the weights
// are multiplied by, normalised so the matrix keeps its overall size.
func saliencyScale(sal []float32, alpha float64) []float32 {
	out := make([]float32, len(sal))
	if alpha == 0 {
		for j := range out {
			out[j] = 1
		}
		return out
	}
	var geo float64
	for _, v := range sal {
		geo += math.Log(math.Max(float64(v), 1e-8))
	}
	geo = math.Exp(geo / float64(len(sal)))
	for j, v := range sal {
		out[j] = float32(math.Pow(math.Max(float64(v), 1e-8)/geo, alpha))
	}
	return out
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

func sizeOf(n int) string {
	switch {
	case n > 1<<30:
		return fmt.Sprintf("%.2f GiB", float64(n)/(1<<30))
	case n > 1<<20:
		return fmt.Sprintf("%.1f MiB", float64(n)/(1<<20))
	}
	return fmt.Sprintf("%.1f KiB", float64(n)/1024)
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
