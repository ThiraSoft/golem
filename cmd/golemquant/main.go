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

// headKey is the calibration site of the tied output head, which belongs to no
// block and so is filed under -1.
const headKey = "-1/head"

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
	clamp := flag.Float64("clamp", 24, "largest factor the salience may scale a column by, either way; 0 lets it run")
	hadGroup := flag.Int("hadamard", 128, "rotation group; 0 leaves the weights unrotated")
	beta := flag.Float64("beta", 2, "how far a block is scaled up before rounding")
	codeBits := flag.Int("bits", 12, "code width: 12 for the ordinary tier, 16 for the wide one")
	scaleBlk := flag.Int("scale", 32, "weights sharing one step code; 32 is what the format stores")
	ntok := flag.Int("tokens", 8192, "calibration tokens")
	ctx := flag.Int("ctx", 512, "calibration window")
	calibFile := flag.String("calib", "", "text to calibrate on; a built-in paragraph when empty")
	embd := flag.String("embd", "rot", "how to store token_embd: rot, plain or bf16")
	search := flag.Bool("search", false, "choose the salience of each site by the output error it leaves")
	sample := flag.Int("sample", 128, "rows a site's salience is chosen on")
	swin := flag.Int("swin", 256, "how many columns the search's Hessian keeps together")
	report := flag.Bool("report", false, "print what each matrix's codes cost it")
	keep := flag.String("keep", "", "comma-separated tensor name fragments left in BF16")
	window := flag.Int("gptq", 0, "error-compensation window in columns; 0 turns it off")
	damp := flag.Float64("damp", 0.01, "ridge on the Hessian diagonal, as a fraction of its mean")
	flag.Parse()

	text := calibText
	if *calibFile != "" {
		b, err := os.ReadFile(*calibFile)
		must(err)
		text = string(b)
	}
	win := *window
	if *search && win < *swin {
		win = *swin
	}
	salience, accs := calibrate(*src, text, *ntok, *ctx, win, *scaleBlk)

	g, err := tensors.OpenGGUF(*src)
	must(err)
	defer g.Close()

	names := make([]string, 0, len(g.Tensors))
	for n := range g.Tensors {
		names = append(names, n)
	}
	sort.Strings(names)

	params := compress.D4Params{Beta: *beta, ScaleBlock: *scaleBlk,
		HadGroup: *hadGroup, Bits: *codeBits, SearchScale: true}
	dtype := "D4G"
	if *codeBits == nn.D4Bits16 {
		dtype = "D4G16"
	}

	// One vector a site: the sign flips of the rotation over the salience
	// scale. The weights are multiplied by it, the activations by its
	// reciprocal, and nn.PrepareD4G is both.
	signs := map[int][]float32{}
	pre := map[string][]float32{}    // what the activations meet
	weight := map[string][]float32{} // its reciprocal, what the weights meet
	build := func(key string, alpha, clamp float64) ([]float32, []float32) {
		sal := salience[key]
		cols := len(sal)
		if signs[cols] == nil {
			signs[cols] = compress.RandomSigns(cols, int64(cols)*7919)
		}
		sc := saliencyScale(sal, alpha, clamp)
		p := make([]float32, cols)
		q := make([]float32, cols)
		for j := range sc {
			q[j] = signs[cols][j] * sc[j]
			p[j] = 1 / q[j]
		}
		return p, q
	}
	for key := range salience {
		pre[key], weight[key] = build(key, *alpha, *clamp)
	}

	// A site's salience is a guess at how to move error away from the columns
	// the activations use most, and a guess can be checked. With -search each
	// site tries a few exponents and bounds on a sample of one of its matrices
	// and keeps the one that leaves the product closest — measured against the
	// activations themselves, because weight error cannot see the trade the
	// scaling is making.
	//
	// It is off, because it does not work. Over six settings of the sample and
	// the window — 32, 128 and 512 rows against Hessians 256 and 1024 columns
	// wide — Qwen3-0.6B reads 39.20, 40.23, 39.64, 39.75, 39.94 and 39.64, a
	// mean of 39.73 either side of the 39.80 that one bound chosen for the
	// whole model gives. The spread is the choosing, not the choice: a metric
	// that ranks candidates by a windowed Hessian over a sample of rows is
	// noisier than the differences between the candidates, so the search picks
	// a different winner each time and lands where it started. The first run
	// read 39.20 and it would have been easy to keep only that one.
	//
	// What survives is the machinery — Acc.Energy and EnergyD4G measure what a
	// matrix costs the product rather than what it costs the weights, and that
	// is the right question whatever asks it next.
	type cand struct{ alpha, clamp float64 }
	// A narrow grid on purpose. Widening it to eighteen candidates reaching an
	// exponent of 1 and bounds of 4 and 96 makes the answer worse — 40.42
	// against 39.20 on Qwen3-0.6B, and 41.30 with three times the sample, so
	// it is not the sample. The windowed Hessian cannot see what an aggressive
	// scaling moves beyond its own window, so it ranks the extremes too well
	// and the search believes it. The grid is kept to the range the metric can
	// be trusted over.
	cands := []cand{{0.5, 0}, {0.35, 24}, {0.5, 12}, {0.5, 24}, {0.5, 48}, {0.65, 12}, {0.65, 24}, {0.8, 8}}
	searched := map[string]bool{}
	pickSalience := func(key string, w []float32, rows, cols int) {
		if !*search || searched[key] || accs[key] == nil {
			return
		}
		searched[key] = true
		n := *sample
		if n > rows {
			n = rows
		}
		p := params
		best, bestAt := math.Inf(1), cand{*alpha, *clamp}
		for _, c := range cands {
			pv, qv := build(key, c.alpha, c.clamp)
			data := compress.EncodeD4G(w[:n*cols], n, cols, qv, p, nil)
			num, den := compress.EnergyD4G(w[:n*cols], n, cols, qv, pv, p, data, accs[key])
			if e := num / den; e < best {
				best, bestAt = e, c
			}
		}
		pre[key], weight[key] = build(key, bestAt.alpha, bestAt.clamp)
		fmt.Printf("  salience %-10s alpha %.2f bound %4.0f, output error %.4f\n",
			key, bestAt.alpha, bestAt.clamp, math.Sqrt(best))
	}

	// The second pass. The Hessian of a site is what says how to spend the
	// columns not yet quantized on the error of the ones already are, and it
	// has to be taken in the basis the weights were rotated into — so it can
	// only be built once the vectors above exist, which is why this is a pass
	// of its own and not a tally kept during the first.
	comps := map[string]*compress.Comp{}
	if *window > 0 {
		comps = hessians(*src, text, *ntok, *ctx, pre, *hadGroup, *embd,
			*window, *scaleBlk, *damp)
	}

	var out []tensors.OutTensor
	var bits, count float64
	t0 := time.Now()
	for _, name := range names {
		t := g.Tensors[name]
		if kept(name, *keep) {
			out = append(out, tensors.OutTensor{Name: name, Shape: t.Shape,
				DType: t.DType, Data: append([]byte(nil), t.Raw...)})
			bits += float64(len(t.Raw)) * 8
			count += float64(t.Elems())
			continue
		}
		if name == "token_embd.weight" && *embd == "bf16" {
			// The probe: what the head alone costs. It is also the input
			// table, so it is the one tensor whose error is felt twice.
			out = append(out, tensors.OutTensor{Name: name, Shape: t.Shape,
				DType: t.DType, Data: append([]byte(nil), t.Raw...)})
			bits += float64(len(t.Raw)) * 8
			count += float64(t.Elems())
			continue
		}
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
		} else if name == "token_embd.weight" && *embd == "rot" {
			key = headKey
			// The table is also the logit head, and the head is a site like any
			// other: it has activations, so it has a salience and a rotation.
			// What the input path pays for that is one transform of the model's
			// width per token, which is nothing beside reading the row.
			q = weight[headKey]
		}
		p := params
		if q == nil {
			p.HadGroup = 0
			key = ""
		}
		pickSalience(key, w, rows, cols)
		if key != "" {
			q = weight[key]
		}
		comp := comps[key]
		data := compress.EncodeD4G(w, rows, cols, q, p, comp)
		note := ""
		if *report {
			// What the codes cost in the basis they were written in. The
			// theoretical floor for a memoryless Gaussian at this rate is
			// about seventeen decibels, so this says how much of the gap is
			// the quantizer's own and how much is everything else.
			e := compress.RelErrD4G(w, rows, cols, q, p, data)
			note = fmt.Sprintf("  rel %.4f  %.2f dB", e, -20*math.Log10(float64(e)))
		}
		out = append(out, tensors.OutTensor{Name: name, Shape: t.Shape,
			DType: dtype, Data: data})
		bits += float64(len(data)) * 8
		count += float64(len(w))
		fmt.Printf("  %-32s %6dx%-6d %s%s\n", name, rows, cols, sizeOf(len(data)), note)
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
		if k == headKey {
			if *embd != "rot" {
				continue
			}
			name = "output.pre"
		}
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
	meta["golem.d4.code_bits"] = uint32(*codeBits)
	meta["general.file_type"] = uint32(1000)

	must(tensors.WriteGGUF(*dst, meta, out))
	fmt.Printf("\n%.0f M weights at %.3f bits each — %s\n",
		count/1e6, bits/count, sizeOf(int(bits/8)))
	fmt.Printf("written to %s in %s\n", *dst, time.Since(t0).Round(time.Second))
}

// runCalib walks a text through the model in windows, with the tap set. The
// cache is forgotten between windows: each is its own context, which is what
// makes one long text into many independent samples.
func runCalib(path, text string, ntok, ctx int, hook func(int, string, [][]float32)) int {
	m, err := qwen.Open(path, ctx)
	must(err)
	defer m.Close()
	v, err := bytebpe.Load(m.File())
	must(err)
	ids := v.Encode(text, true, false)
	if len(ids) > ntok {
		ids = ids[:ntok]
	}
	qwen.Calib = hook
	defer func() { qwen.Calib = nil }()
	n := 0
	for start := 0; start+ctx <= len(ids); start += ctx {
		m.Reset()
		m.ForwardBatch(ids[start:start+ctx], 0)
		n += ctx
	}
	return n
}

// calibrate keeps, for each site, the per-column power of the activations that
// reach it. That is all the salience scaling needs, and it is what the second
// pass has to know before it can build anything.
func calibrate(path, text string, ntok, ctx, win, block int) (map[string][]float32, map[string]*compress.Acc) {
	sums := map[string][]float64{}
	seen := map[string]int{}
	accs := map[string]*compress.Acc{}
	t0 := time.Now()
	n := runCalib(path, text, ntok, ctx, func(blk int, site string, rows [][]float32) {
		key := fmt.Sprintf("%d/%s", blk, site)
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
		if win > 0 {
			// The Hessian in the basis the activations arrive in, which is the
			// one an error has to be brought back to before it is weighed.
			a := accs[key]
			if a == nil {
				a = compress.NewAcc(len(rows[0]), fitWindow(len(rows[0]), win, block))
				accs[key] = a
			}
			a.AddRows(rows)
		}
	})
	out := map[string][]float32{}
	for k, s := range sums {
		v := make([]float32, len(s))
		for j := range s {
			v[j] = float32(math.Sqrt(s[j] / float64(seen[k])))
		}
		out[k] = v
	}
	fmt.Printf("calibrated on %d tokens in %s, %d sites\n",
		n, time.Since(t0).Round(time.Millisecond), len(out))
	return out, accs
}

// hessians is the second pass: the same text again, with each activation put
// through the site's own vector and rotation before it is counted, so that what
// comes out is the Hessian the quantizer will actually meet.
func hessians(path, text string, ntok, ctx int, pre map[string][]float32,
	group int, embd string, want, block int, damp float64) map[string]*compress.Comp {
	accs := map[string]*compress.Acc{}
	buf := [][]float32{}
	t0 := time.Now()
	runCalib(path, text, ntok, ctx, func(blk int, site string, rows [][]float32) {
		key := fmt.Sprintf("%d/%s", blk, site)
		cols := len(rows[0])
		a := accs[key]
		if a == nil {
			a = compress.NewAcc(cols, fitWindow(cols, want, block))
			accs[key] = a
		}
		for len(buf) < len(rows) {
			buf = append(buf, make([]float32, cols))
		}
		use := buf[:len(rows)]
		for i, r := range rows {
			if len(use[i]) != cols {
				use[i] = make([]float32, cols)
			}
			copy(use[i], r)
			nn.PrepareD4G(use[i], pre[key], hadamardOf(key, group, embd))
		}
		a.AddRows(use)
	})
	out := map[string]*compress.Comp{}
	for k, a := range accs {
		out[k] = a.Comp(damp)
	}
	fmt.Printf("factored %d sites in %s\n", len(out), time.Since(t0).Round(time.Second))
	return out
}

// fitWindow shrinks a requested window until it divides the row and holds a
// whole number of scale blocks: a ragged window would either straddle a block
// or leave one uncompensated, and neither is worth the special case.
func fitWindow(cols, want, block int) int {
	if want > cols {
		want = cols
	}
	want -= want % block
	for w := want; w >= block; w -= block {
		if cols%w == 0 {
			return w
		}
	}
	return block
}

// hadamardOf says how wide the rotation is at a site, which is nothing for a
// table stored plain.
func hadamardOf(key string, group int, embd string) int {
	if key == headKey && embd != "rot" {
		return 0
	}
	return group
}

// saliencyScale turns per-column activation power into the factor the weights
// are multiplied by, normalised so the matrix keeps its overall size.
//
// clamp bounds how far it may go, and it is not a detail: the scale is applied
// before the rotation, and the rotation mixes a hundred and twenty-eight
// columns into each other. A column shrunk by two thousand is mixed with one
// left alone, quantized as if it were the second, and then multiplied back by
// two thousand on the activation side — so its error comes back two thousand
// times larger. At an exponent of 0.75 the span reaches eighteen thousand and
// the model reads at a perplexity of 246 rather than 40. Salience and
// incoherence do not compose freely, and this is where they are made to.
func saliencyScale(sal []float32, alpha, clamp float64) []float32 {
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
		s := math.Pow(math.Max(float64(v), 1e-8)/geo, alpha)
		if clamp > 1 {
			s = math.Min(math.Max(s, 1/clamp), clamp)
		}
		out[j] = float32(s)
	}
	return out
}

// kept says whether a tensor is one of those a probe is leaving alone, so that
// what the others cost can be read off on its own.
func kept(name, list string) bool {
	if list == "" {
		return false
	}
	for _, frag := range strings.Split(list, ",") {
		if frag != "" && strings.Contains(name, frag) {
			return true
		}
	}
	return false
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
