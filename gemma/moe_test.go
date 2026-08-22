package gemma

// The mixture, against the obvious way of computing it.
//
// A fixture says that two implementations agree with a recording. This file
// says what the arithmetic is: every expert computed, the chosen ones weighted
// and added. It needs no checkpoint and no llama.cpp, so it fails on a laptop
// with nothing on disk, which is where a routing bug is cheapest to find.

import (
	"encoding/binary"
	"math"
	"math/rand"
	"testing"

	"github.com/ThiraSoft/golem/nn"
)

// randomMoEBlock builds a mixture block whose weights are F32 and
// reproducible, so that the oracle and the engine read the same numbers.
func randomMoEBlock(cfg *Config, seed int64) *BlockWeights {
	r := rand.New(rand.NewSource(seed))
	f32 := func(n int) []byte {
		b := make([]byte, 4*n)
		for i := 0; i < n; i++ {
			binary.LittleEndian.PutUint32(b[4*i:], math.Float32bits(float32(r.NormFloat64()*0.1)))
		}
		return b
	}
	gains := func(n int) []float32 {
		g := make([]float32, n)
		for i := range g {
			g[i] = 1 + float32(r.NormFloat64()*0.05)
		}
		return g
	}
	stack := func(rows, cols int) ExpertStack {
		return ExpertStack{
			Data: f32(rows * cols * cfg.Experts), Quant: nn.F32,
			Rows: rows, Cols: cols, Count: cfg.Experts,
		}
	}
	return &BlockWeights{
		OutScale:     1,
		Router:       nn.Matrix{Data: f32(cfg.Experts * cfg.Dim), Quant: nn.F32, Rows: cfg.Experts, Cols: cfg.Dim},
		RouterScale:  gains(cfg.Dim),
		PreFFWNorm2:  gains(cfg.Dim),
		PostFFWNorm:  gains(cfg.Dim),
		PostFFWNorm1: gains(cfg.Dim),
		PostFFWNorm2: gains(cfg.Dim),
		GateUpExps:   stack(2*cfg.ExpertFFN, cfg.Dim),
		DownExps:     stack(cfg.Dim, cfg.ExpertFFN),
		DownScale:    gains(cfg.Experts),
	}
}

func moeConfig() *Config {
	return &Config{
		Dim: 32, Eps: 1e-6,
		Experts: 6, ExpertsUsed: 2, ExpertFFN: 64,
		Blocks: []BlockConfig{{Index: 0, MoE: true, FFN: 64, Heads: 2, KVHeads: 1, HeadDim: 16}},
	}
}

// denseOracle is what the mixture means, written the slow and obvious way.
func denseOracle(cfg *Config, bw *BlockWeights, resid, in []float32) []float32 {
	// The router, by hand: an RMS norm with no gain, one over the square root
	// of the width, and the scale vector.
	x := append([]float32(nil), resid...)
	nn.RMSNormPlain(x, nil, cfg.Eps)
	scale := float32(1 / math.Sqrt(float64(cfg.Dim)))
	for i := range x {
		x[i] *= scale * bw.RouterScale[i]
	}
	logits := make([]float32, cfg.Experts)
	for e := 0; e < cfg.Experts; e++ {
		w := row(bw.Router.Data, cfg.Dim, e)
		for i := range x {
			logits[e] += w[i] * x[i]
		}
	}

	ids := make([]int32, cfg.ExpertsUsed)
	weights := make([]float32, cfg.ExpertsUsed)
	chooseExperts(cfg, logits, ids, weights)

	out := make([]float32, cfg.Dim)
	for k, id := range ids {
		gate := make([]float32, cfg.ExpertFFN)
		up := make([]float32, cfg.ExpertFFN)
		// The gate is the expert's first ExpertFFN rows, the up the next.
		gu := bw.GateUpExps.At(int(id))
		for r := 0; r < cfg.ExpertFFN; r++ {
			gw, uw := row(gu.Data, cfg.Dim, r), row(gu.Data, cfg.Dim, r+cfg.ExpertFFN)
			for i := range in {
				gate[r] += gw[i] * in[i]
				up[r] += uw[i] * in[i]
			}
		}
		nn.GELUTable(gate)
		for i := range gate {
			gate[i] *= up[i]
		}
		d := bw.DownExps.At(int(id))
		for r := 0; r < cfg.Dim; r++ {
			dw := row(d.Data, cfg.ExpertFFN, r)
			var acc float32
			for i := range gate {
				acc += dw[i] * gate[i]
			}
			out[r] += weights[k] * bw.DownScale[id] * acc
		}
	}
	return out
}

// row reads one row of an F32 matrix, which is all the oracle needs and all
// the engine's Matrix does not offer: every product in nn goes through a
// quantized batch, and the point of the oracle is to go nowhere near one.
func row(data []byte, cols, r int) []float32 {
	out := make([]float32, cols)
	for i := range out {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(data[4*(r*cols+i):]))
	}
	return out
}

func wave(n int, phase float64) []float32 {
	x := make([]float32, n)
	for i := range x {
		x[i] = float32(math.Sin(float64(i)*0.7 + phase))
	}
	return x
}

// TestExpertFFNMatchesTheOracle is the check the design turns on.
func TestExpertFFNMatchesTheOracle(t *testing.T) {
	cfg := moeConfig()
	bw := randomMoEBlock(cfg, 1)
	s := NewScratch(cfg)

	in := s.ExpertIn(0) // borrowed only to hold a properly shaped input
	_ = in
	branch := nn.NewBatch(cfg.Dim, 1)
	branch.Set(0, wave(cfg.Dim, 0))
	// The router's input is not the branch's input, and the whole point of
	// reading them apart is that they differ.
	copy(s.resid[0], wave(cfg.Dim, 1.3))

	want := denseOracle(cfg, bw, s.resid[0], branch.F[0])
	out := [][]float32{make([]float32, cfg.Dim)}
	ExpertFFN(cfg, bw, s, branch, out)

	for i := range want {
		if d := math.Abs(float64(want[i] - out[0][i])); d > 1e-4 {
			t.Fatalf("element %d: the engine says %v, the oracle says %v", i, out[0][i], want[i])
		}
	}
}

// TestExpertFFNBatchMatchesOnePosition pins that a batch is the same
// arithmetic as the positions taken one at a time — the property prefill
// depends on, and the one a shared buffer breaks first.
func TestExpertFFNBatchMatchesOnePosition(t *testing.T) {
	cfg := moeConfig()
	bw := randomMoEBlock(cfg, 2)

	const batch = 5
	inputs := make([][]float32, batch)
	resids := make([][]float32, batch)
	for t0 := range inputs {
		inputs[t0] = wave(cfg.Dim, float64(t0)*0.31)
		resids[t0] = wave(cfg.Dim, float64(t0)*0.91+2)
	}

	s := NewScratch(cfg)
	s.Reserve(batch)
	together := nn.NewBatch(cfg.Dim, batch)
	for t0 := range inputs {
		together.Set(t0, inputs[t0])
		copy(s.resid[t0], resids[t0])
	}
	got := make([][]float32, batch)
	for i := range got {
		got[i] = make([]float32, cfg.Dim)
	}
	ExpertFFN(cfg, bw, s, together, got)

	for t0 := 0; t0 < batch; t0++ {
		one := NewScratch(cfg)
		copy(one.resid[0], resids[t0])
		single := nn.NewBatch(cfg.Dim, 1)
		single.Set(0, inputs[t0])
		alone := [][]float32{make([]float32, cfg.Dim)}
		ExpertFFN(cfg, bw, one, single, alone)
		for i := range alone[0] {
			if d := math.Abs(float64(alone[0][i] - got[t0][i])); d > 1e-5 {
				t.Fatalf("position %d element %d: %v in a batch of five, %v alone",
					t0, i, got[t0][i], alone[0][i])
			}
		}
	}
}

// TestChooseExpertsRenormalizes pins the softmax: over the chosen few alone,
// summing to one, largest first.
func TestChooseExpertsRenormalizes(t *testing.T) {
	cfg := &Config{Experts: 5, ExpertsUsed: 3}
	logits := []float32{1, 4, 0, 3, 2}
	ids := make([]int32, cfg.ExpertsUsed)
	weights := make([]float32, cfg.ExpertsUsed)
	chooseExperts(cfg, logits, ids, weights)

	if ids[0] != 1 || ids[1] != 3 || ids[2] != 4 {
		t.Fatalf("chose %v, expected 1, 3 then 4", ids)
	}
	sum := float32(0)
	for _, w := range weights {
		sum += w
	}
	if math.Abs(float64(sum)-1) > 1e-6 {
		t.Fatalf("the weights sum to %v, not to one", sum)
	}
	// Renormalized over three, not over five: the ratios are the exponentials
	// of the differences, and the sum makes them weights.
	want := math.Exp(4) / (math.Exp(4) + math.Exp(3) + math.Exp(2))
	if math.Abs(float64(weights[0])-want) > 1e-6 {
		t.Fatalf("the first weight is %v, expected %v", weights[0], want)
	}
}

// TestChooseExpertsIsStableInTies keeps a tie going to the lower index, so
// that a comparison against a recording means something.
func TestChooseExpertsIsStableInTies(t *testing.T) {
	cfg := &Config{Experts: 4, ExpertsUsed: 2}
	logits := []float32{2, 2, 2, 1}
	for run := 0; run < 8; run++ {
		ids := make([]int32, cfg.ExpertsUsed)
		chooseExperts(cfg, logits, ids, make([]float32, cfg.ExpertsUsed))
		if ids[0] != 0 || ids[1] != 1 {
			t.Fatalf("run %d chose %v; a tie must go to the lower index", run, ids)
		}
	}
}

// TestExpertFFNDoesNotAllocate is the constraint the engine is written under,
// checked where it is easiest to break: a buffer taken with make inside the
// loop would cost a hundred thousand allocations a generation.
//
// The floor is three, and they are not this file's doing: a closure handed to
// nn.InParallel is one allocation, the whole engine pays it in every section,
// and there are three sections here — the router's norm, the router's product,
// and the experts.
func TestExpertFFNDoesNotAllocate(t *testing.T) {
	cfg := moeConfig()
	bw := randomMoEBlock(cfg, 3)
	s := NewScratch(cfg)
	in := nn.NewBatch(cfg.Dim, 1)
	in.Set(0, wave(cfg.Dim, 0))
	out := [][]float32{make([]float32, cfg.Dim)}
	ExpertFFN(cfg, bw, s, in, out) // the first call fills the lazy buffers
	const sections = 3
	if n := testing.AllocsPerRun(20, func() { ExpertFFN(cfg, bw, s, in, out) }); n > sections {
		t.Fatalf("%v allocations per call, above the %d parallel sections", n, sections)
	}
}

// TestMoEHalfAddsBothBranches pins the shape of the mixture block: the dense
// feed forward and the experts leave the same residual, each through its own
// pair of norms, and the block's output is the sum of the two — not one after
// the other, and not the experts alone.
func TestMoEHalfAddsBothBranches(t *testing.T) {
	cfg := moeConfig()
	bw := randomMoEBlock(cfg, 4)
	// The shared expert is the ordinary feed forward, so this block needs the
	// dense projections too.
	dense := randomMoEBlock(cfg, 5)
	bw.Gate = nn.Matrix{Data: dense.GateUpExps.Data[:4*cfg.Blocks[0].FFN*cfg.Dim], Quant: nn.F32, Rows: cfg.Blocks[0].FFN, Cols: cfg.Dim}
	bw.Up = nn.Matrix{Data: dense.GateUpExps.Data[4*cfg.Blocks[0].FFN*cfg.Dim : 8*cfg.Blocks[0].FFN*cfg.Dim], Quant: nn.F32, Rows: cfg.Blocks[0].FFN, Cols: cfg.Dim}
	bw.Down = nn.Matrix{Data: dense.DownExps.Data[:4*cfg.Dim*cfg.Blocks[0].FFN], Quant: nn.F32, Rows: cfg.Dim, Cols: cfg.Blocks[0].FFN}

	s := NewScratch(cfg)
	resid := wave(cfg.Dim, 0.2)
	copy(s.resid[0], resid)

	normed := nn.NewBatch(cfg.Dim, 1)
	normed.Set(0, wave(cfg.Dim, 0.9))

	xs := [][]float32{make([]float32, cfg.Dim)}
	moeHalf(cfg, cfg.Blocks[0], bw, s, xs, normed, 1)

	shared, experts := s.MoEShared(), s.MoEExperts()
	if allZero(shared) {
		t.Fatal("the shared expert contributed nothing")
	}
	if allZero(experts) {
		t.Fatal("the expert branch contributed nothing")
	}
	same := true
	for i := range shared {
		if shared[i] != experts[i] {
			same = false
			break
		}
	}
	if same {
		t.Fatal("the two branches computed the same thing; one is reading the other's buffer")
	}
	// The sum of the two branches goes through the block's own post-norm
	// before joining the stream — the norm a dense block applies to its one
	// branch. Leaving it out is a block whose every inner waypoint matches the
	// reference and whose output does not, so the identity is pinned here.
	combined := make([]float32, cfg.Dim)
	for i := range combined {
		combined[i] = shared[i] + experts[i]
	}
	nn.RMSNormPlain(combined, bw.PostFFWNorm, cfg.Eps)
	for i := range xs[0] {
		want := resid[i] + combined[i]
		if d := math.Abs(float64(want - xs[0][i])); d > 1e-5 {
			t.Fatalf("element %d: the block gave %v, the branches under post_ffw_norm give %v",
				i, xs[0][i], want)
		}
	}
}

func allZero(x []float32) bool {
	for _, v := range x {
		if v != 0 {
			return false
		}
	}
	return true
}
