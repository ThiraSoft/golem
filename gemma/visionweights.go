package gemma

// Binding the projector file's tensors.
//
// Everything here is bfloat16 or F32 and none of it is repacked: the tower
// runs once per image, not once per token, and the layout that pays for itself
// in the text model would buy nothing here.
//
// Each weight in the file is shadowed by four scalars — input_min, input_max,
// output_min, output_max — the clipping ranges the model was quantized under.
// llama.cpp applies them to the projection alone (build_mm in
// models/gemma4v.cpp), and so does this file.

import (
	"fmt"
	"math"

	"github.com/ThiraSoft/golem/nn"
	"github.com/ThiraSoft/golem/tensors"
)

// Clamp is one Gemma4ClippableLinear's two ranges.
type Clamp struct{ InMin, InMax, OutMin, OutMax float32 }

// VisionLinear is a projection that was quantization-aware trained: the input
// is held inside the range the weight was fitted for, and so is the output.
//
// Every product in this tower is one of these. clip.cpp says so by overriding
// build_mm for gemma4v, which is the function build_vit calls for the query,
// the key, the value, the output projection and all three of the feed
// forward's — not only for the final projection, which is what the four
// scalars next to each weight in the file already suggested.
type VisionLinear struct {
	nn.Linear
	Clamp Clamp
}

// PrepareInput writes into tmp the activation the products below read:
// clamped to the range the weights were fitted for, rounded to bfloat16, and
// laid out the way the kernel widens its weights.
//
// It is a step of its own because several projections share it. The file gives
// every weight its own input range, and the query, the key and the value turn
// out to have the same one — they see the same activation, so they were
// calibrated on the same numbers — as do the two halves of the feed forward's
// gate. Preparing it once for the three of them is one pass over the grid
// instead of three.
//
// The rounding to bfloat16 is not an economy. ggml's kernel for a bfloat16
// weight takes both operands in bfloat16 — vec_dot_type is BF16 — so an
// activation that kept its float32 mantissa here would not be more accurate
// than the reference, it would be a thousandth away from it, and a thousandth
// compounded over sixteen blocks is visible at the end.
func (l VisionLinear) PrepareInput(x, tmp []float32, batch int) {
	lo, hi := l.Clamp.InMin, l.Clamp.InMax
	interleaved := nn.Interleaved(l.Inputs)
	nn.InParallel(batch, batch*l.Inputs, func(first, last int) {
		from, to := first*l.Inputs, last*l.Inputs
		if interleaved && nn.PrepareIlv(tmp[from:to], x[from:to], lo, hi) {
			return
		}
		for i := from; i < to; i++ {
			tmp[i] = nn.RoundBF16(clampF32(x[i], lo, hi))
		}
	})
}

// rows computes rows [start, end) of Y = clamp(W * tmp) on the caller's
// thread, for an input PrepareInput has already written.
func (l VisionLinear) rows(tmp, y []float32, batch, start, end int) {
	in, out := batch*l.Inputs, batch*l.Outputs
	if nn.Interleaved(l.Inputs) {
		nn.MatMatBF16IlvRows(l.Weights, tmp[:in], l.Outputs, l.Inputs, batch, y[:out], start, end)
	} else {
		l.ApplyRows(tmp[:in], y[:out], batch, start, end)
	}
	// The rows this core produced, clamped before the barrier: a pass over the
	// outputs afterwards would be a pass on one core.
	lo, hi := l.Clamp.OutMin, l.Clamp.OutMax
	for c := 0; c < batch; c++ {
		nn.Clamp(y[c*l.Outputs+start:c*l.Outputs+end], lo, hi)
	}
}

// Apply is one projection over a whole batch: the input prepared, then the
// product.
//
// The batch is the point. A projection read one vector at a time reads its
// weight from memory for every vector, and this tower has a thousand of them
// per block: the same matrix would be pulled across the bus a thousand times.
// Read once for the batch, it is pulled once. The split is over the output
// rows, so each core holds a panel of the weight small enough to stay in its
// own cache while the whole batch streams past it.
func (l VisionLinear) Apply(x, tmp, y []float32, batch int) {
	l.PrepareInput(x, tmp, batch)
	nn.InParallel(l.Outputs, batch*l.Inputs*l.Outputs, func(start, end int) {
		l.rows(tmp, y, batch, start, end)
	})
}

// ApplyShared is several projections of the same prepared input, in one pass.
//
// One pass rather than one each because the cores meet at the end of every
// one of them, and three projections that could have been one are two
// meetings nobody needed. The rows of all of them are laid end to end and
// shared out as if they were a single taller matrix.
func ApplyShared(ls []VisionLinear, x, tmp []float32, ys [][]float32, batch int) {
	total, work := 0, 0
	for _, l := range ls {
		total += l.Outputs
		work += batch * l.Inputs * l.Outputs
	}
	ls[0].PrepareInput(x, tmp, batch)
	nn.InParallel(total, work, func(start, end int) {
		base := 0
		for i, l := range ls {
			from, to := max(start, base), min(end, base+l.Outputs)
			if from < to {
				l.rows(tmp, ys[i], batch, from-base, to-base)
			}
			base += l.Outputs
		}
	})
}

func clampF32(v, lo, hi float32) float32 {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}

type VisionBlockWeights struct {
	LN1, LN2                  []float32
	QNorm, KNorm              []float32
	Q, K, V, O                VisionLinear
	PostAttnNorm, PostFFNNorm []float32
	Gate, Up, Down            VisionLinear
}

type VisionWeights struct {
	PatchEmbd  []float32
	PosX, PosY []float32
	Positions  int
	Blocks     []VisionBlockWeights
	Proj       VisionLinear
	StdBias    []float32
	StdScale   []float32
}

// linear binds a BF16 matrix as a projection with no bias. GGUF writes the row
// length first, so Shape[0] counts inputs and Shape[1] outputs — the same
// convention gemma/weights.go reads.
func linear(g *tensors.GGUF, name string) (VisionLinear, error) {
	t, ok := g.Tensors[name]
	if !ok {
		return VisionLinear{}, fmt.Errorf("tensor %q is absent", name)
	}
	if t.DType != "BF16" {
		return VisionLinear{}, fmt.Errorf("tensor %q is %s; the projector is expected to be BF16", name, t.DType)
	}
	if len(t.Shape) != 2 {
		return VisionLinear{}, fmt.Errorf("tensor %q has %d dimensions", name, len(t.Shape))
	}
	return VisionLinear{
		Linear: nn.Linear{Weights: t.Raw, Inputs: t.Shape[0], Outputs: t.Shape[1]},
		Clamp:  clampOf(g, name),
	}, nil
}

// scalar reads one of the QAT range tensors, which are one-element F32. A
// range the file does not carry is no range at all, which is what the infinite
// fallback says.
func scalar(g *tensors.GGUF, name string, fallback float32) float32 {
	v, err := floats(g, name)
	if err != nil || len(v) != 1 {
		return fallback
	}
	return v[0]
}

func clampOf(g *tensors.GGUF, weight string) Clamp {
	base := weight[:len(weight)-len(".weight")]
	inf := float32(math.Inf(1))
	return Clamp{
		InMin:  scalar(g, base+".input_min", -inf),
		InMax:  scalar(g, base+".input_max", inf),
		OutMin: scalar(g, base+".output_min", -inf),
		OutMax: scalar(g, base+".output_max", inf),
	}
}

// LoadVisionWeights binds every tensor the tower computes with.
func LoadVisionWeights(g *tensors.GGUF, cfg *VisionConfig) (*VisionWeights, error) {
	w := &VisionWeights{Blocks: make([]VisionBlockWeights, cfg.Blocks)}

	var err error
	if w.PatchEmbd, err = floats(g, "v.patch_embd.weight"); err != nil {
		return nil, err
	}
	if n := cfg.PatchSize * cfg.PatchSize * 3 * cfg.Dim; len(w.PatchEmbd) != n {
		return nil, fmt.Errorf("the patch embedding holds %d floats, expected %d", len(w.PatchEmbd), n)
	}

	// One tensor holding two tables: the x lookup, then the y lookup.
	pos, err := floats(g, "v.position_embd.weight")
	if err != nil {
		return nil, err
	}
	if len(pos)%(2*cfg.Dim) != 0 {
		return nil, fmt.Errorf("the position table holds %d floats, which is not two tables of %d-wide rows", len(pos), cfg.Dim)
	}
	w.Positions = len(pos) / (2 * cfg.Dim)
	w.PosX = pos[:w.Positions*cfg.Dim]
	w.PosY = pos[w.Positions*cfg.Dim:]

	for i := range w.Blocks {
		p := fmt.Sprintf("v.blk.%d.", i)
		b := VisionBlockWeights{}
		for _, f := range []struct {
			name string
			dst  *[]float32
		}{
			{p + "ln1.weight", &b.LN1},
			{p + "ln2.weight", &b.LN2},
			{p + "attn_q_norm.weight", &b.QNorm},
			{p + "attn_k_norm.weight", &b.KNorm},
			{p + "attn_post_norm.weight", &b.PostAttnNorm},
			{p + "ffn_post_norm.weight", &b.PostFFNNorm},
		} {
			if *f.dst, err = floats(g, f.name); err != nil {
				return nil, err
			}
		}
		for _, f := range []struct {
			name string
			dst  *VisionLinear
		}{
			{p + "attn_q.weight", &b.Q},
			{p + "attn_k.weight", &b.K},
			{p + "attn_v.weight", &b.V},
			{p + "attn_out.weight", &b.O},
			{p + "ffn_gate.weight", &b.Gate},
			{p + "ffn_up.weight", &b.Up},
			{p + "ffn_down.weight", &b.Down},
		} {
			if *f.dst, err = linear(g, f.name); err != nil {
				return nil, err
			}
		}
		if b.Q.Inputs != cfg.Dim || b.Q.Outputs != cfg.Dim {
			return nil, fmt.Errorf("block %d: query is %dx%d, expected %dx%d",
				i, b.Q.Outputs, b.Q.Inputs, cfg.Dim, cfg.Dim)
		}
		if b.Gate.Outputs != cfg.FFN || b.Down.Inputs != cfg.FFN {
			return nil, fmt.Errorf("block %d: feed forward is %d wide, expected %d", i, b.Gate.Outputs, cfg.FFN)
		}
		if len(b.QNorm) != cfg.HeadDim {
			return nil, fmt.Errorf("block %d: query norm has %d entries for a head of %d", i, len(b.QNorm), cfg.HeadDim)
		}
		w.Blocks[i] = b
	}

	if w.Proj, err = linear(g, "mm.input_projection.weight"); err != nil {
		return nil, err
	}
	if w.Proj.Inputs != cfg.Dim || w.Proj.Outputs != cfg.ProjDim {
		return nil, fmt.Errorf("the projection is %dx%d, expected %dx%d",
			w.Proj.Outputs, w.Proj.Inputs, cfg.ProjDim, cfg.Dim)
	}

	// Optional, and llama.cpp treats them as a pair: both or neither. E2B's
	// projector carries neither, and the standardisation is then not a step.
	bias, errB := floats(g, "v.std_bias")
	scale, errS := floats(g, "v.std_scale")
	if errB == nil && errS == nil {
		w.StdBias, w.StdScale = bias, scale
	}
	return w, nil
}
