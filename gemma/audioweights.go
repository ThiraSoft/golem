package gemma

// Binding the audio encoder's tensors.
//
// The same file and the same rules as the vision side: bfloat16 or F32, no
// repacking, and four QAT scalars beside every projection. VisionLinear and
// Clamp are reused unchanged — they are exactly this file's need, and the name
// says which tower they were written for rather than which they belong to.
//
// Two names in the file are worth reading twice.
//
// `a.pre_encode.out` is not the front of the tower. It is the projection that
// comes after the twelve blocks, on its way to the embedder — clip.cpp binds
// it as audio_out_proj. The projection that flattens the subsampled mel into
// the tower's width is `a.input_projection`. The two read backwards from their
// names, and clip.cpp reads them this way round; so does this file.
//
// `conv_norm` and `norm_conv` are swapped too, and clip.cpp says so in a
// comment: the conversion script wrote each under the other's name. The norm
// before the pointwise expansion is `conv_norm`, the one after the depthwise
// convolution is `norm_conv`. They are bound here by role, and the roles are
// what the field names say.

import (
	"fmt"

	"github.com/ThiraSoft/golem/tensors"
)

// AudioConv is one of the two subsampling convolutions: a 3x3 kernel over
// In channels producing Out, and the gain of the layer norm that follows it.
//
// The file writes it column, row, input channel, output channel. Neither of
// the two ways the tower reads it wants that order, so both are built once at
// load: doing it per point of the grid would mean doing it fifty-six thousand
// times.
//
// Packed is output-channel-major, one contiguous row of KH*KW*In per output
// channel, which is a dot product against a gathered patch. PackedT is the
// transpose of it — one row of Out per tap — which is the other way to
// compute the same thing: scale each tap into every accumulator at once. A
// convolution with few taps and many channels wants the second, because a dot
// product of nine would spend its time being called.
type AudioConv struct {
	W               []float32 // KW*KH*In*Out, the first dimension fastest
	Packed          []float32 // Out by KH*KW*In, the input channel fastest
	PackedT         []float32 // KH*KW*In by Out
	Norm            []float32 // Out gains; the norm has no bias
	KW, KH, In, Out int
}

// pack reorders a convolution's weight for the gather the tower does.
func (c *AudioConv) pack() {
	taps := c.KH * c.KW * c.In
	c.Packed = make([]float32, c.Out*taps)
	c.PackedT = make([]float32, c.Out*taps)
	for o := 0; o < c.Out; o++ {
		for ic := 0; ic < c.In; ic++ {
			for kh := 0; kh < c.KH; kh++ {
				for kw := 0; kw < c.KW; kw++ {
					tap := (kh*c.KW+kw)*c.In + ic
					v := c.W[((o*c.In+ic)*c.KH+kh)*c.KW+kw]
					c.Packed[o*taps+tap] = v
					c.PackedT[tap*c.Out+o] = v
				}
			}
		}
	}
}

// AudioLinear is a projection whose weight the file stores as F32 rather than
// bfloat16. One tensor in this tower is like that — a.input_projection — and
// giving it a type of its own is cheaper than teaching VisionLinear a second
// element size for a matrix that is multiplied once per recording.
type AudioLinear struct {
	W               []float32 // Outputs x Inputs, row-major
	Inputs, Outputs int
	Clamp           Clamp
}

type AudioBlock struct {
	// The first half-step feed forward, and the second. No gate: the graph
	// calls build_ffn with three null arguments, which is SiLU on the up
	// projection alone rather than SwiGLU.
	FFNNorm, FFNPostNorm   []float32
	FFNUp, FFNDown         VisionLinear
	FFNNorm1, FFNPostNorm1 []float32
	FFNUp1, FFNDown1       VisionLinear

	AttnPreNorm, AttnPostNorm []float32
	Q, K, V, Out              VisionLinear
	KRel                      VisionLinear // attn_k_rel, over the position rows
	PerDimScale               []float32    // one gain per dimension of a head

	// The convolution module, in the order it runs: a norm, a pointwise
	// expansion to twice the width, a gated halving, a causal depthwise
	// convolution of width five, a second norm, and a pointwise return.
	ConvPreNorm   []float32 // the file calls this one conv_norm
	ConvPW1       VisionLinear
	ConvDW        []float32 // 5 taps per channel, the tap the faster index
	ConvInnerNorm []float32 // the file calls this one norm_conv
	ConvPW2       VisionLinear

	BlockNorm []float32 // ln2, the block's own output norm
}

type AudioWeights struct {
	Conv      [2]AudioConv
	InputProj AudioLinear // a.input_projection: the flattened mel into the width

	Blocks []AudioBlock

	OutProj     VisionLinear // a.pre_encode.out
	OutProjBias []float32    // added after the projection's output clamp, not before
	SoftEmbNorm []float32    // mm.a.soft_emb_norm, absent from E2B's projector
	MMProj      VisionLinear // mm.a.input_projection
}

// LoadAudioWeights binds every tensor the encoder computes with. The unified
// projector has one of them and this returns as soon as it is bound.
func LoadAudioWeights(g *tensors.GGUF, cfg *AudioConfig) (*AudioWeights, error) {
	w := &AudioWeights{}
	var err error

	if w.MMProj, err = linear(g, "mm.a.input_projection.weight"); err != nil {
		return nil, err
	}
	if w.MMProj.Outputs != cfg.ProjDim {
		return nil, fmt.Errorf("the audio projection gives %d-wide rows, expected %d", w.MMProj.Outputs, cfg.ProjDim)
	}
	if cfg.Unified {
		if w.MMProj.Inputs != cfg.FrameSize {
			return nil, fmt.Errorf("the audio projection reads %d values, expected a frame of %d", w.MMProj.Inputs, cfg.FrameSize)
		}
		return w, nil
	}

	for i := range w.Conv {
		p := fmt.Sprintf("a.conv1d.%d.", i)
		t, ok := g.Tensors[p+"weight"]
		if !ok {
			return nil, fmt.Errorf("tensor %q is absent", p+"weight")
		}
		if len(t.Shape) != 4 {
			return nil, fmt.Errorf("tensor %q has %d dimensions, expected four", p+"weight", len(t.Shape))
		}
		c := AudioConv{KW: t.Shape[0], KH: t.Shape[1], In: t.Shape[2], Out: t.Shape[3]}
		if c.W, err = floats(g, p+"weight"); err != nil {
			return nil, err
		}
		if c.Norm, err = floats(g, p+"norm.weight"); err != nil {
			return nil, err
		}
		if len(c.Norm) != c.Out {
			return nil, fmt.Errorf("%snorm.weight has %d gains for %d channels", p, len(c.Norm), c.Out)
		}
		c.pack()
		w.Conv[i] = c
	}
	if w.InputProj, err = audioLinear(g, "a.input_projection.weight"); err != nil {
		return nil, err
	}
	if w.InputProj.Outputs != cfg.Dim {
		return nil, fmt.Errorf("the input projection gives %d, expected the tower's %d", w.InputProj.Outputs, cfg.Dim)
	}

	w.Blocks = make([]AudioBlock, cfg.Blocks)
	for i := range w.Blocks {
		p := fmt.Sprintf("a.blk.%d.", i)
		b := AudioBlock{}
		for _, f := range []struct {
			name string
			dst  *[]float32
		}{
			{p + "ffn_norm.weight", &b.FFNNorm},
			{p + "ffn_post_norm.weight", &b.FFNPostNorm},
			{p + "ffn_norm_1.weight", &b.FFNNorm1},
			{p + "ffn_post_norm_1.weight", &b.FFNPostNorm1},
			{p + "attn_pre_norm.weight", &b.AttnPreNorm},
			{p + "attn_post_norm.weight", &b.AttnPostNorm},
			{p + "per_dim_scale.weight", &b.PerDimScale},
			{p + "conv_norm.weight", &b.ConvPreNorm},
			{p + "norm_conv.weight", &b.ConvInnerNorm},
			{p + "conv_dw.weight", &b.ConvDW},
			{p + "ln2.weight", &b.BlockNorm},
		} {
			if *f.dst, err = floats(g, f.name); err != nil {
				return nil, err
			}
		}
		for _, f := range []struct {
			name string
			dst  *VisionLinear
		}{
			{p + "ffn_up.weight", &b.FFNUp},
			{p + "ffn_down.weight", &b.FFNDown},
			{p + "ffn_up_1.weight", &b.FFNUp1},
			{p + "ffn_down_1.weight", &b.FFNDown1},
			{p + "attn_q.weight", &b.Q},
			{p + "attn_k.weight", &b.K},
			{p + "attn_v.weight", &b.V},
			{p + "attn_out.weight", &b.Out},
			{p + "attn_k_rel.weight", &b.KRel},
			{p + "conv_pw1.weight", &b.ConvPW1},
			{p + "conv_pw2.weight", &b.ConvPW2},
		} {
			if *f.dst, err = linear(g, f.name); err != nil {
				return nil, err
			}
		}
		if b.Q.Inputs != cfg.Dim || b.Q.Outputs != cfg.Dim {
			return nil, fmt.Errorf("block %d: query is %dx%d, expected %d square", i, b.Q.Outputs, b.Q.Inputs, cfg.Dim)
		}
		if b.FFNUp.Outputs != cfg.FFN || b.FFNDown.Inputs != cfg.FFN {
			return nil, fmt.Errorf("block %d: feed forward is %d wide, expected %d", i, b.FFNUp.Outputs, cfg.FFN)
		}
		if len(b.PerDimScale) != cfg.HeadDim {
			return nil, fmt.Errorf("block %d: per-dimension scale has %d entries for a head of %d", i, len(b.PerDimScale), cfg.HeadDim)
		}
		if len(b.ConvDW) != 5*cfg.Dim {
			return nil, fmt.Errorf("block %d: depthwise convolution has %d weights, expected five per channel of %d", i, len(b.ConvDW), cfg.Dim)
		}
		if b.ConvPW1.Outputs != 2*cfg.Dim {
			return nil, fmt.Errorf("block %d: the pointwise expansion gives %d, expected twice %d", i, b.ConvPW1.Outputs, cfg.Dim)
		}
		w.Blocks[i] = b
	}

	if w.OutProj, err = linear(g, "a.pre_encode.out.weight"); err != nil {
		return nil, err
	}
	// The bias is kept beside the projection rather than inside it. clip.cpp
	// adds it after build_mm has clamped the output, and a bias folded into
	// the product would be clamped along with it.
	if w.OutProjBias, err = floats(g, "a.pre_encode.out.bias"); err != nil {
		return nil, err
	}
	if w.OutProj.Inputs != cfg.Dim || w.OutProj.Outputs != w.MMProj.Inputs {
		return nil, fmt.Errorf("the output projection is %dx%d, expected %d into the embedder's %d",
			w.OutProj.Outputs, w.OutProj.Inputs, cfg.Dim, w.MMProj.Inputs)
	}
	// Optional: E2B's projector carries no soft-embedding gain, and the graph
	// then normalizes without scaling.
	if gain, err := floats(g, "mm.a.soft_emb_norm.weight"); err == nil {
		w.SoftEmbNorm = gain
	}
	return w, nil
}

// audioLinear binds an F32 matrix with its QAT ranges.
func audioLinear(g *tensors.GGUF, name string) (AudioLinear, error) {
	t, ok := g.Tensors[name]
	if !ok {
		return AudioLinear{}, fmt.Errorf("tensor %q is absent", name)
	}
	if len(t.Shape) != 2 {
		return AudioLinear{}, fmt.Errorf("tensor %q has %d dimensions", name, len(t.Shape))
	}
	v, err := floats(g, name)
	if err != nil {
		return AudioLinear{}, err
	}
	return AudioLinear{W: v, Inputs: t.Shape[0], Outputs: t.Shape[1], Clamp: clampOf(g, name)}, nil
}
