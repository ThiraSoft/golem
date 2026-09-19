package krea2

// The text encoder: Qwen3-VL-4B's language model, read out of ComfyUI's
// fp8-scaled checkpoint, run over the prompt with its template, and tapped
// where Krea 2 taps it — the hidden states entering layers 2, 5, ..., 35,
// raw, twelve of them side by side for every token of the user's text
// (comfy/text_encoders/krea2.py).
//
// A tap is the state entering a layer, so the last one enters layer 35 and
// layer 35 itself never runs, nor the final norm: ComfyUI computes both and
// Krea 2 reads neither.

import (
	"encoding/binary"
	"fmt"
	"math"

	"github.com/ThiraSoft/golem/tensors"
	"github.com/ThiraSoft/golem/vk"
)

// The text model's shape (comfy/text_encoders/llama.py, Qwen3VL_4BConfig).
const (
	encWidth   = 2560
	encHeads   = 32
	encKVHeads = 8
	encHeadDim = 128
	encFFN     = 9728
	encVocab   = 151936
	encEps     = 1e-6
	encTheta   = 5000000
	// MaxPromptTokens bounds the template and the prompt together; it sizes
	// the encoder's arena.
	MaxPromptTokens = 1024
)

// Taps are the layers Krea 2 reads the text model's state on entry to.
var Taps = [12]int{2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35}

// CondWidth is one token's conditioning: twelve taps of the model's width.
const CondWidth = len(Taps) * encWidth

type encLayer struct {
	q, k, v, o, gate, up, down int
	scale                      [7]float32
	inNorm, postNorm           uint32
	qNorm, kNorm               uint32
}

// Encoder is the text encoder on the card.
type Encoder struct {
	k      *vk.K2
	ck     *checkpoint
	embed  tensors.Tensor
	layers []encLayer

	// The arena, sized for MaxPromptTokens.
	x, n, q, kk, v, att, g, u, taps, table uint32
	programs                               map[int]*vk.Program
}

// OpenEncoder uploads the text model's first 35 layers to the card.
func OpenEncoder(d *vk.Device, path string) (*Encoder, error) {
	ck, err := openCheckpoint(path)
	if err != nil {
		return nil, err
	}
	e := &Encoder{ck: ck, programs: map[int]*vk.Program{}}
	fail := func(err error) (*Encoder, error) {
		e.Close()
		return nil, err
	}
	if e.embed, err = ck.get("model.embed_tokens.weight", "BF16", encVocab, encWidth); err != nil {
		return fail(err)
	}

	var par params
	layers := Taps[len(Taps)-1]
	norms := make([][4][]float32, layers)
	scales := make([][7]float32, layers)
	names := [7]string{"self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"}
	for i := 0; i < layers; i++ {
		pre := fmt.Sprintf("model.layers.%d.", i)
		for j, name := range []string{"input_layernorm", "post_attention_layernorm", "self_attn.q_norm", "self_attn.k_norm"} {
			n := encWidth
			if j >= 2 {
				n = encHeadDim
			}
			if norms[i][j], err = ck.floats(pre+name+".weight", n); err != nil {
				return fail(err)
			}
		}
		for j, name := range names {
			if scales[i][j], err = ck.scalar(pre + name + ".weight_scale"); err != nil {
				return fail(err)
			}
		}
	}
	e.layers = make([]encLayer, layers)
	for i := range e.layers {
		l := &e.layers[i]
		l.inNorm = par.add(norms[i][0])
		l.postNorm = par.add(norms[i][1])
		l.qNorm = par.add(norms[i][2])
		l.kNorm = par.add(norms[i][3])
		l.scale = scales[i]
	}

	var a arena
	T := MaxPromptTokens
	e.x = a.take(T * encWidth)
	e.n = a.take(T * encWidth)
	e.q = a.take(T * encHeads * encHeadDim)
	e.kk = a.take(T * encKVHeads * encHeadDim)
	e.v = a.take(T * encKVHeads * encHeadDim)
	e.att = a.take(T * encHeads * encHeadDim)
	e.g = a.take(T * encFFN)
	e.u = a.take(T * encFFN)
	e.taps = a.take(len(Taps) * T * encWidth)
	e.table = a.take(T * encHeadDim)

	if e.k, err = vk.NewK2(d, a.n, par.data); err != nil {
		return fail(err)
	}
	for i := range e.layers {
		l := &e.layers[i]
		pre := fmt.Sprintf("model.layers.%d.", i)
		shapes := [7][2]int{
			{encHeads * encHeadDim, encWidth}, {encKVHeads * encHeadDim, encWidth}, {encKVHeads * encHeadDim, encWidth},
			{encWidth, encHeads * encHeadDim}, {encFFN, encWidth}, {encFFN, encWidth}, {encWidth, encFFN},
		}
		handles := [7]*int{&l.q, &l.k, &l.v, &l.o, &l.gate, &l.up, &l.down}
		for j, name := range names {
			if *handles[j], err = ck.fp8(e.k, pre+name+".weight", shapes[j][0], shapes[j][1]); err != nil {
				return fail(err)
			}
		}
	}
	return e, nil
}

// program records the encoder for tokens positions.
func (e *Encoder) program(tokens int) (*vk.Program, error) {
	if p, ok := e.programs[tokens]; ok {
		return p, nil
	}
	T := uint32(tokens)
	qw, kw := uint32(encHeads*encHeadDim), uint32(encKVHeads*encHeadDim)
	mm := func(r *vk.Recorder, w int, scale float32, outs, ins, x, y uint32, mode uint32) {
		e.k.MM(r, w, true, vk.K2MM{Outputs: outs, Inputs: ins, Cols: T, X: x, XStride: ins, Y: y, YStride: outs,
			Bias: vk.K2None, Scale: scale, Mode: mode})
	}
	p, err := e.k.Compile(func(r *vk.Recorder) {
		tap := 0
		for i, l := range e.layers {
			if tap < len(Taps) && Taps[tap] == i {
				e.k.Act(r, vk.K2Act{X: e.taps + uint32(tap)*T*encWidth, XStride: encWidth, Y: e.x, YStride: encWidth, N: encWidth, Rows: T, Op: vk.K2Copy})
				tap++
			}
			e.k.Norm(r, vk.K2Norm{Src: e.x, SrcStride: encWidth, Dst: e.n, DstStride: encWidth, N: encWidth, Rows: T,
				Weight: l.inNorm, Eps: encEps, ScaleA: vk.K2None})
			mm(r, l.q, l.scale[0], qw, encWidth, e.n, e.q, vk.K2Store)
			mm(r, l.k, l.scale[1], kw, encWidth, e.n, e.kk, vk.K2Store)
			mm(r, l.v, l.scale[2], kw, encWidth, e.n, e.v, vk.K2Store)
			e.k.Norm(r, vk.K2Norm{Src: e.q, SrcStride: encHeadDim, Dst: e.q, DstStride: encHeadDim, N: encHeadDim, Rows: T * encHeads,
				Weight: l.qNorm, Eps: encEps, ScaleA: vk.K2None})
			e.k.Norm(r, vk.K2Norm{Src: e.kk, SrcStride: encHeadDim, Dst: e.kk, DstStride: encHeadDim, N: encHeadDim, Rows: T * encKVHeads,
				Weight: l.kNorm, Eps: encEps, ScaleA: vk.K2None})
			e.k.Rope(r, vk.K2Rope{X: e.q, Stride: qw, Heads: encHeads, Dim: encHeadDim, Tokens: T, Table: e.table, Halves: 1})
			e.k.Rope(r, vk.K2Rope{X: e.kk, Stride: kw, Heads: encKVHeads, Dim: encHeadDim, Tokens: T, Table: e.table, Halves: 1})
			e.k.Attn(r, vk.K2Attn{Q: e.q, QStride: qw, K: e.kk, KStride: kw, V: e.v, VStride: kw, O: e.att, OStride: qw,
				Queries: T, Keys: T, Group: encHeads / encKVHeads, Causal: 1, Scale: float32(1 / math.Sqrt(encHeadDim)),
				Heads: encHeads, Seqs: 1, HeadDim: encHeadDim})
			mm(r, l.o, l.scale[3], encWidth, qw, e.att, e.x, vk.K2Add)
			e.k.Norm(r, vk.K2Norm{Src: e.x, SrcStride: encWidth, Dst: e.n, DstStride: encWidth, N: encWidth, Rows: T,
				Weight: l.postNorm, Eps: encEps, ScaleA: vk.K2None})
			mm(r, l.gate, l.scale[4], encFFN, encWidth, e.n, e.g, vk.K2Store)
			mm(r, l.up, l.scale[5], encFFN, encWidth, e.n, e.u, vk.K2Store)
			e.k.Act(r, vk.K2Act{X: e.g, XStride: encFFN, Y: e.u, YStride: encFFN, N: encFFN, Rows: T, Op: vk.K2SwiGLU})
			mm(r, l.down, l.scale[6], encWidth, encFFN, e.g, e.x, vk.K2Add)
		}
		e.k.Act(r, vk.K2Act{X: e.taps + uint32(tap)*T*encWidth, XStride: encWidth, Y: e.x, YStride: encWidth, N: encWidth, Rows: T, Op: vk.K2Copy})
	})
	if err != nil {
		return nil, err
	}
	e.programs[tokens] = p
	return p, nil
}

// Encode runs the prompt's tokens and returns the conditioning Krea 2 reads:
// for each token from p.Start on, the twelve taps side by side.
func (e *Encoder) Encode(p Prompt) ([]float32, int, error) {
	T := len(p.IDs)
	if T > MaxPromptTokens {
		return nil, 0, fmt.Errorf("krea2: a prompt of %d tokens is past the %d the encoder holds", T, MaxPromptTokens)
	}
	if p.Start < 0 || p.Start >= T {
		return nil, 0, fmt.Errorf("krea2: the prompt's text starts at %d of %d tokens", p.Start, T)
	}
	x := make([]float32, T*encWidth)
	for i, id := range p.IDs {
		if id < 0 || int(id) >= encVocab {
			return nil, 0, fmt.Errorf("krea2: token %d is outside the vocabulary", id)
		}
		row := e.embed.Raw[int(id)*encWidth*2:]
		for j := 0; j < encWidth; j++ {
			x[i*encWidth+j] = math.Float32frombits(uint32(binary.LittleEndian.Uint16(row[2*j:])) << 16)
		}
	}
	if err := e.k.Write(int(e.x), x); err != nil {
		return nil, 0, err
	}
	if err := e.k.Write(int(e.table), encoderRope(T)); err != nil {
		return nil, 0, err
	}
	prog, err := e.program(T)
	if err != nil {
		return nil, 0, err
	}
	if err := prog.Run(); err != nil {
		return nil, 0, err
	}
	taps, err := e.k.Read(int(e.taps), len(Taps)*T*encWidth)
	if err != nil {
		return nil, 0, err
	}
	seq := T - p.Start
	cond := make([]float32, seq*CondWidth)
	for t := 0; t < seq; t++ {
		for j := range Taps {
			copy(cond[t*CondWidth+j*encWidth:], taps[(j*T+p.Start+t)*encWidth:(j*T+p.Start+t+1)*encWidth])
		}
	}
	return cond, seq, nil
}

// encoderRope is Qwen3's rotation table for positions 0..T-1, the way
// precompute_freqs_cis computes it: float32 inverse frequencies, float32
// angles.
func encoderRope(T int) []float32 {
	half := encHeadDim / 2
	inv := make([]float32, half)
	for i := range inv {
		inv[i] = 1 / float32(math.Pow(encTheta, float64(float32(2*i)/encHeadDim)))
	}
	out := make([]float32, T*encHeadDim)
	for t := 0; t < T; t++ {
		for i := 0; i < half; i++ {
			a := float64(inv[i] * float32(t))
			out[t*encHeadDim+i] = float32(math.Cos(a))
			out[t*encHeadDim+half+i] = float32(math.Sin(a))
		}
	}
	return out
}

// WeightBytes is what the encoder holds on the card.
func (e *Encoder) WeightBytes() int { return e.k.WeightBytes() }

// Close frees the card and the checkpoint.
func (e *Encoder) Close() {
	for _, p := range e.programs {
		p.Close()
	}
	e.programs = nil
	if e.k != nil {
		e.k.Close()
		e.k = nil
	}
	if e.ck != nil {
		e.ck.close()
		e.ck = nil
	}
}
