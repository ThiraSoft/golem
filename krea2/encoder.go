package krea2

// The text encoder: Qwen3-VL-4B's language model, read out of ComfyUI's
// fp8-scaled checkpoint, run over the prompt with its template, and tapped
// where Krea 2 taps it -- the hidden states entering layers 2, 5, ..., 35,
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
	"github.com/ThiraSoft/golem/token/bytebpe"
	"github.com/ThiraSoft/golem/vk"
)

// Tokenizer is Qwen2's byte-level BPE tokenizer.
type Tokenizer = bytebpe.Vocab

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
	k         *vk.K2
	ck        *checkpoint
	embed     tensors.Tensor
	layers    []encLayer
	tapLayers []int

	// The arena, sized for MaxPromptTokens.
	x, n, q, kk, v, att, g, u, taps, table uint32
	programs                               map[[2]int]*vk.Program
}

// OpenEncoder uploads the text model's first 35 layers to the card, or,
// with host set, keeps them in system memory the card reads across the bus:
// a prompt reads each weight once, so that costs about half a second a
// prompt and none of the card, which the DiT wants all of.
func OpenEncoder(d *vk.Device, path string, host bool) (*Encoder, error) {
	return OpenEncoderTaps(d, path, host, Taps[:])
}

// OpenEncoderTaps uploads the text model up to the last tap in taps to the
// card (or system memory with host set) with taps recorded.
func OpenEncoderTaps(d *vk.Device, path string, host bool, taps []int) (*Encoder, error) {
	if len(taps) == 0 {
		return nil, fmt.Errorf("krea2: taps must not be empty")
	}
	for i := 1; i < len(taps); i++ {
		if taps[i] <= taps[i-1] {
			return nil, fmt.Errorf("krea2: taps must be strictly increasing, got %d <= %d", taps[i], taps[i-1])
		}
	}
	if taps[0] < 0 {
		return nil, fmt.Errorf("krea2: taps must be non-negative, got %d", taps[0])
	}
	ck, err := openCheckpoint(path)
	if err != nil {
		return nil, err
	}
	e := &Encoder{
		ck:        ck,
		tapLayers: append([]int(nil), taps...),
		programs:  map[[2]int]*vk.Program{},
	}
	fail := func(err error) (*Encoder, error) {
		e.Close()
		return nil, err
	}
	if e.embed, err = ck.get("model.embed_tokens.weight", "BF16", encVocab, encWidth); err != nil {
		return fail(err)
	}

	var par params
	layers := taps[len(taps)-1]
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
	e.taps = a.take(len(e.tapLayers) * T * encWidth)
	e.table = a.take(T * encHeadDim)

	if e.k, err = vk.NewK2(d, a.n, par.data); err != nil {
		return fail(err)
	}
	add := e.k.AddWeights
	if host {
		add = e.k.AddHostWeights
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
			t, err := ck.get(pre+name+".weight", "F8_E4M3", shapes[j][0], shapes[j][1])
			if err != nil {
				return fail(err)
			}
			if *handles[j], err = add(t.Raw); err != nil {
				return fail(err)
			}
		}
	}
	return e, nil
}

// program records the encoder for tokens positions with real keys attended to.
func (e *Encoder) program(tokens, real int) (*vk.Program, error) {
	key := [2]int{tokens, real}
	if p, ok := e.programs[key]; ok {
		return p, nil
	}
	T := uint32(tokens)
	K := uint32(real)
	qw, kw := uint32(encHeads*encHeadDim), uint32(encKVHeads*encHeadDim)
	mm := func(r *vk.Recorder, w int, scale float32, outs, ins, x, y uint32, mode uint32) {
		e.k.MM(r, w, true, vk.K2MM{Outputs: outs, Inputs: ins, Cols: T, X: x, XStride: ins, Y: y, YStride: outs,
			Bias: vk.K2None, Scale: scale, Mode: mode})
	}
	p, err := e.k.Compile(func(r *vk.Recorder) {
		tap := 0
		for i, l := range e.layers {
			if tap < len(e.tapLayers) && e.tapLayers[tap] == i {
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
				Queries: T, Keys: K, Group: encHeads / encKVHeads, Causal: 1, Scale: float32(1 / math.Sqrt(encHeadDim)),
				Heads: encHeads, Seqs: 1, HeadDim: encHeadDim})
			mm(r, l.o, l.scale[3], encWidth, qw, e.att, e.x, vk.K2Add)
			e.k.Norm(r, vk.K2Norm{Src: e.x, SrcStride: encWidth, Dst: e.n, DstStride: encWidth, N: encWidth, Rows: T,
				Weight: l.postNorm, Eps: encEps, ScaleA: vk.K2None})
			mm(r, l.gate, l.scale[4], encFFN, encWidth, e.n, e.g, vk.K2Store)
			mm(r, l.up, l.scale[5], encFFN, encWidth, e.n, e.u, vk.K2Store)
			e.k.Act(r, vk.K2Act{X: e.g, XStride: encFFN, Y: e.u, YStride: encFFN, N: encFFN, Rows: T, Op: vk.K2SwiGLU})
			mm(r, l.down, l.scale[6], encWidth, encFFN, e.g, e.x, vk.K2Add)
		}
		if tap < len(e.tapLayers) && e.tapLayers[tap] == len(e.layers) {
			e.k.Act(r, vk.K2Act{X: e.taps + uint32(tap)*T*encWidth, XStride: encWidth, Y: e.x, YStride: encWidth, N: encWidth, Rows: T, Op: vk.K2Copy})
		}
	})
	if err != nil {
		return nil, err
	}
	e.programs[key] = p
	return p, nil
}

// Encode runs the prompt's tokens and returns the conditioning Krea 2 reads:
// for each token from p.Start on, the twelve taps side by side.
func (e *Encoder) Encode(p Prompt) ([]float32, int, error) {
	T := len(p.IDs)
	if p.Start < 0 || p.Start >= T {
		return nil, 0, fmt.Errorf("krea2: the prompt's text starts at %d of %d tokens", p.Start, T)
	}
	all, err := e.EncodeMasked(p.IDs, T)
	if err != nil {
		return nil, 0, err
	}
	condWidth := len(e.tapLayers) * encWidth
	seq := T - p.Start
	cond := make([]float32, seq*condWidth)
	copy(cond, all[p.Start*condWidth:])
	return cond, seq, nil
}

// EncodeMasked runs T tokens where the first real tokens are text and the
// remainder are padding. Pads are masked as keys in attention, their positions
// in the RoPE table are forced to 1, and the returned conditioning has all T
// rows with len(taps) * 2560 values each.
func (e *Encoder) EncodeMasked(ids []int32, real int) ([]float32, error) {
	T := len(ids)
	if T > MaxPromptTokens {
		return nil, fmt.Errorf("krea2: a prompt of %d tokens is past the %d the encoder holds", T, MaxPromptTokens)
	}
	if real <= 0 || real > T {
		return nil, fmt.Errorf("krea2: %d real tokens for a sequence of %d", real, T)
	}
	x := make([]float32, T*encWidth)
	for i, id := range ids {
		if id < 0 || int(id) >= encVocab {
			return nil, fmt.Errorf("krea2: token %d is outside the vocabulary", id)
		}
		row := e.embed.Raw[int(id)*encWidth*2:]
		for j := 0; j < encWidth; j++ {
			x[i*encWidth+j] = math.Float32frombits(uint32(binary.LittleEndian.Uint16(row[2*j:])) << 16)
		}
	}
	if err := e.k.Write(int(e.x), x); err != nil {
		return nil, err
	}
	if err := e.k.Write(int(e.table), encoderRopeMasked(T, real)); err != nil {
		return nil, err
	}
	prog, err := e.program(T, real)
	if err != nil {
		return nil, err
	}
	if err := prog.Run(); err != nil {
		return nil, err
	}
	taps, err := e.k.Read(int(e.taps), len(e.tapLayers)*T*encWidth)
	if err != nil {
		return nil, err
	}
	condWidth := len(e.tapLayers) * encWidth
	cond := make([]float32, T*condWidth)
	for t := 0; t < T; t++ {
		for j := range e.tapLayers {
			copy(cond[t*condWidth+j*encWidth:(t*condWidth+(j+1)*encWidth)], taps[(j*T+t)*encWidth:(j*T+t+1)*encWidth])
		}
	}
	return cond, nil
}

// encoderRope is Qwen3's rotation table for positions 0..T-1, the way
// precompute_freqs_cis computes it: float32 inverse frequencies, float32
// angles.
func encoderRope(T int) []float32 {
	return encoderRopeMasked(T, T)
}

// encoderRopeMasked computes Qwen3's rotation table for positions where
// real tokens take positions 0..real-1 and any padding tokens at or after real
// take position 1.
func encoderRopeMasked(T, real int) []float32 {
	half := encHeadDim / 2
	inv := make([]float32, half)
	for i := range inv {
		inv[i] = 1 / float32(math.Pow(encTheta, float64(float32(2*i)/encHeadDim)))
	}
	out := make([]float32, T*encHeadDim)
	for t := 0; t < T; t++ {
		pos := t
		if t >= real {
			pos = 1
		}
		for i := 0; i < half; i++ {
			a := float64(inv[i] * float32(pos))
			out[t*encHeadDim+i] = float32(math.Cos(a))
			out[t*encHeadDim+half+i] = float32(math.Sin(a))
		}
	}
	return out
}

// Trim gives the encoder's working memory back to the card; the next Encode
// takes it again.
func (e *Encoder) Trim() {
	for _, p := range e.programs {
		p.Close()
	}
	e.programs = map[[2]int]*vk.Program{}
	e.k.FreeArena()
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
