package stt

import (
	"math"

	"github.com/ThiraSoft/golem/nn"
)

const (
	DModel    = 2048
	NumLayers = 16
	NumHeads  = 16
	HeadDim   = 128
	DimFF     = 5632
	Context   = 750
	MaxPeriod = 100000.0
	TextCard  = 8000
	Codebooks = 32
	CodeCard  = 2048
	TextPadID = 3
)

type Layer struct {
	Norm1, Norm2 []float32
	InProj       nn.Linear // 2048 -> 6144, Q K V concatenated
	OutProj      nn.Linear // 2048 -> 2048
	GateIn       nn.Linear // 2048 -> 11264
	GateOut      nn.Linear // 5632 -> 2048
}

// KV is the cache of one layer, a ring over Context positions.
type KV struct {
	K, V     []float32
	Position int
}

func NewKV() []*KV {
	kvs := make([]*KV, NumLayers)
	for i := range kvs {
		kvs[i] = &KV{
			K: make([]float32, Context*NumHeads*HeadDim),
			V: make([]float32, Context*NumHeads*HeadDim),
		}
	}
	return kvs
}

func (kv *KV) write(k, v []float32) {
	slot := kv.Position % Context
	base := slot * NumHeads * HeadDim
	copy(kv.K[base:base+DModel], k)
	copy(kv.V[base:base+DModel], v)
}

func (kv *KV) attend(q []float32) []float32 {
	scale := float32(1.0 / math.Sqrt(float64(HeadDim)))
	first := 0
	if kv.Position+1 > Context {
		first = kv.Position + 1 - Context
	}
	count := kv.Position + 1 - first
	out := make([]float32, DModel)
	buffer := make([]float32, count)

	for h := 0; h < NumHeads; h++ {
		qh := q[h*HeadDim : (h+1)*HeadDim]
		for p := first; p <= kv.Position; p++ {
			slot := p % Context
			kp := kv.K[(slot*NumHeads+h)*HeadDim : (slot*NumHeads+h+1)*HeadDim]
			buffer[p-first] = nn.DotF32(qh, kp) * scale
		}
		nn.SoftmaxInPlace(buffer)
		oh := out[h*HeadDim : (h+1)*HeadDim]
		for p := first; p <= kv.Position; p++ {
			slot := p % Context
			vp := kv.V[(slot*NumHeads+h)*HeadDim : (slot*NumHeads+h+1)*HeadDim]
			nn.AxpyFull(oh, vp, buffer[p-first])
		}
	}
	return out
}

// Step advances one position in place. x holds DModel values.
func (l *Layer) Step(x []float32, kv *KV) {
	h := make([]float32, DModel)
	copy(h, x)
	nn.RMSNormPlain(h, l.Norm1, 1e-5)

	qkv := make([]float32, 3*DModel)
	l.InProj.Apply(h, qkv)
	q, k, v := qkv[:DModel], qkv[DModel:2*DModel], qkv[2*DModel:]
	for head := 0; head < NumHeads; head++ {
		nn.ApplyRoPE(q[head*HeadDim:(head+1)*HeadDim], kv.Position, MaxPeriod)
		nn.ApplyRoPE(k[head*HeadDim:(head+1)*HeadDim], kv.Position, MaxPeriod)
	}
	kv.write(k, v)

	attn := kv.attend(q) // NumHeads x HeadDim, softmax over the visible past
	out := make([]float32, DModel)
	l.OutProj.Apply(attn, out)
	for i := range x {
		x[i] += out[i]
	}

	copy(h, x)
	nn.RMSNormPlain(h, l.Norm2, 1e-5)
	gate := make([]float32, 2*DimFF)
	l.GateIn.Apply(h, gate)
	nn.SwiGLURange(gate[:DimFF], gate[DimFF:], 0, DimFF) // silu(first) * second
	ff := make([]float32, DModel)
	l.GateOut.Apply(gate[:DimFF], ff)
	for i := range x {
		x[i] += ff[i]
	}
	kv.Position++
}
