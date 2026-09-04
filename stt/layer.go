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

	// NormEps is what the trunk's RMS norms add under the square root, and it
	// is 1e-8 and not the 1e-5 that a transformer usually carries: the
	// checkpoint asks for "rms_norm_f32", which the reference builds with
	// 1e-8. On a variance around 5e-3 the difference is one part in a
	// thousand, applied twice a layer and sixteen layers deep — enough to put
	// the trunk's waypoint eighty-six per cent of its own scale away from the
	// reference while still transcribing well enough to look correct.
	NormEps = 1e-8
)

type Layer struct {
	Norm1, Norm2 []float32
	InProj       nn.Matrix // 2048 -> 6144, Q K V concatenated
	OutProj      nn.Matrix // 2048 -> 2048
	GateIn       nn.Matrix // 2048 -> 11264
	GateOut      nn.Matrix // 5632 -> 2048
}

// Scratch is what one pass through the stack needs and one stream owns. The
// weights are shared by every conversation the server holds at once, so
// nothing that a product writes into may live beside them.
type Scratch struct {
	wide *nn.Batch // DModel-wide activations: the three projections and the gate
	deep *nn.Batch // DimFF-wide: what comes out of the gate
	h    []float32
	qkv  []float32
	out  []float32
	gate []float32
	ff   []float32
	attn []float32 // where attend writes: NumHeads x HeadDim
	// scores is one buffer of Context per head, and not one shared buffer,
	// because the heads are attended in parallel: two heads sharing the row of
	// scores they softmax would each read what the other wrote.
	scores [][]float32
}

func NewScratch() *Scratch {
	s := &Scratch{
		wide:   nn.NewBatch(DModel, 1),
		deep:   nn.NewBatch(DimFF, 1),
		h:      make([]float32, DModel),
		qkv:    make([]float32, 3*DModel),
		out:    make([]float32, DModel),
		gate:   make([]float32, 2*DimFF),
		ff:     make([]float32, DModel),
		attn:   make([]float32, DModel),
		scores: make([][]float32, NumHeads),
	}
	for i := range s.scores {
		s.scores[i] = make([]float32, Context)
	}
	return s
}

// product computes y = W*x through whichever kernel the matrix's format asks
// for. The activation is quantized only when the weights are: a bfloat16
// matrix reads the floats as they are, and rounding them first would move the
// answer away from what the reference computes.
func product(m nn.Matrix, b *nn.Batch, x, y []float32) {
	copy(b.F[0], x)
	if m.Quant != nn.BF16 {
		b.Quantize()
	}
	m.MatVec(b, y)
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

// attend is the softmax over the visible past, one head at a time and the heads
// on the pool.
//
// The heads are independent — each reads its own slice of K and V and writes
// its own slice of the answer — and there are sixteen of them over a window of
// seven hundred and fifty positions, which is enough work to be worth splitting
// and was not being split: this walked one thread while the other seven spun,
// and at a full window it was forty-three per cent of what a block costs. The
// buffers come from the scratch the stream owns, so a second stream attending
// at the same time writes into its own.
func (kv *KV) attend(q []float32, s *Scratch) []float32 {
	scale := float32(1.0 / math.Sqrt(float64(HeadDim)))
	first := 0
	if kv.Position+1 > Context {
		first = kv.Position + 1 - Context
	}
	count := kv.Position + 1 - first
	out := s.attn
	clear(out)

	// Two dot products of HeadDim per position per head: what the split has to
	// be worth is measured on the work, not on the head count.
	nn.InParallel(NumHeads, NumHeads*count*HeadDim*2, func(start, end int) {
		for h := start; h < end; h++ {
			buffer := s.scores[h][:count]
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
	})
	return out
}

// Step advances one position in place. x holds DModel values.
func (l *Layer) Step(x []float32, kv *KV, s *Scratch) {
	h := s.h
	copy(h, x)
	nn.RMSNormPlain(h, l.Norm1, NormEps)

	qkv := s.qkv
	product(l.InProj, s.wide, h, qkv)
	q, k, v := qkv[:DModel], qkv[DModel:2*DModel], qkv[2*DModel:]
	for head := 0; head < NumHeads; head++ {
		nn.ApplyRoPE(q[head*HeadDim:(head+1)*HeadDim], kv.Position, MaxPeriod)
		nn.ApplyRoPE(k[head*HeadDim:(head+1)*HeadDim], kv.Position, MaxPeriod)
	}
	kv.write(k, v)

	attn := kv.attend(q, s) // NumHeads x HeadDim, softmax over the visible past
	out := s.out
	product(l.OutProj, s.wide, attn, out)
	for i := range x {
		x[i] += out[i]
	}

	copy(h, x)
	nn.RMSNormPlain(h, l.Norm2, NormEps)
	gate := s.gate
	product(l.GateIn, s.wide, h, gate)
	nn.SwiGLURange(gate[:DimFF], gate[DimFF:], 0, DimFF) // silu(first) * second
	ff := s.ff
	product(l.GateOut, s.deep, gate[:DimFF], ff)
	for i := range x {
		x[i] += ff[i]
	}
	kv.Position++
}
