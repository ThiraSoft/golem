package stt

// The trunk's four matrices on the card.
//
// What moves and what does not. The four products of a block move: they are
// fifty-one million weights read once a position, and on the processor that
// read is the whole of what a frame waits on. Attention stays — each stream
// owns a cache of seven hundred and fifty positions, and moving those would be
// moving the streams themselves. The Mimi codec and the split quantiser stay,
// and once the products are gone they are what a frame costs.
//
// So a block on the card is four dispatches with the processor between them,
// and that middle is not free: a product staged, dispatched and copied back
// costs about twice what its kernel does. It is paid anyway, because the same
// four products batched over eight streams cost 63.6 ms on this processor and
// 11.0 ms this way — the round trip is a third of a wall that was six times
// taller.

import (
	"fmt"
	"sync"

	"github.com/ThiraSoft/golem/nn"
	"github.com/ThiraSoft/golem/vk"
)

// A Card holds the trunk's matrices in device memory, for passes of up to the
// width it was built for.
type Card struct {
	d      *vk.Device
	width  int
	quant  nn.Quant
	layers []cardLayer

	// mu is held for a whole block.
	//
	// A Model is shared — the server holds one and every transcription reads
	// its weights at once — and that was true without cost while the weights
	// were only read. A card is not only read: a product stages its columns
	// into buffers that belong to the matrix and reads its answer out of
	// another, so two callers on one Model would write each other's
	// activations and each read a mixture. Nothing would fail; the transcripts
	// would simply be of the wrong sound.
	//
	// There is one card and one queue on it, so serializing here costs what
	// the hardware costs anyway. It is the block and not the product that is
	// locked, so that the attention between two products of the same block
	// cannot be overtaken by another caller's first one.
	mu sync.Mutex
}

// cardLayer is one block's four matrices, in the order Step uses them.
type cardLayer struct {
	inProj, outProj, gateIn, gateOut *vk.QuantBatch
}

// UseVulkan puts the trunk's products on a Vulkan device, for groups of up to
// width streams. It answers an error and changes nothing when there is no
// device or the format has no kernel, so a caller may always ask.
//
// The head stays on the processor: it is bfloat16, which none of these kernels
// read, and it costs 2.68 ms of a frame against the trunk's thirty.
func (m *Model) UseVulkan(width int) error {
	if m.card != nil {
		return fmt.Errorf("stt: the trunk is already on a card")
	}
	if width < 1 {
		return fmt.Errorf("stt: a pass is at least one stream, given %d", width)
	}
	q := m.weights.Quant
	d, err := vk.Open()
	if err != nil {
		return err
	}
	c := &Card{d: d, width: width, quant: q}
	for _, l := range m.weights.Layers {
		var built cardLayer
		for _, spec := range []struct {
			into **vk.QuantBatch
			m    nn.Matrix
		}{
			{&built.inProj, l.InProj},
			{&built.outProj, l.OutProj},
			{&built.gateIn, l.GateIn},
			{&built.gateOut, l.GateOut},
		} {
			b, err := vk.NewQuantBatch(d, q, spec.m.Data, spec.m.Rows, spec.m.Cols, width)
			if err != nil {
				c.Close()
				return err
			}
			*spec.into = b
		}
		c.layers = append(c.layers, built)
	}
	m.card = c
	return nil
}

// Vulkan says whether the trunk's products are on a card, and how wide a pass
// it was built for.
func (m *Model) Vulkan() (bool, int) {
	if m.card == nil {
		return false, 0
	}
	return true, m.card.width
}

func (c *Card) Close() {
	for _, l := range c.layers {
		for _, b := range []*vk.QuantBatch{l.inProj, l.outProj, l.gateIn, l.gateOut} {
			if b != nil {
				b.Close()
			}
		}
	}
	c.layers = nil
	if c.d != nil {
		c.d.Close()
		c.d = nil
	}
}

// product is one of the four, staged from the scratch's batch and read back
// into ys. It is the only place the card is spoken to, so that the block below
// reads like the one it mirrors.
func (c *Card) product(b *vk.QuantBatch, src *nn.Batch, ys [][]float32) error {
	src.Quantize()
	// Only the columns asked for are staged. A binary built at four columns run
	// for three still reads a fourth, and what it reads there is either zero —
	// the buffers start that way — or an activation a previous pass staged.
	// Both are finite, the columns of this kernel do not touch one another, and
	// the answer of the fourth is dropped by Run.
	for i := range ys {
		if err := b.SetColumn(i, src); err != nil {
			return err
		}
	}
	return b.Run(ys)
}

// StepBatchOn is StepBatch with the four products on the card. It is the same
// block in the same order — the difference is where the weights are read from,
// and that attention still happens here.
func (c *Card) StepBatchOn(layer int, l *Layer, xs [][]float32, kvs []*KV, s *BatchScratch) error {
	n := len(xs)
	if n > c.width {
		return fmt.Errorf("stt: a pass of %d streams on a card built for %d", n, c.width)
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	on := c.layers[layer]
	s.wide.Size, s.deep.Size = n, n

	for i := 0; i < n; i++ {
		copy(s.wide.F[i], xs[i])
		nn.RMSNormPlain(s.wide.F[i], l.Norm1, NormEps)
	}
	if err := c.product(on.inProj, s.wide, s.qkv[:n]); err != nil {
		return err
	}

	for i := 0; i < n; i++ {
		qkv := s.qkv[i]
		q, k, v := qkv[:DModel], qkv[DModel:2*DModel], qkv[2*DModel:]
		for head := 0; head < NumHeads; head++ {
			nn.ApplyRoPE(q[head*HeadDim:(head+1)*HeadDim], kvs[i].Position, MaxPeriod)
			nn.ApplyRoPE(k[head*HeadDim:(head+1)*HeadDim], kvs[i].Position, MaxPeriod)
		}
		kvs[i].write(k, v)
		s.qs[i] = q
	}
	attendBatch(kvs, s.qs[:n], s.attn[:n], s)

	for i := 0; i < n; i++ {
		copy(s.wide.F[i], s.attn[i])
	}
	if err := c.product(on.outProj, s.wide, s.out[:n]); err != nil {
		return err
	}
	for i := 0; i < n; i++ {
		x, out := xs[i], s.out[i]
		for j := range x {
			x[j] += out[j]
		}
	}

	for i := 0; i < n; i++ {
		copy(s.wide.F[i], xs[i])
		nn.RMSNormPlain(s.wide.F[i], l.Norm2, NormEps)
	}
	if err := c.product(on.gateIn, s.wide, s.gate[:n]); err != nil {
		return err
	}
	for i := 0; i < n; i++ {
		gate := s.gate[i]
		nn.SwiGLURange(gate[:DimFF], gate[DimFF:], 0, DimFF)
		copy(s.deep.F[i], gate[:DimFF])
	}
	if err := c.product(on.gateOut, s.deep, s.ff[:n]); err != nil {
		return err
	}
	for i := 0; i < n; i++ {
		x, ff := xs[i], s.ff[i]
		for j := range x {
			x[j] += ff[j]
		}
		kvs[i].Position++
	}
	return nil
}
