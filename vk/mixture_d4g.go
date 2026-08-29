package vk

// The shared branch of a feed forward, in the D4G format.
//
// A dense model's whole feed forward is this branch, and a mixture's is this
// branch beside its experts. Only the dense half is here: an expert stack in
// D4G is a different question — one matrix a row of the stack, and a routing
// that says which — and the converter writes it before anything reads it.
//
// What the block does is vk/d4g_ffn.go's, and this is where a Mixture holds
// one. The input arrives as floats rather than in the Q8_0 form the Q4_0 path
// reads, which is the same reason vk/attention_d4g.go gives: the scale and the
// rotation a D4G matrix undoes on its activation are float arithmetic.

import "fmt"

// SharedFloatInput is where a D4G shared branch reads its normed input. The
// stack binds the residual norm's float output to it, in place of the Q8_0
// pair SharedInput returns.
func (m *Mixture) SharedFloatInput() *Buffer { return m.dxf }

// D4G says whether this mixture's shared branch reads D4G weights.
func (m *Mixture) D4G() bool { return len(m.blocks) > 0 && m.blocks[0].d4g != nil }

// AddBlockD4G is AddBlock for a block whose shared branch is D4G. A mixture
// with experts is not yet one of these.
func (m *Mixture) AddBlockD4G(k *D4GKernels, gate, up, down []byte, preGateUp, preDown []float32) error {
	if m.experts > 0 {
		return fmt.Errorf("vk: a D4G mixture with %d experts is not written yet", m.experts)
	}
	if m.dxf == nil {
		var err error
		if m.dxf, err = m.d.Local(m.dim*4*maxColumns, bufferUsageStorage); err != nil {
			return err
		}
	}
	f, err := NewD4GFFN(k, m.dim, m.dense, maxColumns, gate, up, down, preGateUp, preDown, m.dxf, m.dout)
	if err != nil {
		return err
	}
	m.blocks = append(m.blocks, &mixtureBlock{d4g: f})
	return nil
}

// recordD4G is the whole shared branch for a block that has one.
func (m *Mixture) recordD4G(r *Recorder, block, columns int) error {
	return m.blocks[block].d4g.Record(r, columns)
}
