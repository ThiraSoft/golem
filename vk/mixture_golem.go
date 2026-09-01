package vk

// The shared branch of a feed forward, in the Golem format.
//
// A dense model's whole feed forward is this branch, and a mixture's is this
// branch beside its experts. Only the dense half is here: an expert stack in
// Golem is a different question — one matrix a row of the stack, and a routing
// that says which — and the converter writes it before anything reads it.
//
// What the block does is vk/golem_ffn.go's, and this is where a Mixture holds
// one. The input arrives as floats rather than in the Q8_0 form the Q4_0 path
// reads, which is the same reason vk/attention_golem.go gives: the scale and
// the rotation a Golem matrix undoes on its activation are float arithmetic.

import "fmt"

// SharedFloatInput is where a Golem shared branch reads its normed input. The
// stack binds the residual norm's float output to it, in place of the Q8_0
// pair SharedInput returns.
func (m *Mixture) SharedFloatInput() *Buffer { return m.dxf }

// Golem says whether this mixture's shared branch reads Golem weights.
func (m *Mixture) Golem() bool { return len(m.blocks) > 0 && m.blocks[0].golem != nil }

// AddBlockGolem is AddBlock for a block whose shared branch is Golem. A mixture
// with experts is not yet one of these.
func (m *Mixture) AddBlockGolem(k *GolemKernels, gate, up, down []byte, preGateUp, preDown []float32) error {
	if m.experts > 0 {
		return fmt.Errorf("vk: a Golem mixture with %d experts is not written yet", m.experts)
	}
	if m.dxf == nil {
		var err error
		if m.dxf, err = m.d.Local(m.dim*4*maxColumns, bufferUsageStorage); err != nil {
			return err
		}
	}
	f, err := NewGolemFFN(k, m.dim, m.dense, maxColumns, gate, up, down, preGateUp, preDown, m.dxf, m.dout)
	if err != nil {
		return err
	}
	m.blocks = append(m.blocks, &mixtureBlock{golem: f})
	return nil
}

// recordGolem is the whole shared branch for a block that has one.
func (m *Mixture) recordGolem(r *Recorder, block, columns int) error {
	return m.blocks[block].golem.Record(r, columns)
}
