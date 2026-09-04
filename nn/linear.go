package nn

// Linear is an affine projection. The transformer uses no bias; the flow net
// and the decoder have them everywhere. The weights stay the bytes of the
// memory-mapped file: that is the constraint holding the throughput, and the
// conversion happens inside the kernel.
type Linear struct {
	Weights []byte    // row-major, Outputs x Inputs
	Bias    []float32 // nil when the layer has none
	Inputs  int
	Outputs int

	// F32 says the bytes are float32 rather than bfloat16. Kyutai ships the
	// same codec in both: pocket-tts's Mimi is bfloat16 throughout, the one
	// that comes with the STT is float32 throughout, and the reference
	// computes each in the precision it was given. Rounding the float32 one
	// down to bfloat16 to reuse the narrower kernel costs about four parts in
	// a thousand on every weight, which eight layers of a transformer turn
	// into one and a half per cent — thirty thousand times the gap the rest of
	// this repository holds against its reference.
	//
	// The zero value is bfloat16, which is what every caller written before
	// this field meant.
	F32 bool
}

// Apply computes y = W*x. y must hold Outputs elements.
func (l Linear) Apply(x, y []float32) { l.ApplyBatch(x, y, 1) }

// ApplyBatch computes Y = W*X for `batch` consecutive vectors.
func (l Linear) ApplyBatch(x, y []float32, batch int) {
	InParallel(l.Outputs, l.Outputs*l.Inputs*batch, func(start, end int) {
		l.ApplyRows(x, y, batch, start, end)
	})
}

// ApplyRows computes rows [start, end) of the product, bias included, on the
// caller's thread.
//
// It exists so that a caller already inside a section can finish what it
// produced before the barrier: an activation over the rows it just computed
// costs a section of its own otherwise, and a section is a barrier every core
// waits at. The bias is here for the same reason — a pass over the outputs
// after the section is a pass on one core.
func (l Linear) ApplyRows(x, y []float32, batch, start, end int) {
	switch {
	case l.F32 && batch == 1:
		matVecF32Rows(l.Weights, x, l.Inputs, y, start, end)
	case l.F32:
		matMatF32Rows(l.Weights, x, l.Outputs, l.Inputs, batch, y, start, end)
	case batch == 1:
		MatVecBF16Rows(l.Weights, x, l.Inputs, y, start, end)
	default:
		MatMatBF16Rows(l.Weights, x, l.Outputs, l.Inputs, batch, y, start, end)
	}
	if l.Bias == nil {
		return
	}
	for k := 0; k < batch; k++ {
		block := y[k*l.Outputs : (k+1)*l.Outputs]
		for i := start; i < end; i++ {
			block[i] += l.Bias[i]
		}
	}
}
