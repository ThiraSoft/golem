package nn

// Linear is an affine projection. The transformer uses no bias; the flow net
// and the decoder have them everywhere. The weights stay the bfloat16 bytes of
// the memory-mapped file: that is the constraint holding the throughput, and
// the conversion happens inside the kernel.
type Linear struct {
	Weights []byte    // row-major, Outputs x Inputs
	Bias    []float32 // nil when the layer has none
	Inputs  int
	Outputs int
}

// Apply computes y = W*x. y must hold Outputs elements.
func (l Linear) Apply(x, y []float32) {
	MatVecBF16(l.Weights, x, l.Outputs, l.Inputs, y)
	for i, b := range l.Bias {
		y[i] += b
	}
}

// ApplyBatch computes Y = W*X for `batch` consecutive vectors.
func (l Linear) ApplyBatch(x, y []float32, batch int) {
	if batch == 1 {
		l.Apply(x, y)
		return
	}
	MatMatBF16(l.Weights, x, l.Outputs, l.Inputs, batch, y)
	for k := 0; k < batch; k++ {
		block := y[k*l.Outputs : (k+1)*l.Outputs]
		for i, b := range l.Bias {
			block[i] += b
		}
	}
}
