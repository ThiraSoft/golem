package vk

import (
	"math"
	"math/rand"
	"testing"

	"github.com/ThiraSoft/golem/compress"
	"github.com/ThiraSoft/golem/nn"
)

// A whole feed forward on the card against the same one on the processor.
//
// This is the first stage of the D4G path that is a block of a model rather
// than a kernel: the input prepared, the gate and the up in one product, the
// SiLU between them, the intermediate prepared for its own site, and the down
// projection. Five dispatches and two rotations, and any one of them wrong in
// a way a single kernel's test would not have caught — the wrong vector on the
// wrong site, the halves of the stacked product swapped, a barrier missing
// between a write and the read of it.
func TestD4GFFNMatchesCPU(t *testing.T) {
	d, err := Open()
	if err != nil {
		t.Skip(err)
	}
	defer d.Close()

	const dim, ffn, columns = 512, 1024, 4
	r := rand.New(rand.NewSource(31))
	gateW := randomWeights(r, ffn*dim)
	upW := randomWeights(r, ffn*dim)
	downW := randomWeights(r, dim*ffn)

	qGateUp := compress.RandomSigns(dim, 41)
	qDown := compress.RandomSigns(ffn, 43)
	preGateUp, preDown := reciprocal(qGateUp), reciprocal(qDown)

	p := compress.D4Params{ScaleBlock: nn.T4GBlock, HadGroup: 128}
	gateD := compress.EncodeT4GAs(gateW, ffn, dim, qGateUp, p, nn.T4G)
	upD := compress.EncodeT4GAs(upW, ffn, dim, qGateUp, p, nn.T4G)
	downD := compress.EncodeT4GAs(downW, dim, ffn, qDown, p, nn.T4G)

	k, err := NewGolemKernels(d, nn.T4G)
	if err != nil {
		t.Fatal(err)
	}
	defer k.Close()
	in, err := d.Host(dim*columns*4, bufferUsageStorage)
	if err != nil {
		t.Fatal(err)
	}
	defer in.Close()
	out, err := d.Readback(dim*columns*4, bufferUsageStorage)
	if err != nil {
		t.Fatal(err)
	}
	defer out.Close()

	f, err := NewD4GFFN(k, dim, ffn, columns, gateD, upD, downD, preGateUp, preDown, in, out)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()

	xs := make([][]float32, columns)
	for c := range xs {
		x := make([]float32, dim)
		for i := range x {
			x[i] = float32(r.NormFloat64()) * 0.5
		}
		xs[c] = x
		copy(in.Floats()[c*dim:], x)
	}
	var recErr error
	if err := d.Submit(func(rec *Recorder) { recErr = f.Record(rec, columns) }); err != nil {
		t.Fatal(err)
	}
	if recErr != nil {
		t.Fatal(recErr)
	}

	// The same arithmetic on the processor, from the same bytes.
	gateM := nn.Matrix{Data: gateD, Quant: nn.T4G, Rows: ffn, Cols: dim}
	upM := nn.Matrix{Data: upD, Quant: nn.T4G, Rows: ffn, Cols: dim}
	downM := nn.Matrix{Data: downD, Quant: nn.T4G, Rows: dim, Cols: ffn}
	worst, scale := 0.0, 0.0
	got := out.Floats()
	for c := 0; c < columns; c++ {
		x := append([]float32(nil), xs[c]...)
		nn.PrepareD4G(x, preGateUp, 128)
		b := nn.NewBatch(dim, 1)
		copy(b.F[0], x)
		g := make([]float32, ffn)
		u := make([]float32, ffn)
		gateM.MatVec(b, g)
		upM.MatVec(b, u)
		a := make([]float32, ffn)
		for i := range a {
			a[i] = float32(float64(g[i])/(1+math.Exp(-float64(g[i])))) * u[i]
		}
		nn.PrepareD4G(a, preDown, 128)
		ab := nn.NewBatch(ffn, 1)
		copy(ab.F[0], a)
		want := make([]float32, dim)
		downM.MatVec(ab, want)
		for i, v := range want {
			if m := math.Abs(float64(v)); m > scale {
				scale = m
			}
			if e := math.Abs(float64(got[c*dim+i]) - float64(v)); e > worst {
				worst = e
			}
		}
	}
	t.Logf("%d columns of %d through %d, worst gap %.3g of the largest output", columns, dim, ffn, worst/scale)
	if worst > 1e-3*scale {
		t.Errorf("the card's feed forward is %g away from the processor's", worst/scale)
	}
}

func randomWeights(r *rand.Rand, n int) []float32 {
	w := make([]float32, n)
	for i := range w {
		w[i] = float32(r.NormFloat64()) * 0.02
	}
	return w
}

func reciprocal(q []float32) []float32 {
	out := make([]float32, len(q))
	for i, v := range q {
		out[i] = 1 / v
	}
	return out
}
