package vk

import (
	"math"
	"math/rand"
	"testing"

	"github.com/ThiraSoft/golem/nn"
)

// The card and the processor have to agree about what preparing an activation
// means, to the last bit that matters. A rotation applied one way on one side
// and another way on the other is the failure this catches, and it is silent:
// the weights still look like weights and the answer is somebody else's.
func TestPrepareGolemMatchesCPU(t *testing.T) {
	d, err := Open()
	if err != nil {
		t.Skip(err)
	}
	defer d.Close()

	const n, columns, group = 1024, 3, 128
	r := rand.New(rand.NewSource(4))
	pre := make([]float32, n)
	for i := range pre {
		s := float32(0.4 + r.Float64())
		if r.Intn(2) == 0 {
			s = -s
		}
		pre[i] = s
	}
	x := make([]float32, n*columns)
	for i := range x {
		x[i] = float32(r.NormFloat64())
	}

	act, err := d.Host(len(x)*4, bufferUsageStorage)
	if err != nil {
		t.Fatal(err)
	}
	defer act.Close()
	copy(act.Floats(), x)

	p, err := NewPrepareGolem(d, act, pre, group)
	if err != nil {
		t.Fatal(err)
	}
	defer p.Close()
	if err := p.Run(columns); err != nil {
		t.Fatal(err)
	}

	want := make([]float32, n*columns)
	copy(want, x)
	for c := 0; c < columns; c++ {
		nn.PrepareGolem(want[c*n:(c+1)*n], pre, group)
	}
	got := act.Floats()
	worst := 0.0
	for i := range want {
		if d := math.Abs(float64(got[i] - want[i])); d > worst {
			worst = d
		}
	}
	var scale float64
	for _, v := range want {
		if a := math.Abs(float64(v)); a > scale {
			scale = a
		}
	}
	t.Logf("%d columns of %d, worst gap %g of the largest value", columns, n, worst/scale)
	if worst > 1e-4*scale {
		t.Errorf("the card prepared an activation the processor would not have: worst gap %g", worst)
	}
}

// The same transform reading the form the attention's mix arrives in. What it
// has to get right beyond the float version is the layout of a Q8_0 activation:
// one signed byte a value packed four to a word, and per column the scales of
// its blocks of thirty-two before their corrections. Reading the corrections as
// scales would be a shift of one block and a plausible, wrong answer.
func TestPrepareGolemFromQ8MatchesCPU(t *testing.T) {
	d, err := Open()
	if err != nil {
		t.Skip(err)
	}
	defer d.Close()

	const n, columns, group = 512, 2, 128
	r := rand.New(rand.NewSource(8))
	pre := make([]float32, n)
	for i := range pre {
		s := float32(0.4 + r.Float64())
		if r.Intn(2) == 0 {
			s = -s
		}
		pre[i] = s
	}

	// A Q8_0 activation written the way attn_scores.comp writes one.
	blocks := n / 32
	qv := make([]byte, n*columns)
	qs := make([]float32, 2*blocks*columns)
	want := make([]float32, n*columns)
	for c := 0; c < columns; c++ {
		for b := 0; b < blocks; b++ {
			scale := float32(0.001 + r.Float64()*0.01)
			qs[c*2*blocks+b] = scale
			for i := 0; i < 32; i++ {
				v := int8(r.Intn(255) - 127)
				qv[c*n+b*32+i] = byte(v)
				want[c*n+b*32+i] = float32(v) * scale
			}
		}
	}

	out, err := d.Readback(n*columns*4, bufferUsageStorage)
	if err != nil {
		t.Fatal(err)
	}
	defer out.Close()
	values, err := d.Upload(qv)
	if err != nil {
		t.Fatal(err)
	}
	defer values.Close()
	scales, err := d.Upload(asBytes(qs))
	if err != nil {
		t.Fatal(err)
	}
	defer scales.Close()

	p, err := NewPrepareGolemFromQ8(d, out, values, scales, pre, group)
	if err != nil {
		t.Fatal(err)
	}
	defer p.Close()
	if err := p.Run(columns); err != nil {
		t.Fatal(err)
	}

	for c := 0; c < columns; c++ {
		nn.PrepareGolem(want[c*n:(c+1)*n], pre, group)
	}
	got := out.Floats()
	worst, scale := 0.0, 0.0
	for i := range want {
		if a := math.Abs(float64(want[i])); a > scale {
			scale = a
		}
		if e := math.Abs(float64(got[i] - want[i])); e > worst {
			worst = e
		}
	}
	t.Logf("%d columns of %d from Q8_0, worst gap %.3g of the largest value", columns, n, worst/scale)
	if worst > 1e-5*scale {
		t.Errorf("the card read a Q8_0 activation the processor would not have: %g", worst/scale)
	}
}
