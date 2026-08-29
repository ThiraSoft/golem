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
func TestPrepareD4GMatchesCPU(t *testing.T) {
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

	p, err := NewPrepareD4G(d, act, pre, group)
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
		nn.PrepareD4G(want[c*n:(c+1)*n], pre, group)
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
