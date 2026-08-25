package vk

// The angle table the card makes, against the one nn/rope.go tabulates.
//
// The tables moved onto the card because making them cost six percent of a
// wide prompt on this side; what that trades away is float64, which is why
// this measures the gap rather than assuming it. llama.cpp's rope_funcs.glsl
// computes the same angles in float32 and its exponent with pow(), so this
// path is the more careful of the two.

import (
	"fmt"
	"math"
	"testing"

	"github.com/ThiraSoft/golem/nn"
)

func TestRopeTableMatchesTheCPU(t *testing.T) {
	d, err := Open()
	if err != nil {
		t.Skipf("no Vulkan device: %v", err)
	}
	defer d.Close()

	const dim, heads, kv, queryHeads, context = 256, 256, 256, 4, 4096
	const dims = 128
	a, err := NewAttention(d, dim, heads, kv, queryHeads, context, 1)
	if err != nil {
		t.Fatal(err)
	}
	defer a.Close()

	const base = 1000000.0
	factors := make([]float32, dims/2)
	for i := range factors {
		factors[i] = 1 + float32(i)*0.01
	}
	if err := a.SetGeometry(0, dims, base, factors); err != nil {
		t.Fatal(err)
	}

	columns := 32
	positions := []int{0, 1, 7, 63, 512, 3071, 4095}
	for c := 0; c < columns; c++ {
		if err := a.SetWhere(0, c, positions[c%len(positions)], 0, 0); err != nil {
			t.Fatal(err)
		}
	}

	// The tables are device memory now, so the recording copies them back.
	cos, err := d.Host(heads*4*columns, bufferUsageStorage)
	if err != nil {
		t.Fatal(err)
	}
	defer cos.Close()
	sin, err := d.Host(heads*4*columns, bufferUsageStorage)
	if err != nil {
		t.Fatal(err)
	}
	defer sin.Close()

	p, err := d.Compile(func(r *Recorder) {
		a.RecordRotations(r, columns)
		r.Barrier()
		r.CopyFrom(cos, 0, a.rcos[0], 0, heads*4*columns)
		r.CopyFrom(sin, 0, a.rsin[0], 0, heads*4*columns)
	})
	if err != nil {
		t.Fatal(err)
	}
	defer p.Close()
	if err := p.Run(); err != nil {
		t.Fatal(err)
	}

	var table nn.RoPETable
	gotCos, gotSin := cos.Floats(), sin.Floats()
	worst := 0.0
	worstAt := ""
	for c := 0; c < columns; c++ {
		table.Prepare(dims, positions[c%len(positions)], base, factors)
		for j := range table.Cos {
			for _, pair := range [][2]float32{
				{gotCos[c*heads+j], table.Cos[j]},
				{gotSin[c*heads+j], table.Sin[j]},
			} {
				if e := math.Abs(float64(pair[0] - pair[1])); e > worst {
					worst = e
					worstAt = fmt.Sprintf("col %d pos %d j %d: got %v want %v", c, positions[c%len(positions)], j, pair[0], pair[1])
				}
			}
		}
	}
	// One ULP. Getting there took a pair of floats for each frequency, a
	// reduction that subtracts the quarter turns from the pair rather than the
	// sum, and a sine of our own — shaders/rope_table.comp says why each was
	// needed and what the built-ins cost instead.
	if worst > 1e-6 {
		t.Fatalf("the card's angles are out by %v", worst)
	}
	t.Logf("worst absolute error over %d columns: %v (%s)", columns, worst, worstAt)
}
