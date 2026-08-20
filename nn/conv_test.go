package nn

// The two arrangements against each other.
//
// Both convolutions compute the same thing two ways — an axpy swept along the
// output row, or a dot product against a gathered window — and which one runs
// is decided by the shape. The forms are not interchangeable by accident: they
// sum in different orders and they index the weights differently, and the
// second is where a permutation is easy to get wrong. So each shape of the
// decoder is computed both ways here and the two are required to agree to what
// float32 can tell apart, on every call of a streaming run rather than on the
// first: a mistake in the state shows at the seam between calls, not inside one.

import (
	"math"
	"math/rand"
	"testing"
)

var convShapes = []struct {
	name                                      string
	inputs, outputs, kernel, stride, dilation int
	steps                                     int
}{
	{"projection 32->512 k1", 32, 512, 1, 1, 1, 1},
	{"input 512->512 k7", 512, 512, 7, 1, 1, 16},
	{"residual 256->128 k3", 256, 128, 3, 1, 1, 96},
	{"residual 32->64 k1", 32, 64, 1, 1, 1, 1920},
	{"strided 64->48 k4 s2", 64, 48, 4, 2, 1, 24},
	{"dilated 48->32 k3 d2", 48, 32, 3, 1, 2, 40},
}

func TestConv1dFormsAgree(t *testing.T) {
	for _, s := range convShapes {
		t.Run(s.name, func(t *testing.T) {
			r := rand.New(rand.NewSource(7))
			c := Conv1d{
				Inputs: s.inputs, Outputs: s.outputs, Kernel: s.kernel,
				Stride: s.stride, Dilation: s.dilation,
				Weights: randomConvFloats(r, s.inputs*s.outputs*s.kernel),
				Bias:    randomConvFloats(r, s.outputs),
			}
			state := c.NewState()
			for call := 0; call < 4; call++ {
				x := randomConvFloats(r, s.inputs*s.steps)
				got, outputs := c.Apply(x, s.steps, state)
				got = append([]float32(nil), got...)

				// The same call again, forced through the other form. The
				// state has already moved on, so what is replayed is the input
				// window Apply built, not the call.
				want := make([]float32, c.Outputs*outputs)
				total := (state.keep + s.steps)
				if c.gathered(outputs) {
					c.applyAxpy(state.input[:c.Inputs*total], want, total, outputs)
				} else {
					c.applyGathered(state.input[:c.Inputs*total], want, total, outputs, c.NewState())
				}
				compareForms(t, call, got, want)
			}
		})
	}
}

func TestConvTranspose1dFormsAgree(t *testing.T) {
	shapes := []struct {
		name                                 string
		inputs, outputs, kernel, stride, grp int
		steps                                int
	}{
		{"upsample 512 groups", 512, 512, 32, 16, 512, 1},
		{"expand 512->256 x6", 512, 256, 12, 6, 1, 16},
		{"expand 256->128 x5", 256, 128, 10, 5, 1, 96},
		{"expand 128->64 x4", 128, 64, 8, 4, 1, 480},
		{"uneven kernel 64->64 x3", 64, 64, 7, 3, 1, 12},
		{"two groups", 128, 64, 6, 3, 2, 20},
	}
	for _, s := range shapes {
		t.Run(s.name, func(t *testing.T) {
			r := rand.New(rand.NewSource(11))
			c := ConvTranspose1d{
				Inputs: s.inputs, Outputs: s.outputs, Kernel: s.kernel,
				Stride: s.stride, Groups: s.grp,
				Weights: randomConvFloats(r, s.inputs*(s.outputs/s.grp)*s.kernel),
				Bias:    randomConvFloats(r, s.outputs),
			}
			plain := c
			c.Prepare(s.steps)
			if c.Packed == nil {
				t.Logf("%s keeps the axpy form; the comparison is trivial", s.name)
			}
			slow, fast := plain.NewState(), c.NewState()
			for call := 0; call < 4; call++ {
				x := randomConvFloats(r, s.inputs*s.steps)
				want, n := plain.Apply(append([]float32(nil), x...), s.steps, slow)
				got, m := c.Apply(append([]float32(nil), x...), s.steps, fast)
				if n != m {
					t.Fatalf("call %d: %d steps out, want %d", call, m, n)
				}
				compareForms(t, call, got, want)
			}
		})
	}
}

func compareForms(t *testing.T, call int, got, want []float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("call %d: %d values, want %d", call, len(got), len(want))
	}
	var worst float64
	for i := range got {
		d := math.Abs(float64(got[i] - want[i]))
		if d > worst {
			worst = d
		}
	}
	// The two forms sum in different orders, so they are not required to be
	// identical — only to agree to what float32 can distinguish over a few
	// hundred terms.
	if worst > 1e-4 {
		t.Fatalf("call %d: the two forms differ by %g", call, worst)
	}
}

func randomConvFloats(r *rand.Rand, n int) []float32 {
	v := make([]float32, n)
	for i := range v {
		v[i] = r.Float32() - 0.5
	}
	return v
}
