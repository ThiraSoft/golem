package nn

import (
	"math"
	"testing"
)

// The values ggml's CPU backend actually returns, which are the formula seen
// through two fp16 roundings — the input is rounded to index the table, and the
// table holds fp16 results.
func TestGELUTableMatchesGGML(t *testing.T) {
	in := []float32{1, -1, 0.5, 3, -0.125, 0.0625}
	want := []float32{
		0.8413086, -0.15881348, 0.34570312, 2.9960938, -0.056274414, 0.032806396,
	}
	x := append([]float32(nil), in...)
	GELUTable(x)
	for i := range x {
		if x[i] != want[i] {
			t.Fatalf("GELU(%v) = %v, want exactly %v", in[i], x[i], want[i])
		}
	}
}

func TestGELUTableCutoffs(t *testing.T) {
	x := []float32{-10, -12.5, 10, 42}
	GELUTable(x)
	want := []float32{0, 0, 10, 42}
	for i := range x {
		if x[i] != want[i] {
			t.Fatalf("element %d: %v, want %v", i, x[i], want[i])
		}
	}
}

// Away from the cutoffs the table still has to be GELU, to within what fp16
// can hold — about three decimal digits.
func TestGELUTableTracksTheFormula(t *testing.T) {
	exact := func(x float64) float64 {
		return 0.5 * x * (1 + math.Tanh(0.79788456080286535587989211986876*x*(1+0.044715*x*x)))
	}
	x := make([]float32, 0, 400)
	for v := -9.0; v < 9.0; v += 0.045 {
		x = append(x, float32(v))
	}
	got := append([]float32(nil), x...)
	GELUTable(got)
	for i := range x {
		want := exact(float64(x[i]))
		if math.Abs(float64(got[i])-want) > 5e-3 {
			t.Fatalf("GELU(%v) = %v, formula says %v", x[i], got[i], want)
		}
	}
}
