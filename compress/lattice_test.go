package compress

import "testing"

// The theta series of E8 and D4, which is what the shell counts must reproduce:
// E8 has 240·σ₃(n) vectors of norm 2n, D4 has 24 times the sum of the odd
// divisors of n. A shell miscounted is a rate misreported, and a rate
// misreported makes every comparison in this package a lie.
func TestShellSizeMatchesThetaSeries(t *testing.T) {
	e8 := []struct {
		r2   float32
		want int
	}{{0, 1}, {2, 241}, {4, 2401}, {6, 9121}, {8, 26641}, {10, 56881}, {12, 117361}}
	for _, c := range e8 {
		if got := shellSize(LatE8, c.r2); got != c.want {
			t.Errorf("E8 within norm² %g: %d points, want %d", c.r2, got, c.want)
		}
	}
	d4 := []struct {
		r2   float32
		want int
	}{{0, 1}, {2, 25}, {4, 49}, {6, 145}, {8, 169}, {10, 313}}
	for _, c := range d4 {
		if got := shellSize(LatD4, c.r2); got != c.want {
			t.Errorf("D4 within norm² %g: %d points, want %d", c.r2, got, c.want)
		}
	}
}

// The nearest point of E8 must be a lattice point — integer coordinates of even
// sum, or that shifted by a half — and no further than the naive rounding.
func TestNearestE8IsALatticePoint(t *testing.T) {
	x := make([]float32, 8)
	out := make([]float32, 8)
	tmp := make([]float32, 8)
	seed := int64(1)
	for trial := 0; trial < 200; trial++ {
		for i := range x {
			seed = seed*6364136223846793005 + 1442695040888963407
			x[i] = float32(seed>>40) / float32(1<<22)
		}
		nearestE8(x, out, tmp)
		var sum, frac float32
		for _, v := range out {
			sum += v
			if f := v - float32(int(v)); f != 0 {
				frac = f
			}
		}
		if frac != 0 && frac != 0.5 && frac != -0.5 {
			t.Fatalf("coordinate fraction %v is neither integer nor half", frac)
		}
		if frac == 0 && int(sum)%2 != 0 {
			t.Fatalf("integer point of odd coordinate sum %v", sum)
		}
	}
}
