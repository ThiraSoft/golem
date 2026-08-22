package mel

import (
	"math"
	"testing"
)

// The HTK filterbank llama.cpp builds: 128 triangles over 257 bins, no weight
// above one, and the two triangles covering a bin summing to one.
//
// That sum is the test that says HTK rather than Slaney. Slaney divides every
// triangle by its own width, so a narrow low-frequency filter comes out tall
// and the partition below fails; HTK leaves them all rising to one. Asserting
// a peak of one per row would not do: the lowest filters are narrower than the
// 31.25 Hz between two bins of a 512-point transform at 16 kHz, so no bin
// falls inside them at all and their row is entirely zero. That is what
// llama.cpp builds too.
func TestFilterbankIsTriangles(t *testing.T) {
	fb := Filterbank(128, 512, 16000, true)
	const bins = 512/2 + 1
	for _, v := range fb {
		if v < 0 || v > 1+1e-6 {
			t.Fatalf("a filter weight is %f, outside [0, 1]", v)
		}
	}
	// The first covered bin is the first one any filter reaches; below it the
	// triangles are too narrow to hold one.
	first := bins
	for k := 0; k < bins; k++ {
		for m := 0; m < 128; m++ {
			if fb[m*bins+k] > 0 && k < first {
				first = k
			}
		}
	}
	// Above the last filter's centre only its falling edge remains, and one
	// edge alone does not sum to one; the partition is asserted where two
	// triangles overlap, which is everywhere between the first and the last
	// centre.
	for k := first; k < bins-1; k++ {
		var sum float32
		covered := 0
		for m := 0; m < 128; m++ {
			if fb[m*bins+k] > 0 {
				covered++
			}
			sum += fb[m*bins+k]
		}
		if covered < 2 {
			continue
		}
		if math.Abs(float64(sum)-1) > 1e-3 {
			t.Fatalf("bin %d is covered %f times over, want once", k, sum)
		}
	}
}

// TestFilterbankCentresRise: the centre frequencies are in order, so a filter
// never sits on top of its neighbour.
func TestFilterbankCentresRise(t *testing.T) {
	fb := Filterbank(128, 512, 16000, true)
	const bins = 512/2 + 1
	prev := -1
	for m := 0; m < 128; m++ {
		best, at := float32(-1), -1
		for k, v := range fb[m*bins : (m+1)*bins] {
			if v > best {
				best, at = v, k
			}
		}
		if at < prev {
			t.Fatalf("filter %d peaks at bin %d, below filter %d's %d", m, at, m-1, prev)
		}
		prev = at
	}
}
