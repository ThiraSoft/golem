package mel

import (
	"math"
	"math/cmplx"
	"math/rand"
	"testing"
)

// The fast transform is the slow one, or it is wrong. 512 is the size the
// model asks for; 8 is small enough to read a failure in.
func TestFFTAgreesWithTheDefinition(t *testing.T) {
	for _, n := range []int{8, 512} {
		in := make([]float32, n)
		for i := range in {
			in[i] = rand.Float32()*2 - 1
		}
		got := make([]complex64, n/2+1)
		NewFFT(n).Real(in, got)
		for k := range got {
			var sum complex128
			for j, v := range in {
				sum += complex(float64(v), 0) * cmplx.Exp(complex(0, -2*math.Pi*float64(k*j)/float64(n)))
			}
			if d := cmplx.Abs(complex128(got[k]) - sum); d > 1e-3 {
				t.Fatalf("n=%d bin %d: %v vs %v", n, k, got[k], sum)
			}
		}
	}
}

// A size that is not a power of two is a sentence at construction, not a wrong
// answer at the first frame.
func TestFFTRefusesASizeItCannotDo(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Fatal("NewFFT(300) returned a transform it cannot compute")
		}
	}()
	NewFFT(300)
}
