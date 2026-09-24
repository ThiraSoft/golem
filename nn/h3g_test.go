package nn

import (
	"math"
	"math/rand"
	"testing"
)

// TestH3GGeometry is the arithmetic the format is: 392 bits of path and two
// step codes for 128 weights, which is 51 bytes and 3.1875 bits a weight.
func TestH3GGeometry(t *testing.T) {
	if H3GSeqBytes != 49 {
		t.Fatalf("an H3G sequence is 49 bytes of path, not %d", H3GSeqBytes)
	}
	if got := H3GRowBytes(T4GSeq); got != 51 {
		t.Fatalf("128 weights of H3G are 51 bytes, not %d", got)
	}
	if bpw := float64(H3GRowBytes(1024)*8) / 1024; bpw != 3.1875 {
		t.Fatalf("H3G is %v bits a weight, not 3.1875", bpw)
	}
	if H3GRowBytes(512)%4 != 0 {
		t.Fatalf("a 512-wide row is %d bytes, not a whole number of words", H3GRowBytes(512))
	}
}

// TestH3GRoundTripIsExact: what goes into a sequence comes back out of it.
func TestH3GRoundTripIsExact(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	for trial := 0; trial < 100; trial++ {
		states := make([]uint16, T4GSeq/2)
		states[0] = uint16(rng.Intn(1 << H3GL))
		for i := 1; i < len(states); i++ {
			states[i] = (states[i-1]<<(2*H3GK) | uint16(rng.Intn(1<<(2*H3GK)))) & (1<<H3GL - 1)
		}
		dst := make([]byte, H3GSeqBytes)
		PutH3GStates(dst, states)
		for i := range states {
			if got := H3GStateAt(dst, i); got != states[i] {
				t.Fatalf("pair %d read back as %#x, want %#x", i, got, states[i])
			}
		}
	}
}

// TestH3GCodebookIsHalfExact: the kernels hold the codebook as half2, so every
// value must survive the round trip through half precision, and none may be
// zero-width or non-finite.
func TestH3GCodebookIsHalfExact(t *testing.T) {
	var ss float64
	for i, v := range H3GCodebook() {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatalf("entry %d is %v", i/2, v)
		}
		if halfToFloat(FloatToHalf(v)) != v {
			t.Fatalf("entry %d, %v, is not a half-precision number", i/2, v)
		}
		ss += float64(v) * float64(v)
	}
	// Near unit power: a table of zeros would pass everything else. Not under
	// one, as a quantizer's output is, because the entries are not used
	// equally — the far ones are few states' and rarely reached. 1.088 when
	// written.
	if rms := math.Sqrt(ss / float64(len(H3GCodebook()))); rms < 0.8 || rms > 1.25 {
		t.Fatalf("the codebook's RMS is %.3f", rms)
	}
}

// TestH3GDequantizeMatchesStates: a row decodes, pair by pair, to its step
// times the codebook entry its state names.
func TestH3GDequantizeMatchesStates(t *testing.T) {
	const n = 512
	rng := rand.New(rand.NewSource(3))
	row := make([]byte, H3GRowBytes(n))
	rng.Read(row)
	out := make([]float32, n)
	DequantizeH3G(row, n, out)
	steps, codes := H3GPlanes(row, n)
	for i := 0; i < n; i += 2 {
		seq := codes[i/T4GSeq*H3GSeqBytes:]
		s := H3GStateAt(seq, i%T4GSeq/2)
		a, b := H3GValue(s)
		d := T4GStep(steps[i/T4GBlock])
		if out[i] != a*d || out[i+1] != b*d {
			t.Fatalf("weights %d,%d decode to %v,%v, want %v,%v", i, i+1, out[i], out[i+1], a*d, b*d)
		}
	}
}
