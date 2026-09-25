package nn

import (
	"math"
	"math/rand"
	"testing"
)

// TestPairGeometry is the arithmetic the formats are: H3G's 392 bits of path
// and two step codes for 128 weights, 51 bytes, 3.1875 bits a weight; H4G's
// 519 bits in 65 bytes, T4G's 67 a block.
func TestPairGeometry(t *testing.T) {
	for _, c := range []struct {
		q        Quant
		seq, blk int
		bpw      float64
	}{{H3G, 49, 51, 3.1875}, {H4G, 65, 67, 4.1875}} {
		p := PairTierOf(c.q)
		if want := (p.L + (T4GSeq/2-1)*2*p.K + 7) / 8; p.SeqBytes != want || want != c.seq {
			t.Fatalf("%s: a sequence is %d bytes of path, want %d", c.q, p.SeqBytes, c.seq)
		}
		if got := p.RowBytes(T4GSeq); got != c.blk {
			t.Fatalf("%s: 128 weights are %d bytes, not %d", c.q, got, c.blk)
		}
		if bpw := float64(p.RowBytes(1024)*8) / 1024; bpw != c.bpw {
			t.Fatalf("%s is %v bits a weight, not %v", c.q, bpw, c.bpw)
		}
		if p.RowBytes(512)%4 != 0 {
			t.Fatalf("%s: a 512-wide row is %d bytes, not a whole number of words", c.q, p.RowBytes(512))
		}
		if len(p.Codebook()) != 2*p.Entries {
			t.Fatalf("%s: the codebook holds %d numbers for %d pairs", c.q, len(p.Codebook()), p.Entries)
		}
	}
}

// TestPairRoundTripIsExact: what goes into a sequence comes back out of it.
func TestPairRoundTripIsExact(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	for _, q := range []Quant{H3G, H4G} {
		p := PairTierOf(q)
		for trial := 0; trial < 100; trial++ {
			states := make([]uint16, T4GSeq/2)
			states[0] = uint16(rng.Intn(1 << p.L))
			for i := 1; i < len(states); i++ {
				states[i] = (states[i-1]<<(2*p.K) | uint16(rng.Intn(1<<(2*p.K)))) & (1<<p.L - 1)
			}
			dst := make([]byte, p.SeqBytes)
			p.PutStates(dst, states)
			for i := range states {
				if got := p.StateAt(dst, i); got != states[i] {
					t.Fatalf("%s pair %d read back as %#x, want %#x", q, i, got, states[i])
				}
			}
		}
	}
}

// TestPairCodebookIsHalfExact: the kernels hold a codebook as half2, so every
// value must survive the round trip through half precision, and none may be
// non-finite. And it must be near unit power, which a table of zeros is not.
func TestPairCodebookIsHalfExact(t *testing.T) {
	for _, q := range []Quant{H3G, H4G} {
		var ss float64
		book := PairTierOf(q).Codebook()
		for i, v := range book {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				t.Fatalf("%s entry %d is %v", q, i/2, v)
			}
			if halfToFloat(FloatToHalf(v)) != v {
				t.Fatalf("%s entry %d, %v, is not a half-precision number", q, i/2, v)
			}
			ss += float64(v) * float64(v)
		}
		// Not under one, as a quantizer's output is, because the entries are
		// not used equally: the far ones are few states' and rarely reached.
		// H3G's read 1.088 when written.
		if rms := math.Sqrt(ss / float64(len(book))); rms < 0.8 || rms > 1.25 {
			t.Fatalf("%s: the codebook's RMS is %.3f", q, rms)
		}
	}
}

// TestPairDequantizeMatchesStates: a row decodes, pair by pair, to its step
// times the codebook entry its state names.
func TestPairDequantizeMatchesStates(t *testing.T) {
	const n = 512
	rng := rand.New(rand.NewSource(3))
	for _, q := range []Quant{H3G, H4G} {
		p := PairTierOf(q)
		row := make([]byte, p.RowBytes(n))
		rng.Read(row)
		out := make([]float32, n)
		p.Dequantize(row, n, out)
		steps, codes := p.Planes(row, n)
		for i := 0; i < n; i += 2 {
			s := p.StateAt(codes[i/T4GSeq*p.SeqBytes:], i%T4GSeq/2)
			a, b := p.Value(s)
			d := T4GStep(steps[i/T4GBlock])
			if out[i] != a*d || out[i+1] != b*d {
				t.Fatalf("%s weights %d,%d decode to %v,%v, want %v,%v", q, i, i+1, out[i], out[i+1], a*d, b*d)
			}
		}
	}
}
