package compress

import (
	"math"
	"math/rand"
	"testing"

	"github.com/ThiraSoft/golem/nn"
)

// The packing has to be exact, and it is the one contract of this format that
// is. A decoder is a pure function of the bits it reads, so a round trip that
// loses anything is a packing bug and not a rounding one — see the encoders,
// which are deliberately held to a weaker contract because two minimum-cost
// paths are two valid files.
func TestT4GRoundTripIsExact(t *testing.T) {
	rg := rand.New(rand.NewSource(17))
	for _, kind := range []nn.Quant{nn.T4G, nn.T5G} {
		for _, cols := range []int{128, 256, 1024, 1152} {
			const rows = 5
			w := make([]float32, rows*cols)
			for i := range w {
				w[i] = float32(rg.NormFloat64()) * 0.02
			}
			data := EncodeT4GAs(w, rows, cols, nil, D4Params{ScaleBlock: nn.T4GBlock}, kind)
			if want := rows * nn.T4GRowBytesN(cols, kind); len(data) != want {
				t.Fatalf("%s, %d columns: %d bytes, want %d", kind, cols, len(data), want)
			}

			// What the file says, read back the way a reader will.
			m := nn.Matrix{Data: data, Quant: kind, Rows: rows, Cols: cols}
			got := make([]float32, cols)

			// What the encoder chose, rebuilt from the same path and the same
			// steps without going through any bytes at all.
			for r := 0; r < rows; r++ {
				m.Row(r, got)
				row := make([]float32, cols)
				copy(row, w[r*cols:(r+1)*cols])
				want := reconstructT4G(row, kind)
				for j := range want {
					if got[j] != want[j] {
						t.Fatalf("%s, %d columns, row %d, weight %d: the file reads %v, the encoder chose %v",
							kind, cols, r, j, got[j], want[j])
					}
				}
			}
		}
	}
}

// reconstructT4G is EncodeT4G's arithmetic with no file in the middle: the same
// normalisation, the same path, the same least-squares step on the same grid.
func reconstructT4G(row []float32, kind nn.Quant) []float32 {
	cols := len(row)
	norm := make([]float32, cols)
	for b := 0; b*nn.T4GBlock < cols; b++ {
		blk := row[b*nn.T4GBlock : (b+1)*nn.T4GBlock]
		var ss float64
		for _, v := range blk {
			ss += float64(v) * float64(v)
		}
		inv := float32(0)
		if rms := float32(math.Sqrt(ss / float64(nn.T4GBlock))); rms > 0 {
			inv = 1 / rms
		}
		for i, v := range blk {
			norm[b*nn.T4GBlock+i] = v * inv
		}
	}
	states := make([]uint16, cols)
	QuantizeTrellisPath(norm, T4GOptsFor(kind), states)
	out := make([]float32, cols)
	for b := 0; b*nn.T4GBlock < cols; b++ {
		var num, den float64
		for i := b * nn.T4GBlock; i < (b+1)*nn.T4GBlock; i++ {
			num += float64(row[i]) * float64(norm[i])
			den += float64(norm[i]) * float64(norm[i])
		}
		code := byte(0)
		if den > 0 {
			code = nn.T4GStepCode(float32(num / den))
		}
		step := nn.T4GStep(code)
		for i := b * nn.T4GBlock; i < (b+1)*nn.T4GBlock; i++ {
			out[i] = nn.T4GValue(states[i]) * step
		}
	}
	return out
}

// The bits themselves: a legal path packed into 65 bytes and read back one
// weight at a time. This is the arithmetic every decoder repeats — Go's, the
// shader's — so it is checked on its own before anything is built on it.
func TestT4GStatesPackAndUnpack(t *testing.T) {
	rg := rand.New(rand.NewSource(5))
	for _, tier := range []struct {
		q nn.Quant
		k int
	}{{nn.T4G, nn.T4GK}, {nn.T5G, nn.T5GK}} {
		for trial := 0; trial < 64; trial++ {
			states := make([]uint16, nn.T4GSeq)
			states[0] = uint16(rg.Intn(1 << nn.T4GL))
			for i := 1; i < nn.T4GSeq; i++ {
				states[i] = uint16((uint32(states[i-1])<<uint(tier.k) | uint32(rg.Intn(1<<uint(tier.k)))) & (1<<nn.T4GL - 1))
			}
			buf := make([]byte, nn.T4GSeqBytesN(tier.q))
			nn.PutT4GStatesN(buf, states, tier.q)
			for i, want := range states {
				if got := nn.T4GStateAtN(buf, i, tier.q); got != want {
					t.Fatalf("%s, weight %d: read %012b, wrote %012b", tier.q, i, got, want)
				}
			}
		}
	}
}

// What the format costs, said in the same unit the plan for it was written in.
func TestT4GBitsPerWeight(t *testing.T) {
	const cols = 1024
	for _, want := range []struct {
		q   nn.Quant
		bpw float64
	}{{nn.T4G, 4.1875}, {nn.T5G, 5.1875}} {
		bpw := float64(nn.T4GRowBytesN(cols, want.q)) * 8 / cols
		if bpw != want.bpw {
			t.Fatalf("a %s row of %d is %d bytes, %.4f bits a weight, want %.4f",
				want.q, cols, nn.T4GRowBytesN(cols, want.q), bpw, want.bpw)
		}
	}
}

// What the file itself reconstructs, on the source the rotation makes every
// tensor into. The bench measures the codec; this measures the bytes, which is
// the thing a model reads, and the two should not differ.
//
// It also watches the step grid. A block whose step saturates is quantized
// against a codebook of the wrong size, and it does not look like an error: the
// file loads, every shape agrees, and the tensor reads at five decibels. That
// happened, with D4G's window, whose ceiling of 0.478 a unit-variance source
// walks straight past.
func TestT4GFileReconstructsAGaussian(t *testing.T) {
	nn.T4GStepClipped.Store(0)
	const rows, cols = 64, 1024
	w := gaussian(rows*cols, 21)
	for _, tier := range []struct {
		q     nn.Quant
		floor float64
	}{{nn.T4G, 20}, {nn.T5G, 26}} {
		data := EncodeT4GAs(w, rows, cols, nil, D4Params{ScaleBlock: nn.T4GBlock}, tier.q)

		m := nn.Matrix{Data: data, Quant: tier.q, Rows: rows, Cols: cols}
		rec := make([]float32, rows*cols)
		row := make([]float32, cols)
		for r := 0; r < rows; r++ {
			m.Row(r, row)
			copy(rec[r*cols:], row)
		}
		db := sqnrDB(w, rec)
		t.Logf("%s: %.4f bits a weight, %.2f dB (Shannon is %.2f)",
			tier.q, float64(nn.T4GRowBytesN(cols, tier.q))*8/cols, db,
			6.02*float64(nn.T4GRowBytesN(cols, tier.q))*8/cols)
		if n := nn.T4GStepClipped.Load(); n != 0 {
			t.Errorf("%d blocks landed on an end of the step grid", n)
		}
		// The codec reads 22.65 dB at four bits on a Gaussian and the file's
		// own per-block step buys a little more; a bit a weight is six more.
		// Anything under the floor is a packing bug, not a codebook.
		if db < tier.floor {
			t.Errorf("%s reconstructs at %.2f dB, which is not what the codec does", tier.q, db)
		}
	}
}
