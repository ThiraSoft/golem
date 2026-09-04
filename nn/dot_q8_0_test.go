package nn

import (
	"encoding/binary"
	"math"
	"math/rand/v2"
	"testing"
)

// packQ8_0 writes values into the format the kernel reads: 34 bytes a block, an
// fp16 scale then thirty-two signed bytes. It is the test's own packer on
// purpose — a kernel checked against the packer it ships with checks nothing.
func packQ8_0(values []float32, cols int) []byte {
	rows := len(values) / cols
	blocks := cols / QuantBlock
	out := make([]byte, rows*blocks*q8_0BlockBytes)
	for r := 0; r < rows; r++ {
		for b := 0; b < blocks; b++ {
			block := values[r*cols+b*QuantBlock : r*cols+(b+1)*QuantBlock]
			var amax float32
			for _, v := range block {
				if a := float32(math.Abs(float64(v))); a > amax {
					amax = a
				}
			}
			scale := amax / 127
			var inverse float32
			if scale != 0 {
				inverse = 1 / scale
			}
			at := (r*blocks + b) * q8_0BlockBytes
			binary.LittleEndian.PutUint16(out[at:], FloatToHalf(scale))
			for j, v := range block {
				q := int(math.Round(float64(v * inverse)))
				if q > 127 {
					q = 127
				}
				if q < -128 {
					q = -128
				}
				out[at+2+j] = byte(int8(q))
			}
		}
	}
	return out
}

// naiveQ8_0 is the definition of the product, written the slowest way there is:
// integers out of the bytes, one block at a time, the two scales at the end.
// Everything below is checked against this and against nothing else.
func naiveQ8_0(w []byte, b *Batch, column, cols int) float32 {
	var sum float64
	for step := 0; step < cols/QuantBlock; step++ {
		block := w[step*q8_0BlockBytes : (step+1)*q8_0BlockBytes]
		scale := halfToFloat(binary.LittleEndian.Uint16(block))
		index := step*b.Stride + column
		var acc int64
		for j := 0; j < QuantBlock; j++ {
			acc += int64(int8(block[2+j])) * int64(b.Q[index*QuantBlock+j])
		}
		sum += float64(acc) * float64(scale) * float64(b.Scales[index])
	}
	return float32(sum)
}

func q8Fixture(t testing.TB, rows, cols, batch int) ([]byte, *Batch) {
	t.Helper()
	r := rand.New(rand.NewPCG(7, 11))
	values := make([]float32, rows*cols)
	for i := range values {
		values[i] = float32(r.NormFloat64() * 0.1)
	}
	b := NewBatch(cols, batch)
	for c := 0; c < batch; c++ {
		for i := range b.F[c] {
			b.F[c][i] = float32(r.NormFloat64())
		}
	}
	b.Quantize()
	return packQ8_0(values, cols), b
}

// TestDotQ8_0AgainstDefinition covers whichever path this machine takes — the
// AVX2 kernel where there is one, the Go loop elsewhere — against the naive
// product. The tolerance is what float32 accumulation in a different order
// costs over 512 inputs, and no more.
func TestDotQ8_0AgainstDefinition(t *testing.T) {
	for _, cols := range []int{32, 64, 512, 2048} {
		w, b := q8Fixture(t, 1, cols, 1)
		var state [9]float32
		dotQ8_0(w, b, 0, 0, cols, state[:], Begin|Finish)
		want := naiveQ8_0(w, b, 0, cols)
		if diff := math.Abs(float64(state[0] - want)); diff > 1e-3*math.Abs(float64(want))+1e-4 {
			t.Errorf("cols %d: got %g, want %g", cols, state[0], want)
		}
	}
}

// TestDotQ8_0GoMatchesDefinition pins the portable path on its own, so that a
// machine without AVX2 is covered even when the test above took the kernel.
func TestDotQ8_0GoMatchesDefinition(t *testing.T) {
	const cols = 512
	w, b := q8Fixture(t, 1, cols, 1)
	var state [9]float32
	dotQ8_0Go(w, b, 0, 0, cols, state[:])
	want := naiveQ8_0(w, b, 0, cols)
	if diff := math.Abs(float64(state[0] - want)); diff > 1e-3*math.Abs(float64(want))+1e-4 {
		t.Errorf("got %g, want %g", state[0], want)
	}
}

// TestDotQ8_0Stretches checks the thing that makes tiling safe: a row cut into
// pieces and accumulated across calls must give what the row done in one call
// gives. Begin on the first, Finish on the last, neither in between.
func TestDotQ8_0Stretches(t *testing.T) {
	const cols = 2048
	w, b := q8Fixture(t, 1, cols, 1)

	var whole [9]float32
	dotQ8_0(w, b, 0, 0, cols, whole[:], Begin|Finish)

	var piecewise [9]float32
	const stretch = 512
	for at := 0; at < cols; at += stretch {
		mode := Mode(0)
		if at == 0 {
			mode |= Begin
		}
		if at+stretch >= cols {
			mode |= Finish
		}
		dotQ8_0(w[at/QuantBlock*q8_0BlockBytes:], b, at/QuantBlock, 0, stretch, piecewise[:], mode)
	}
	if diff := math.Abs(float64(whole[0] - piecewise[0])); diff > 1e-4*math.Abs(float64(whole[0]))+1e-5 {
		t.Errorf("whole %g, in pieces %g", whole[0], piecewise[0])
	}
}

// TestMatVecQ8_0Rows covers the matrix entry point over a batch, which is what
// every caller actually reaches.
func TestMatVecQ8_0Rows(t *testing.T) {
	const rows, cols, batch = 48, 512, 5
	w, b := q8Fixture(t, rows, cols, batch)
	ys := make([][]float32, batch)
	for c := range ys {
		ys[c] = make([]float32, rows)
	}
	matVecQ8_0Rows(w, b, cols, ys, 0, rows)

	rowBytes := cols / QuantBlock * q8_0BlockBytes
	for c := 0; c < batch; c++ {
		for r := 0; r < rows; r++ {
			want := naiveQ8_0(w[r*rowBytes:], b, c, cols)
			if diff := math.Abs(float64(ys[c][r] - want)); diff > 1e-3*math.Abs(float64(want))+1e-4 {
				t.Fatalf("column %d row %d: got %g, want %g", c, r, ys[c][r], want)
			}
		}
	}
}

func BenchmarkDotQ8_0(b *testing.B) {
	const cols = 2048
	w, batch := q8Fixture(b, 1, cols, 1)
	var state [9]float32
	b.SetBytes(int64(cols / QuantBlock * q8_0BlockBytes))
	for i := 0; i < b.N; i++ {
		dotQ8_0(w, batch, 0, 0, cols, state[:], Begin|Finish)
	}
}

func BenchmarkDotQ8_0Go(b *testing.B) {
	const cols = 2048
	w, batch := q8Fixture(b, 1, cols, 1)
	var state [9]float32
	b.SetBytes(int64(cols / QuantBlock * q8_0BlockBytes))
	for i := 0; i < b.N; i++ {
		clear(state[:])
		dotQ8_0Go(w, batch, 0, 0, cols, state[:])
	}
}
