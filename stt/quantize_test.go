package stt

import (
	"encoding/binary"
	"math"
	"math/rand/v2"
	"testing"

	"github.com/ThiraSoft/golem/nn"
)

// bf16Bytes writes values the way a checkpoint stores them, so the quantizer is
// fed exactly what it will be fed in service.
func bf16Bytes(values []float32) []byte {
	out := make([]byte, len(values)*2)
	for i, v := range values {
		binary.LittleEndian.PutUint16(out[i*2:], uint16(math.Float32bits(v)>>16))
	}
	return out
}

// TestQuantizeQ4_0RoundTrip holds the quantizer to what the format can promise:
// every value lands within half a step of the grid its block's scale defines.
// A quantizer that is merely close on average passes a cosine test and fails
// this one, which is why this one is here.
func TestQuantizeQ4_0RoundTrip(t *testing.T) {
	const cols = 64
	r := rand.New(rand.NewPCG(1, 2))
	values := make([]float32, cols)
	for i := range values {
		values[i] = nn.RoundBF16(float32(r.NormFloat64()))
	}

	m := nn.Matrix{Data: quantizeQ4_0(bf16Bytes(values), 1, cols), Quant: nn.Q4_0, Rows: 1, Cols: cols}
	got := make([]float32, cols)
	m.Row(0, got)

	for block := 0; block < cols/32; block++ {
		var amax float32
		for _, v := range values[block*32 : (block+1)*32] {
			if a := float32(math.Abs(float64(v))); a > amax {
				amax = a
			}
		}
		// The grid is amax/8 wide, so nothing may be off by more than half of
		// it, plus the room the scale's own half precision takes.
		step := amax / 8
		for i := block * 32; i < (block+1)*32; i++ {
			if diff := math.Abs(float64(got[i] - values[i])); diff > float64(step)*0.55 {
				t.Fatalf("value %d: %g became %g, off by %g for a step of %g", i, values[i], got[i], diff, step)
			}
		}
	}
}

// TestQuantizeQ4_0Product checks the thing that is actually used: the product,
// against the same product read in bfloat16. Twelve bits of mantissa become
// four, so the answer moves; what must not move is its direction.
//
// The threshold is the format's own noise floor and not a wish. Over
// thirty-two normal values the largest is about 2.5 sigma, so the grid step is
// about 0.31 sigma and the rounding error about 0.09 sigma; a dot product
// keeps that ratio however many terms it has, which puts the cosine at
// sqrt(1 - 0.09^2), or 0.996. Measured here: 0.9958. Anything materially below
// that is a fault in the quantizer, and anything above it on random weights
// would mean the test is not testing what it claims.
//
// What Q4_0 costs the transcript is not decided here — it is decided by the
// word error rate on real speech, in stt_test.go.
func TestQuantizeQ4_0Product(t *testing.T) {
	const rows, cols = 256, 512
	r := rand.New(rand.NewPCG(3, 4))
	weights := make([]float32, rows*cols)
	for i := range weights {
		weights[i] = nn.RoundBF16(float32(r.NormFloat64() * 0.05))
	}
	raw := bf16Bytes(weights)

	b := nn.NewBatch(cols, 1)
	for i := range b.F[0] {
		b.F[0][i] = float32(r.NormFloat64())
	}
	b.Quantize()

	wide := nn.Matrix{Data: raw, Quant: nn.BF16, Rows: rows, Cols: cols}
	narrow := nn.Matrix{Data: quantizeQ4_0(raw, rows, cols), Quant: nn.Q4_0, Rows: rows, Cols: cols}

	want, got := make([]float32, rows), make([]float32, rows)
	wide.MatVec(b, want)
	narrow.MatVec(b, got)

	var dot, na, nb float64
	for i := range want {
		dot += float64(want[i]) * float64(got[i])
		na += float64(want[i]) * float64(want[i])
		nb += float64(got[i]) * float64(got[i])
	}
	if cos := dot / math.Sqrt(na*nb); cos < 0.995 {
		t.Fatalf("cosine %f between the bfloat16 product and the Q4_0 one", cos)
	}
}
