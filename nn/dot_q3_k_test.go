package nn

import (
	"encoding/binary"
	"math"
	"math/rand"
	"testing"
)

// TestDotQ3_KMatchesTheDequantiser holds the integer product to the float one.
//
// The two read the same bytes by different routes: DequantizeQ3_K walks the
// superblock weight by weight and multiplies floats, dotQ3_K keeps the
// magnitudes as small integers and pays the recentring against the activation's
// group sums. They agree to the activation's own rounding and no closer, which
// is what the tolerance here is.
func TestDotQ3_KMatchesTheDequantiser(t *testing.T) {
	const n = 1024
	rng := rand.New(rand.NewSource(3))
	w := make([]byte, n/SuperBlock*q3_kBlockBytes)
	rng.Read(w)
	for b := 0; b < n/SuperBlock; b++ {
		binary.LittleEndian.PutUint16(w[b*q3_kBlockBytes+108:], Narrow(rng.Float32()*0.05))
	}

	x := make([]float32, n)
	for i := range x {
		x[i] = rng.Float32()*2 - 1
	}
	batch := NewBatch(n, 1)
	copy(batch.F[0], x)
	batch.QuantizeK()

	row := make([]float32, n)
	DequantizeQ3_K(w, n, row)
	var want float32
	for i := range row {
		// Against the *quantized* activation, which is what the integer
		// product reads. Comparing to the floats would measure Q8_K's rounding
		// and call it a fault in the kernel.
		want += row[i] * dequantK(batch, 0, i)
	}

	got := dotQ3_KGo(w, batch.QK[0:], batch.BSums[0:], batch.KScales[0:], n)
	if diff := math.Abs(float64(got - want)); diff > 1e-3*math.Abs(float64(want)) {
		t.Fatalf("integer product %g, dequantiser %g", got, want)
	}
	if other := dotQ3_K(w, batch.QK[0:], batch.BSums[0:], batch.KScales[0:], n); other != got {
		t.Fatalf("the dispatched product answered %g where the portable one answered %g", other, got)
	}
}

// dequantK is one weight of a batch's Q8_K form back as a float.
func dequantK(b *Batch, column, i int) float32 {
	return float32(b.QK[column*b.Width+i]) * b.KScales[column*b.Width/SuperBlock+i/SuperBlock]
}
