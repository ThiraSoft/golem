package nn

import (
	"encoding/binary"
	"math"
	"math/rand"
	"testing"
)

// randomTernary is rows of either packing with codes 0..2 and a sane fp16
// scale a block.
func randomTernary(r *rand.Rand, quant Quant, rows, cols int) []byte {
	if quant == PQ2_0 {
		stride := cols / TernaryBlock * pq2_0BlockBytes
		data := make([]byte, rows*stride)
		for blk := 0; blk < len(data)/pq2_0BlockBytes; blk++ {
			b := data[blk*pq2_0BlockBytes:]
			binary.LittleEndian.PutUint16(b, floatToHalf(0.01+r.Float32()))
			for i := 2; i < 34; i++ {
				b[i] = byte(r.Intn(3) | r.Intn(3)<<2 | r.Intn(3)<<4 | r.Intn(3)<<6)
			}
		}
		return data
	}
	stride := cols / TernaryBlock * ptq1_0BlockBytes
	data := make([]byte, rows*stride)
	values := make([]float32, cols)
	for row := 0; row < rows; row++ {
		for i := range values {
			values[i] = float32(r.Intn(3)-1) * (0.5 + float32(i/TernaryBlock))
		}
		quantizeRowPTQ1_0Ref(values, data[row*stride:(row+1)*stride])
	}
	return data
}

// The integer forms against the float product of the dequantized row and the
// dequantized Q8_K activation: the same numbers, so the gap is rounding alone.
func TestTernaryDotsMatchTheDequantizedProduct(t *testing.T) {
	r := rand.New(rand.NewSource(11))
	const rows, cols = 8, 4 * SuperBlock
	for _, quant := range []Quant{PQ2_0, PTQ1_0} {
		m := Matrix{Data: randomTernary(r, quant, rows, cols), Quant: quant, Rows: rows, Cols: cols}
		b := NewBatch(cols, 1)
		for i := range b.F[0] {
			b.F[0][i] = r.Float32()*2 - 1
		}
		b.QuantizeK()
		act := ternaryActivation(b, quant == PQ2_0)
		row := make([]float32, cols)
		for i := 0; i < rows; i++ {
			m.Row(i, row)
			var want float64
			for k, w := range row {
				want += float64(w) * float64(b.QK[k]) * float64(b.KScales[k/SuperBlock])
			}
			data := m.Data[i*m.RowBytes() : (i+1)*m.RowBytes()]
			goForm := dotPQ2_0Go
			if quant == PTQ1_0 {
				goForm = dotPTQ1_0Go
			}
			for name, got := range map[string]float32{
				"go":     goForm(data, &act[0], cols),
				"kernel": map[Quant]func([]byte, *ternaryColumn, int) float32{PQ2_0: dotPQ2_0, PTQ1_0: dotPTQ1_0}[quant](data, &act[0], cols),
			} {
				if e := math.Abs(float64(got) - want); e > 1e-4*math.Max(1, math.Abs(want)) {
					t.Fatalf("%s %s row %d: %v against %v", quant, name, i, got, want)
				}
			}
		}
	}
}

// MatVecBatch, rotation and all, against the float product of the rotated
// activation as the Q8_K form holds it: what is left is rounding.
func TestTernaryMatVecBatchMatchesTheFloatProduct(t *testing.T) {
	r := rand.New(rand.NewSource(12))
	const rows, cols, size = 37, 2 * SuperBlock, 3
	for _, quant := range []Quant{PQ2_0, PTQ1_0} {
		pre := make([]float32, cols)
		for i := range pre {
			pre[i] = float32(r.Intn(2)*2 - 1)
		}
		m := Matrix{Data: randomTernary(r, quant, rows, cols), Quant: quant, Rows: rows, Cols: cols, Pre: pre, HadGroup: 128}
		b := NewBatch(cols, size)
		for c := range b.F {
			for i := range b.F[c] {
				b.F[c][i] = r.Float32()*2 - 1
			}
		}
		got := make([][]float32, size)
		for c := range got {
			got[c] = make([]float32, rows)
		}
		m.MatVecBatch(b, got)

		rotated := m.prepare(b)
		rotated.QuantizeK()
		for i := 0; i < rows; i++ {
			// The weight as stored, not as Row hands it back: the rotated
			// activation meets the stored form.
			stored := make([]float32, cols)
			switch quant {
			case PQ2_0:
				DequantizePQ2_0(m.Data[i*m.RowBytes():], cols, stored)
			case PTQ1_0:
				DequantizePTQ1_0(m.Data[i*m.RowBytes():], cols, stored)
			}
			for c := range got {
				var want float64
				for k, w := range stored {
					want += float64(w) * float64(rotated.QK[c*cols+k]) * float64(rotated.KScales[(c*cols+k)/SuperBlock])
				}
				if e := math.Abs(float64(got[c][i]) - want); e > 1e-4*math.Max(1, math.Abs(want)) {
					t.Fatalf("%s column %d row %d: %v against %v", quant, c, i, got[c][i], want)
				}
			}
		}
	}
}

func BenchmarkTernaryDot(b *testing.B) {
	r := rand.New(rand.NewSource(13))
	const cols = 5120
	for _, quant := range []Quant{PQ2_0, PTQ1_0} {
		m := Matrix{Data: randomTernary(r, quant, 1, cols), Quant: quant, Rows: 1, Cols: cols}
		x := NewBatch(cols, 1)
		for i := range x.F[0] {
			x.F[0][i] = r.Float32()*2 - 1
		}
		x.QuantizeK()
		act := ternaryActivation(x, quant == PQ2_0)
		dot := dotPQ2_0
		if quant == PTQ1_0 {
			dot = dotPTQ1_0
		}
		b.Run(quant.String(), func(b *testing.B) {
			b.SetBytes(int64(m.RowBytes()))
			for i := 0; i < b.N; i++ {
				dot(m.Data, &act[0], cols)
			}
		})
	}
}
