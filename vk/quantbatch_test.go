package vk

import (
	"fmt"
	"math"
	"math/rand"
	"testing"

	"github.com/ThiraSoft/golem/nn"
)

// TestQuantBatchMatchesCPU is the layout check.
//
// The kernel reads a column whole and nn.Batch keeps its quantized form block
// major and then column, so staging is a transposition and not a copy. A
// transposition that is wrong by one column answers fluently — every number is
// a real product of real weights, just not the ones asked for — which is why
// this compares against nn's own product rather than checking a shape.
func TestQuantBatchMatchesCPU(t *testing.T) {
	d := open(t)
	defer d.Close()

	for _, q := range []nn.Quant{nn.Q8_0, nn.Q4_0} {
		for _, width := range []int{1, 2, 4, 8} {
			t.Run(fmt.Sprintf("%s/%d", q, width), func(t *testing.T) {
				const rows, cols = 256, 512
				data := quantData(t, q, rows, cols)
				m := nn.Matrix{Data: data, Quant: q, Rows: rows, Cols: cols}

				batch := nn.NewBatch(cols, width)
				r := rand.New(rand.NewSource(int64(width) * 31))
				for c := 0; c < width; c++ {
					for i := range batch.F[c] {
						batch.F[c][i] = float32(math.Sin(float64(i)*0.11+float64(c))) * (1 + r.Float32())
					}
				}
				batch.Quantize()

				want := make([][]float32, width)
				for c := range want {
					want[c] = make([]float32, rows)
				}
				m.MatVecBatch(batch, want)

				g, err := NewQuantBatch(d, q, data, rows, cols, width)
				if err != nil {
					t.Skipf("no kernel: %v", err)
				}
				defer g.Close()
				for c := 0; c < width; c++ {
					if err := g.SetColumn(c, batch); err != nil {
						t.Fatal(err)
					}
				}
				got := make([][]float32, width)
				for c := range got {
					got[c] = make([]float32, rows)
				}
				if err := g.Run(got); err != nil {
					t.Fatal(err)
				}
				for c := 0; c < width; c++ {
					for i := range want[c] {
						if d := want[c][i] - got[c][i]; d > 1e-2 || d < -1e-2 {
							t.Fatalf("column %d row %d: cpu %g, card %g", c, i, want[c][i], got[c][i])
						}
					}
				}
			})
		}
	}
}

// quantData writes a matrix in that format from real floats.
//
// Random bytes would do for a layout check — both sides read the same bytes —
// except that a random pair of them is a half whose value may be an infinity or
// a NaN, and one of those anywhere makes every comparison below meaningless.
// So the blocks are quantized from a smooth function, the way a real one is.
func quantData(t *testing.T, q nn.Quant, rows, cols int) []byte {
	t.Helper()
	if cols%nn.QuantBlock != 0 {
		t.Fatalf("a row of %d is not whole blocks", cols)
	}
	blocks := rows * cols / nn.QuantBlock
	var out []byte
	values := make([]float32, nn.QuantBlock)
	for b := 0; b < blocks; b++ {
		for i := range values {
			values[i] = float32(math.Sin(float64(b*nn.QuantBlock+i)*0.017)) * float32(1+(b+i)%7)
		}
		var peak float32
		for _, v := range values {
			if a := float32(math.Abs(float64(v))); a > peak {
				peak = a
			}
		}
		switch q {
		case nn.Q8_0:
			scale := peak / 127
			out = append(out, byte(nn.FloatToHalf(scale)), byte(nn.FloatToHalf(scale)>>8))
			for _, v := range values {
				out = append(out, byte(int8(math.Round(float64(v/scale)))))
			}
		case nn.Q4_0:
			// The format's own convention: the scale is signed by the value
			// furthest from zero, and the nibbles are biased by eight.
			far := values[0]
			for _, v := range values {
				if math.Abs(float64(v)) > math.Abs(float64(far)) {
					far = v
				}
			}
			scale := far / -8
			out = append(out, byte(nn.FloatToHalf(scale)), byte(nn.FloatToHalf(scale)>>8))
			nibble := func(v float32) byte {
				n := int(math.Round(float64(v/scale))) + 8
				return byte(min(max(n, 0), 15))
			}
			for i := 0; i < nn.QuantBlock/2; i++ {
				out = append(out, nibble(values[i])|nibble(values[i+nn.QuantBlock/2])<<4)
			}
		default:
			t.Fatalf("this test writes Q8_0 and Q4_0, not %s", q)
		}
	}
	return out
}
