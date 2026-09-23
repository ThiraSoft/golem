package nn

import (
	"encoding/binary"
	"math"
	"math/rand"
	"os"
	"testing"

	"github.com/ThiraSoft/golem/tensors"
)

// quantizeRowPTQ1_0Ref is the reference PTQ1_0 quantizer from ggml-quants.c.
func quantizeRowPTQ1_0Ref(x []float32, out []byte) {
	n := len(x)
	if n%TernaryBlock != 0 {
		panic("quantizeRowPTQ1_0Ref: length must be multiple of 128")
	}
	nb := n / TernaryBlock
	xOffset := 0
	for i := 0; i < nb; i++ {
		var amax float32
		for j := 0; j < TernaryBlock; j++ {
			v := float32(math.Abs(float64(x[xOffset+j])))
			if v > amax {
				amax = v
			}
		}
		d := amax
		var id float32
		if d != 0 {
			id = 1.0 / d
		}
		block := out[i*ptq1_0BlockBytes : (i+1)*ptq1_0BlockBytes]
		binary.LittleEndian.PutUint16(block[26:28], floatToHalf(d))

		j := 0
		for s := 0; s < 3; s++ {
			c := ptq1_0_stages[s]
			for ; j+c <= 24; j += c {
				for m := 0; m < c; m++ {
					var q uint32
					for n := 0; n < 5; n++ {
						val := x[xOffset+m+n*c] * id
						xi := int(math.Round(float64(val))) + 1
						q *= 3
						q += uint32(xi)
					}
					q = (q*256 + 242) / 243
					block[j+m] = byte(q)
				}
				xOffset += 5 * c
			}
		}
		for h := 0; h < 2; h++ {
			var q uint32
			for m := 0; m < 4; m++ {
				val := x[xOffset+h+m*2] * id
				xi := int(math.Round(float64(val))) + 1
				q *= 3
				q += uint32(xi)
			}
			q *= 3
			q = (q*256 + 242) / 243
			block[24+h] = byte(q)
		}
		xOffset += 4 * 2
	}
}

func TestPQ2_0HandBuilt(t *testing.T) {
	// A block of 128 weights with scale d = 2.0.
	block := make([]byte, pq2_0BlockBytes)
	scale := float32(2.0)
	binary.LittleEndian.PutUint16(block[:2], floatToHalf(scale))

	// Byte 0 of qs encodes 4 two-bit quants:
	// bit 0-1: 0 -> (0-1)*2 = -2.0
	// bit 2-3: 1 -> (1-1)*2 = 0.0
	// bit 4-5: 2 -> (2-1)*2 = 2.0
	// bit 6-7: 3 -> (3-1)*2 = 4.0
	// 0b11_10_01_00 = 0xE4
	block[2] = 0xE4
	// Fill remaining 31 bytes with code 1 (0b01_01_01_01 = 0x55) -> 0.0
	for i := 3; i < 34; i++ {
		block[i] = 0x55
	}

	out := make([]float32, TernaryBlock)
	DequantizePQ2_0(block, TernaryBlock, out)

	want := [4]float32{-2.0, 0.0, 2.0, 4.0}
	for i := 0; i < 4; i++ {
		if out[i] != want[i] {
			t.Fatalf("out[%d] = %g, want %g", i, out[i], want[i])
		}
	}
	for i := 4; i < TernaryBlock; i++ {
		if out[i] != 0.0 {
			t.Fatalf("out[%d] = %g, want 0.0", i, out[i])
		}
	}
}

func TestPTQ1_0HandComputedByteAndRoundTrip(t *testing.T) {
	// 1. Check one fully hand-computed byte.
	// In stage 1 (c=16), for m=0, the 5 trits at positions 0, 16, 32, 48, 64
	// are chosen as 2, 1, 0, 2, 1 (corresponding to weight values +1, 0, -1, +1, 0 with d=1).
	// xi = 2, 1, 0, 2, 1 -> q = ((((2*3+1)*3+0)*3+2)*3+1) = 196
	// (196 * 256 + 242) / 243 = 50418 / 243 = 207 (0xCF).
	x := make([]float32, TernaryBlock)
	x[0] = 1.0   // trit 2
	x[16] = 0.0  // trit 1
	x[32] = -1.0 // trit 0
	x[48] = 1.0  // trit 2
	x[64] = 0.0  // trit 1

	encoded := make([]byte, ptq1_0BlockBytes)
	quantizeRowPTQ1_0Ref(x, encoded)

	if encoded[0] != 207 {
		t.Fatalf("encoded byte 0 = %d, want 207 (0xCF)", encoded[0])
	}

	decoded := make([]float32, TernaryBlock)
	DequantizePTQ1_0(encoded, TernaryBlock, decoded)

	if decoded[0] != 1.0 || decoded[16] != 0.0 || decoded[32] != -1.0 || decoded[48] != 1.0 || decoded[64] != 0.0 {
		t.Fatalf("decoded values mismatch: got [%g, %g, %g, %g, %g], want [1, 0, -1, 1, 0]",
			decoded[0], decoded[16], decoded[32], decoded[48], decoded[64])
	}

	// 2. Test round-trip decode(encode(t)) == t for random trit vectors.
	rng := rand.New(rand.NewSource(42))
	tritChoices := []float32{-1.0, 0.0, 1.0}
	scale := float32(1.5)

	for trial := 0; trial < 10; trial++ {
		tVec := make([]float32, TernaryBlock)
		for i := range tVec {
			tVec[i] = tritChoices[rng.Intn(3)] * scale
		}
		quantizeRowPTQ1_0Ref(tVec, encoded)
		DequantizePTQ1_0(encoded, TernaryBlock, decoded)
		for i := 0; i < TernaryBlock; i++ {
			if decoded[i] != tVec[i] {
				t.Fatalf("trial %d: mismatch at index %d: got %g, want %g", trial, i, decoded[i], tVec[i])
			}
		}
	}
}

func TestTernaryPackingsAgree(t *testing.T) {
	const (
		pqPath = "/mnt/data/LLMs_models/prism-ml/Ternary-Bonsai-2-27B-gguf/Ternary-Bonsai-2-27B-PQ2_0.gguf"
		ptPath = "/mnt/data/LLMs_models/prism-ml/Ternary-Bonsai-2-27B-gguf/Ternary-Bonsai-2-27B-PTQ1_0.gguf"
	)
	if _, err := os.Stat(pqPath); err != nil {
		t.Skipf("%s is not present", pqPath)
	}
	if _, err := os.Stat(ptPath); err != nil {
		t.Skipf("%s is not present", ptPath)
	}

	fPQ, err := tensors.OpenGGUF(pqPath)
	if err != nil {
		t.Fatal(err)
	}
	defer fPQ.Close()

	fPT, err := tensors.OpenGGUF(ptPath)
	if err != nil {
		t.Fatal(err)
	}
	defer fPT.Close()

	tensorsToTest := []string{
		"blk.0.ffn_down.weight",
		"blk.3.attn_q.weight",
		"token_embd.weight",
	}

	for _, name := range tensorsToTest {
		tPQ, err := fPQ.Get(name)
		if err != nil {
			t.Fatalf("PQ file missing %s: %v", name, err)
		}
		tPT, err := fPT.Get(name)
		if err != nil {
			t.Fatalf("PT file missing %s: %v", name, err)
		}

		qPQ, ok := QuantOf(tPQ.DType)
		if !ok || qPQ != PQ2_0 {
			t.Fatalf("%s in PQ file has unexpected type %s", name, tPQ.DType)
		}
		qPT, ok := QuantOf(tPT.DType)
		if !ok || qPT != PTQ1_0 {
			t.Fatalf("%s in PT file has unexpected type %s", name, tPT.DType)
		}

		cols := int(tPQ.Shape[0])
		rows := int(tPQ.Shape[1])
		mPQ := Matrix{Data: tPQ.Raw, Quant: qPQ, Rows: rows, Cols: cols}
		mPT := Matrix{Data: tPT.Raw, Quant: qPT, Rows: rows, Cols: cols}

		outPQ := make([]float32, cols)
		outPT := make([]float32, cols)

		checkRows := min(8, rows)
		for r := 0; r < checkRows; r++ {
			mPQ.Row(r, outPQ)
			mPT.Row(r, outPT)
			for i := 0; i < cols; i++ {
				if math.Float32bits(outPQ[i]) != math.Float32bits(outPT[i]) {
					t.Fatalf("%s row %d col %d mismatch: PQ %g (%#x) vs PT %g (%#x)",
						name, r, i, outPQ[i], math.Float32bits(outPQ[i]), outPT[i], math.Float32bits(outPT[i]))
				}
			}
		}

		// Also require that no PQ2_0 code 3 appears in those rows' bytes.
		var code3Count int
		stride := mPQ.RowBytes()
		for r := 0; r < checkRows; r++ {
			rowBytes := tPQ.Raw[r*stride : (r+1)*stride]
			for b := 0; b < cols/TernaryBlock; b++ {
				block := rowBytes[b*pq2_0BlockBytes : (b+1)*pq2_0BlockBytes]
				qs := block[2:34]
				for _, bVal := range qs {
					for shift := 0; shift < 8; shift += 2 {
						if (bVal>>shift)&3 == 3 {
							code3Count++
						}
					}
				}
			}
		}
		if code3Count > 0 {
			t.Fatalf("%s: found %d PQ2_0 quants with code 3 (expected only codes 0..2)", name, code3Count)
		}
	}
}

func TestTernaryMatVec(t *testing.T) {
	// Test that rows matches float dot of Row output, for 1 and 3 batch columns.
	const (
		rows = 4
		cols = 256
	)
	rng := rand.New(rand.NewSource(1234))

	for _, quant := range []Quant{PQ2_0, PTQ1_0} {
		var data []byte
		if quant == PQ2_0 {
			stride := cols / TernaryBlock * pq2_0BlockBytes
			data = make([]byte, rows*stride)
			for r := 0; r < rows; r++ {
				for b := 0; b < cols/TernaryBlock; b++ {
					blk := data[r*stride+b*pq2_0BlockBytes : r*stride+(b+1)*pq2_0BlockBytes]
					binary.LittleEndian.PutUint16(blk[:2], floatToHalf(1.0))
					for i := 2; i < 34; i++ {
						// Use codes 0, 1, 2 only
						c0 := byte(rng.Intn(3))
						c1 := byte(rng.Intn(3))
						c2 := byte(rng.Intn(3))
						c3 := byte(rng.Intn(3))
						blk[i] = c0 | (c1 << 2) | (c2 << 4) | (c3 << 6)
					}
				}
			}
		} else {
			stride := cols / TernaryBlock * ptq1_0BlockBytes
			data = make([]byte, rows*stride)
			rowFloats := make([]float32, cols)
			for r := 0; r < rows; r++ {
				for i := range rowFloats {
					rowFloats[i] = float32(rng.Intn(3) - 1)
				}
				quantizeRowPTQ1_0Ref(rowFloats, data[r*stride:(r+1)*stride])
			}
		}

		m := Matrix{Data: data, Quant: quant, Rows: rows, Cols: cols}

		for _, numCols := range []int{1, 3} {
			b := &Batch{
				Size:  numCols,
				Width: cols,
				F:     make([][]float32, numCols),
			}
			for c := 0; c < numCols; c++ {
				b.F[c] = make([]float32, cols)
				for i := 0; i < cols; i++ {
					b.F[c][i] = rng.Float32()*2 - 1
				}
			}

			ys := make([][]float32, numCols)
			for c := 0; c < numCols; c++ {
				ys[c] = make([]float32, rows)
			}

			m.rows(b, ys, 0, rows)

			rowBuf := make([]float32, cols)
			for r := 0; r < rows; r++ {
				m.Row(r, rowBuf)
				for c := 0; c < numCols; c++ {
					want := DotF32(rowBuf, b.F[c])
					got := ys[c][r]
					diff := float32(math.Abs(float64(got - want)))
					if diff > 1e-4 {
						t.Fatalf("%s numCols=%d row=%d col=%d: got %g, want %g (diff %g)",
							quant, numCols, r, c, got, want, diff)
					}
				}
			}
		}
	}
}
