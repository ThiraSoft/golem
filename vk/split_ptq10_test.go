package vk

import (
	"encoding/binary"
	"math/rand"
	"os"
	"testing"

	"github.com/ThiraSoft/golem/nn"
	"github.com/ThiraSoft/golem/tensors"
)

var ptq10Pow3 = [5]uint32{1, 3, 9, 27, 81}

// decodeTrit decodes one trit from a byte given power index n (0..4).
func decodeTrit(b byte, n int) uint32 {
	return ((uint32(b) * ptq10Pow3[n] & 0xFF) * 3) >> 8
}

// TestPTQ10ByteAndPowerMatchesDequantize tests that decoding using the
// (byteIndex, powerIndex) table matches nn.DequantizePTQ1_0 weight for weight.
func TestPTQ10ByteAndPowerMatchesDequantize(t *testing.T) {
	// 1. Verify sub-block table equivalence against ptq10ByteAndPower for all 128 weights.
	for v := 0; v < 128; v++ {
		bIdx, pIdx := ptq10ByteAndPower(v)
		sub := v / 32
		offset := v % 32

		var expectedBIdx, expectedPIdx int
		switch sub {
		case 0:
			if offset < 16 {
				expectedBIdx = offset
				expectedPIdx = 0
			} else {
				expectedBIdx = offset - 16
				expectedPIdx = 1
			}
		case 1:
			if offset < 16 {
				expectedBIdx = offset
				expectedPIdx = 2
			} else {
				expectedBIdx = offset - 16
				expectedPIdx = 3
			}
		case 2:
			if offset < 16 {
				expectedBIdx = offset
				expectedPIdx = 4
			} else if offset < 24 {
				expectedBIdx = 16 + (offset - 16)
				expectedPIdx = 0
			} else {
				expectedBIdx = 16 + (offset - 24)
				expectedPIdx = 1
			}
		case 3:
			if offset < 8 {
				expectedBIdx = 16 + offset
				expectedPIdx = 2
			} else if offset < 16 {
				expectedBIdx = 16 + (offset - 8)
				expectedPIdx = 3
			} else if offset < 24 {
				expectedBIdx = 16 + (offset - 16)
				expectedPIdx = 4
			} else {
				qhIdx := (offset - 24) % 2
				qhPow := (offset - 24) / 2
				expectedBIdx = 24 + qhIdx
				expectedPIdx = qhPow
			}
		}

		if bIdx != expectedBIdx || pIdx != expectedPIdx {
			t.Fatalf("v=%d (sub=%d, offset=%d): got byte=%d pow=%d, expected byte=%d pow=%d",
				v, sub, offset, bIdx, pIdx, expectedBIdx, expectedPIdx)
		}
	}

	// 2. Verify on random blocks against nn.DequantizePTQ1_0.
	rng := rand.New(rand.NewSource(99))
	block := make([]byte, 28)
	want := make([]float32, 128)
	got := make([]float32, 128)

	for trial := 0; trial < 100; trial++ {
		rng.Read(block)
		binary.LittleEndian.PutUint16(block[26:28], nn.Narrow(rng.Float32()*0.1))
		d := nn.Widen(binary.LittleEndian.Uint16(block[26:28]))

		nn.DequantizePTQ1_0(block, 128, want)

		for v := 0; v < 128; v++ {
			bIdx, pIdx := ptq10ByteAndPower(v)
			trit := decodeTrit(block[bIdx], pIdx)
			got[v] = float32(int(trit)-1) * d
			if got[v] != want[v] {
				t.Fatalf("trial %d weight %d: got %g, want %g (byte=%d, pow=%d, raw=%d)",
					trial, v, got[v], want[v], bIdx, pIdx, block[bIdx])
			}
		}
	}

	// 3. Verify on real tensors from the PTQ1_0 checkpoint.
	path := os.Getenv("GOLEM_MODEL_BONSAI_PTQ10")
	if path == "" {
		path = "/mnt/data/LLMs_models/prism-ml/Ternary-Bonsai-2-27B-gguf/Ternary-Bonsai-2-27B-PTQ1_0.gguf"
	}
	if _, err := os.Stat(path); err != nil {
		t.Skipf("%s absent", path)
	}
	g, err := tensors.OpenGGUF(path)
	if err != nil {
		t.Fatal(err)
	}
	defer g.Close()

	for _, name := range []string{"blk.0.ffn_down.weight", "blk.0.attn_qkv.weight"} {
		tensor, err := g.Get(name)
		if err != nil {
			t.Fatalf("missing tensor %s: %v", name, err)
		}
		cols := tensor.Shape[0]
		rCount := min(tensor.Shape[1], 8)
		rowFloats := make([]float32, cols)
		for r := 0; r < rCount; r++ {
			rowBytes := cols / nn.TernaryBlock * 28
			rowData := tensor.Raw[r*rowBytes : (r+1)*rowBytes]
			nn.DequantizePTQ1_0(rowData, cols, rowFloats)

			for tb := 0; tb < cols/nn.TernaryBlock; tb++ {
				blk := rowData[tb*28 : (tb+1)*28]
				d := nn.Widen(binary.LittleEndian.Uint16(blk[26:28]))
				for v := 0; v < 128; v++ {
					bIdx, pIdx := ptq10ByteAndPower(v)
					trit := decodeTrit(blk[bIdx], pIdx)
					val := float32(int(trit)-1) * d
					wantVal := rowFloats[tb*128+v]
					if val != wantVal {
						t.Fatalf("tensor %s row %d weight %d: got %g, want %g",
							name, r, tb*128+v, val, wantVal)
					}
				}
			}
		}
	}
}

// TestSplitPTQ1_0 verifies the card layout copy.
func TestSplitPTQ1_0(t *testing.T) {
	for _, cols := range []int{128, 256, 5120} {
		stride := rowBytesPTQ1_0(cols)
		src := make([]byte, 2*stride)
		for i := range src {
			src[i] = byte(i)
		}
		dst := splitPTQ1_0(src, 2, cols)
		if len(dst) != len(src) {
			t.Fatalf("len(dst) = %d, want %d", len(dst), len(src))
		}
		for i := range src {
			if dst[i] != src[i] {
				t.Fatalf("dst[%d] = %d, want %d", i, dst[i], src[i])
			}
		}
	}
}

func decodeWord4(w uint32, p uint32) uint32 {
	decodeT := func(b uint32, pow uint32) uint32 {
		return ((b * pow & 0xFF) * 3) >> 8
	}
	t0 := decodeT(w&0xFF, p)
	t1 := decodeT((w>>8)&0xFF, p)
	t2 := decodeT((w>>16)&0xFF, p)
	t3 := decodeT(w>>24, p)
	return t0 | (t1 << 8) | (t2 << 16) | (t3 << 24)
}

// TestPTQ10ShaderUnpackMatchesDequantize tests the exact logic used in the
// shaders to construct lo[4] and hi[4] per sub-block.
func TestPTQ10ShaderUnpackMatchesDequantize(t *testing.T) {
	rng := rand.New(rand.NewSource(12345))
	block := make([]byte, 28)
	want := make([]float32, 128)

	for trial := 0; trial < 100; trial++ {
		rng.Read(block)
		binary.LittleEndian.PutUint16(block[26:28], nn.Narrow(rng.Float32()*0.1))
		d := nn.Widen(binary.LittleEndian.Uint16(block[26:28]))
		nn.DequantizePTQ1_0(block, 128, want)

		w := make([]uint32, 7)
		for i := 0; i < 7; i++ {
			w[i] = binary.LittleEndian.Uint32(block[i*4 : (i+1)*4])
		}

		for sub := 0; sub < 4; sub++ {
			var lo, hi [4]uint32
			switch sub {
			case 0:
				lo[0] = decodeWord4(w[0], 1)
				lo[1] = decodeWord4(w[1], 1)
				lo[2] = decodeWord4(w[2], 1)
				lo[3] = decodeWord4(w[3], 1)

				hi[0] = decodeWord4(w[0], 3)
				hi[1] = decodeWord4(w[1], 3)
				hi[2] = decodeWord4(w[2], 3)
				hi[3] = decodeWord4(w[3], 3)
			case 1:
				lo[0] = decodeWord4(w[0], 9)
				lo[1] = decodeWord4(w[1], 9)
				lo[2] = decodeWord4(w[2], 9)
				lo[3] = decodeWord4(w[3], 9)

				hi[0] = decodeWord4(w[0], 27)
				hi[1] = decodeWord4(w[1], 27)
				hi[2] = decodeWord4(w[2], 27)
				hi[3] = decodeWord4(w[3], 27)
			case 2:
				lo[0] = decodeWord4(w[0], 81)
				lo[1] = decodeWord4(w[1], 81)
				lo[2] = decodeWord4(w[2], 81)
				lo[3] = decodeWord4(w[3], 81)

				hi[0] = decodeWord4(w[4], 1)
				hi[1] = decodeWord4(w[5], 1)
				hi[2] = decodeWord4(w[4], 3)
				hi[3] = decodeWord4(w[5], 3)
			case 3:
				lo[0] = decodeWord4(w[4], 9)
				lo[1] = decodeWord4(w[5], 9)
				lo[2] = decodeWord4(w[4], 27)
				lo[3] = decodeWord4(w[5], 27)

				hi[0] = decodeWord4(w[4], 81)
				hi[1] = decodeWord4(w[5], 81)

				decodeT := func(b uint32, pow uint32) uint32 {
					return ((b * pow & 0xFF) * 3) >> 8
				}
				h0 := w[6] & 0xFF
				h1 := (w[6] >> 8) & 0xFF
				hi[2] = decodeT(h0, 1) | (decodeT(h1, 1) << 8) | (decodeT(h0, 3) << 16) | (decodeT(h1, 3) << 24)
				hi[3] = decodeT(h0, 9) | (decodeT(h1, 9) << 8) | (decodeT(h0, 27) << 16) | (decodeT(h1, 27) << 24)
			}

			// Verify weights against want
			for u := 0; u < 4; u++ {
				for bytePos := 0; bytePos < 4; bytePos++ {
					tLo := (lo[u] >> uint(bytePos*8)) & 0xFF
					wLo := float32(int(tLo)-1) * d
					idxLo := sub*32 + u*4 + bytePos
					if wLo != want[idxLo] {
						t.Fatalf("trial %d sub %d lo[%d][%d]: got %g, want %g",
							trial, sub, u, bytePos, wLo, want[idxLo])
					}

					tHi := (hi[u] >> uint(bytePos*8)) & 0xFF
					wHi := float32(int(tHi)-1) * d
					idxHi := sub*32 + 16 + u*4 + bytePos
					if wHi != want[idxHi] {
						t.Fatalf("trial %d sub %d hi[%d][%d]: got %g, want %g",
							trial, sub, u, bytePos, wHi, want[idxHi])
					}
				}
			}
		}
	}
}

// TestPTQ10MatmulCoopUnpackMatchesDequantize verifies the exact prefetch and stageTile
// logic used for PTQ10 in matmul_coop.comp.
func TestPTQ10MatmulCoopUnpackMatchesDequantize(t *testing.T) {
	packByte4 := func(w uint32, p uint32) uint32 {
		t0 := (((w & 0xFF) * p & 0xFF) * 3) >> 8
		t1 := ((((w >> 8) & 0xFF) * p & 0xFF) * 3) >> 8
		t2 := ((((w >> 16) & 0xFF) * p & 0xFF) * 3) >> 8
		t3 := (((w >> 24) * p & 0xFF) * 3) >> 8
		return t0 | (t1 << 2) | (t2 << 4) | (t3 << 6)
	}
	decodeT := func(b uint32, pow uint32) uint32 {
		return ((b * pow & 0xFF) * 3) >> 8
	}

	rng := rand.New(rand.NewSource(54321))
	block := make([]byte, 28)
	want := make([]float32, 128)

	for trial := 0; trial < 100; trial++ {
		rng.Read(block)
		binary.LittleEndian.PutUint16(block[26:28], nn.Narrow(rng.Float32()*0.1))
		d := nn.Widen(binary.LittleEndian.Uint16(block[26:28]))
		nn.DequantizePTQ1_0(block, 128, want)

		w := make([]uint32, 7)
		for i := 0; i < 7; i++ {
			w[i] = binary.LittleEndian.Uint32(block[i*4 : (i+1)*4])
		}

		for sub := 0; sub < 4; sub++ {
			var pwX, pwY uint32
			switch sub {
			case 0:
				pwX = packByte4(w[0], 1) | (packByte4(w[1], 1) << 8) | (packByte4(w[2], 1) << 16) | (packByte4(w[3], 1) << 24)
				pwY = packByte4(w[0], 3) | (packByte4(w[1], 3) << 8) | (packByte4(w[2], 3) << 16) | (packByte4(w[3], 3) << 24)
			case 1:
				pwX = packByte4(w[0], 9) | (packByte4(w[1], 9) << 8) | (packByte4(w[2], 9) << 16) | (packByte4(w[3], 9) << 24)
				pwY = packByte4(w[0], 27) | (packByte4(w[1], 27) << 8) | (packByte4(w[2], 27) << 16) | (packByte4(w[3], 27) << 24)
			case 2:
				pwX = packByte4(w[0], 81) | (packByte4(w[1], 81) << 8) | (packByte4(w[2], 81) << 16) | (packByte4(w[3], 81) << 24)
				pwY = packByte4(w[4], 1) | (packByte4(w[5], 1) << 8) | (packByte4(w[4], 3) << 16) | (packByte4(w[5], 3) << 24)
			case 3:
				pwX = packByte4(w[4], 9) | (packByte4(w[5], 9) << 8) | (packByte4(w[4], 27) << 16) | (packByte4(w[5], 27) << 24)
				h0 := w[6] & 0xFF
				h1 := (w[6] >> 8) & 0xFF
				b2 := decodeT(h0, 1) | (decodeT(h1, 1) << 2) | (decodeT(h0, 3) << 4) | (decodeT(h1, 3) << 6)
				b3 := decodeT(h0, 9) | (decodeT(h1, 9) << 2) | (decodeT(h0, 27) << 4) | (decodeT(h1, 27) << 6)
				pwY = packByte4(w[4], 81) | (packByte4(w[5], 81) << 8) | (b2 << 16) | (b3 << 24)
			}

			// Now simulate stageTile() unpacking, exactly like PQ20:
			for u := uint32(0); u < 4; u++ {
				x0 := (pwX >> (8 * u)) & 0xFF
				x1 := (pwY >> (8 * u)) & 0xFF
				w0 := (x0 & 3) | ((x0 & 12) << 6) | ((x0 & 48) << 12) | ((x0 & 192) << 18)
				w1 := (x1 & 3) | ((x1 & 12) << 6) | ((x1 & 48) << 12) | ((x1 & 192) << 18)

				// In stageTile(): unpackUnorm4x8(w0) * 255.0 - 1.0 gives 4 floats:
				for bytePos := uint32(0); bytePos < 4; bytePos++ {
					tLo := (w0 >> (bytePos * 8)) & 0xFF
					v0 := (float32(tLo) - 1.0) * d
					idxLo := sub*32 + int(u)*4 + int(bytePos)
					if v0 != want[idxLo] {
						t.Fatalf("trial %d sub %d u %d byte %d: got %g, want %g",
							trial, sub, u, bytePos, v0, want[idxLo])
					}

					tHi := (w1 >> (bytePos * 8)) & 0xFF
					v1 := (float32(tHi) - 1.0) * d
					idxHi := sub*32 + 16 + int(u)*4 + int(bytePos)
					if v1 != want[idxHi] {
						t.Fatalf("trial %d sub %d u %d byte %d: got %g, want %g",
							trial, sub, u, bytePos, v1, want[idxHi])
					}
				}
			}
		}
	}
}

