package vk

import (
	"encoding/binary"
	"math/rand"
	"os"
	"testing"

	"github.com/ThiraSoft/golem/nn"
	"github.com/ThiraSoft/golem/tensors"
)

// TestSplitPQ2_0HoldsTheSameWeights verifies that dequantizing from the card
// layout in Go produces identical weights to nn.DequantizePQ2_0, both on
// synthetic rows and on a real tensor from the Bonsai checkpoint.
func TestSplitPQ2_0HoldsTheSameWeights(t *testing.T) {
	const rows = 3
	for _, cols := range []int{128, 256, 1024, 5120} {
		t.Run(itoa(cols), func(t *testing.T) { splitPQ20HoldsTheSameWeights(t, rows, cols) })
	}
	t.Run("real_tensor", func(t *testing.T) {
		path := os.Getenv("GOLEM_MODEL_BONSAI_PQ20")
		if path == "" {
			path = "/mnt/data/LLMs_models/prism-ml/Ternary-Bonsai-2-27B-gguf/Ternary-Bonsai-2-27B-PQ2_0.gguf"
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
			cols, rCount := tensor.Shape[0], min(tensor.Shape[1], 16)
			packed := splitPQ2_0(tensor.Raw, rCount, cols)
			checkSplitPQ20Layout(t, tensor.Raw, packed, rCount, cols)
		}
	})
}

func splitPQ20HoldsTheSameWeights(t *testing.T, rows, cols int) {
	const blockBytes = 34
	ntb := cols / nn.TernaryBlock

	rng := rand.New(rand.NewSource(42))
	src := make([]byte, rows*ntb*blockBytes)
	rng.Read(src)
	for r := 0; r < rows; r++ {
		for tb := 0; tb < ntb; tb++ {
			at := (r*ntb + tb) * blockBytes
			binary.LittleEndian.PutUint16(src[at:], nn.Narrow(rng.Float32()*0.05))
		}
	}

	packed := splitPQ2_0(src, rows, cols)
	if got, want := len(packed), rows*rowBytesPQ2_0(cols); got != want {
		t.Fatalf("packed length %d, want %d", got, want)
	}
	checkSplitPQ20Layout(t, src, packed, rows, cols)
}

func checkSplitPQ20Layout(t *testing.T, src, packed []byte, rows, cols int) {
	t.Helper()
	const blockBytes = 34
	ntb := cols / nn.TernaryBlock
	stride := rowBytesPQ2_0(cols)
	scaleBytes := pq20BlockBase(ntb)

	want := make([]float32, cols)
	for r := 0; r < rows; r++ {
		nn.DequantizePQ2_0(src[r*ntb*blockBytes:(r+1)*ntb*blockBytes], cols, want)
		row := packed[r*stride : (r+1)*stride]
		blocks := row[scaleBytes:]

		for tb := 0; tb < ntb; tb++ {
			d := nn.Widen(binary.LittleEndian.Uint16(row[2*tb:]))
			qs := blocks[tb*32 : (tb+1)*32]

			for sub := 0; sub < 4; sub++ {
				pack := qs[sub*8 : (sub+1)*8]
				w0 := binary.LittleEndian.Uint32(pack[:4])
				w1 := binary.LittleEndian.Uint32(pack[4:])

				for j := 0; j < 16; j++ {
					q0 := int((w0 >> uint(2*j)) & 3)
					got0 := float32(q0-1) * d
					idx0 := tb*nn.TernaryBlock + sub*32 + j
					if got0 != want[idx0] {
						t.Fatalf("row %d weight %d: got %g, want %g", r, idx0, got0, want[idx0])
					}

					q1 := int((w1 >> uint(2*j)) & 3)
					got1 := float32(q1-1) * d
					idx1 := tb*nn.TernaryBlock + sub*32 + 16 + j
					if got1 != want[idx1] {
						t.Fatalf("row %d weight %d: got %g, want %g", r, idx1, got1, want[idx1])
					}
				}
			}
		}
	}
}
