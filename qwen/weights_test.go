package qwen

import "testing"

func loadForTest(t *testing.T) (*Config, *Weights) {
	t.Helper()
	g := openFile(t)
	cfg, err := LoadConfig(g, 4096)
	if err != nil {
		t.Fatal(err)
	}
	w, err := LoadWeights(g, cfg)
	if err != nil {
		t.Fatal(err)
	}
	return cfg, w
}

func TestLoadWeightsBindsEveryBlock(t *testing.T) {
	cfg, w := loadForTest(t)

	if len(w.Blocks) != len(cfg.Blocks) {
		t.Fatalf("%d blocks of weights against %d of geometry", len(w.Blocks), len(cfg.Blocks))
	}
	for i, b := range w.Blocks {
		bc := cfg.Blocks[i]
		if len(b.QNorm) != bc.HeadDim || len(b.KNorm) != bc.HeadDim {
			t.Fatalf("block %d: query norm %d, key norm %d, want %d each",
				i, len(b.QNorm), len(b.KNorm), bc.HeadDim)
		}
		if len(b.AttnNorm) != cfg.Dim || len(b.FFNNorm) != cfg.Dim {
			t.Fatalf("block %d: the block norms are %d and %d, not %d",
				i, len(b.AttnNorm), len(b.FFNNorm), cfg.Dim)
		}
		// The query projection is wider than the model and the key projection
		// narrower: sixteen heads against eight, of a hundred and twenty-eight.
		if b.Q.Rows != bc.Heads*bc.HeadDim {
			t.Fatalf("block %d: Q has %d rows, want %d", i, b.Q.Rows, bc.Heads*bc.HeadDim)
		}
		if b.K.Rows != bc.KVHeads*bc.HeadDim || b.V.Rows != bc.KVHeads*bc.HeadDim {
			t.Fatalf("block %d: K and V have %d and %d rows, want %d",
				i, b.K.Rows, b.V.Rows, bc.KVHeads*bc.HeadDim)
		}
		if b.O.Cols != bc.Heads*bc.HeadDim || b.O.Rows != cfg.Dim {
			t.Fatalf("block %d: the output projection is %d by %d", i, b.O.Rows, b.O.Cols)
		}
	}
}

// The head is the embedding read the other way round. If a checkpoint ever
// carries its own, this engine has to be told rather than silently ignore it.
func TestLoadWeightsRefusesAnUntiedHead(t *testing.T) {
	g := openFile(t)
	if _, ok := g.Tensors["output.weight"]; ok {
		t.Fatal("this checkpoint has an output.weight, so the engine's assumption is wrong")
	}
}

// Repack is a second copy of every matrix a product reads. It must not touch
// the embedding, which is read a row at a time by both the input and the head.
func TestRepackLeavesTheEmbeddingAlone(t *testing.T) {
	_, w := loadForTest(t)

	before := w.TokenEmbd
	w.Repack()
	if w.TokenEmbd.Rows != before.Rows || w.TokenEmbd.Cols != before.Cols {
		t.Error("Repack changed the embedding's shape")
	}
	if len(w.Blocks[0].Q.Packed) == 0 && w.Blocks[0].Q.Quant.String() == "Q4_0" {
		t.Error("Repack left a Q4_0 matrix unpacked")
	}
}
