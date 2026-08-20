package gemma

import "testing"

func TestEmbedMatchesTheReference(t *testing.T) {
	f := loadFixture(t, "layers")
	g := openModel(t)
	cfg, _ := LoadConfig(g, 4096)
	w, err := LoadWeights(g, cfg)
	if err != nil {
		t.Fatal(err)
	}

	out := make([]float32, cfg.Dim)
	for pos, token := range f.Tokens {
		Embed(cfg, w, token, out)
		compare(t, "inp_scaled at position "+itoa(pos),
			out, f.column(t, "inp_scaled", pos), 1e-4)
	}
}

// The recordings of the per-layer path are three-dimensional, and the two of
// them are not in the same order: llama.cpp permutes between them.
//
//	per_layer_proj  is [256, n_layer, n_tokens]
//	inp_per_layer   is [256, n_tokens, n_layer]
func plVector(t *testing.T, f *fixture, name string, token, block int) []float32 {
	t.Helper()
	e := f.Tensors[name]
	all := f.tensor(t, name)
	width := int(e.NE[0])

	var index int
	switch {
	case int(e.NE[1]) == f.NLayer && int(e.NE[2]) == len(f.Tokens):
		index = token*f.NLayer + block
	case int(e.NE[1]) == len(f.Tokens) && int(e.NE[2]) == f.NLayer:
		index = block*len(f.Tokens) + token
	default:
		t.Fatalf("%s has shape %v, which is neither layout", name, e.NE)
	}
	return all[index*width : (index+1)*width]
}

func TestPerLayerInputsMatchTheReference(t *testing.T) {
	f := loadFixture(t, "layers")
	g := openModel(t)
	cfg, _ := LoadConfig(g, 4096)
	w, err := LoadWeights(g, cfg)
	if err != nil {
		t.Fatal(err)
	}
	s := NewScratch(cfg)

	embedded := s.Batch(cfg.Dim, 1)
	ple := [][]float32{make([]float32, len(cfg.Blocks)*cfg.PLEDim)}

	for pos, token := range f.Tokens {
		// Driven by the reference's own embedding, so that a fault in Embed
		// cannot be mistaken for a fault here.
		embedded.Set(0, f.column(t, "inp_scaled", pos))
		PerLayerInputs(cfg, w, s, embedded, []int32{token}, ple)

		for _, block := range []int{0, 1, 17, 34} {
			got := ple[0][block*cfg.PLEDim : (block+1)*cfg.PLEDim]
			compare(t, "inp_per_layer block "+itoa(block)+" at position "+itoa(pos),
				got, plVector(t, f, "inp_per_layer", pos, block), 2e-4)
		}
	}
}

// A model without per-layer embeddings must not be given any. The 12B is that
// model, and the engine has to run it unchanged.
func TestPerLayerInputsAbsent(t *testing.T) {
	cfg := &Config{Dim: 1536, PLEDim: 0, Blocks: make([]BlockConfig, 4)}
	out := [][]float32{{1, 2, 3}}
	PerLayerInputs(cfg, &Weights{}, NewScratch(cfg), nil, []int32{7}, out)
	if out[0][0] != 1 || out[0][1] != 2 || out[0][2] != 3 {
		t.Fatalf("something was written where there is nothing to write: %v", out)
	}
}
