package gemma

import "testing"

func TestLoadVisionWeights(t *testing.T) {
	g := openMMProj(t)
	cfg, err := LoadVisionConfig(g)
	if err != nil {
		t.Fatal(err)
	}
	w, err := LoadVisionWeights(g, cfg)
	if err != nil {
		t.Fatal(err)
	}
	if len(w.Blocks) != cfg.Blocks {
		t.Fatalf("bound %d blocks, expected %d", len(w.Blocks), cfg.Blocks)
	}
	if n := cfg.PatchSize * cfg.PatchSize * 3 * cfg.Dim; len(w.PatchEmbd) != n {
		t.Errorf("the patch embedding holds %d floats, expected %d", len(w.PatchEmbd), n)
	}
	if len(w.PosX) != w.Positions*cfg.Dim || len(w.PosY) != w.Positions*cfg.Dim {
		t.Errorf("the position tables hold %d and %d floats for %d positions of %d",
			len(w.PosX), len(w.PosY), w.Positions, cfg.Dim)
	}
	if w.Proj.Inputs != cfg.Dim || w.Proj.Outputs != cfg.ProjDim {
		t.Errorf("the projection is %dx%d, expected %dx%d",
			w.Proj.Outputs, w.Proj.Inputs, cfg.ProjDim, cfg.Dim)
	}
	// The QAT ranges are what make this a clipped linear rather than a linear.
	if w.ProjClamp.InMax <= w.ProjClamp.InMin || w.ProjClamp.OutMax <= w.ProjClamp.OutMin {
		t.Errorf("the projection's clamp is empty: %+v", w.ProjClamp)
	}
	b := w.Blocks[0]
	if b.Q.Inputs != cfg.Dim || b.Q.Outputs != cfg.Dim {
		t.Errorf("block 0 query is %dx%d", b.Q.Outputs, b.Q.Inputs)
	}
	if len(b.QNorm) != cfg.HeadDim || len(b.KNorm) != cfg.HeadDim {
		t.Errorf("block 0 query and key norms are %d and %d wide, expected %d",
			len(b.QNorm), len(b.KNorm), cfg.HeadDim)
	}
	if b.Gate.Outputs != cfg.FFN {
		t.Errorf("block 0 feed forward is %d wide, expected %d", b.Gate.Outputs, cfg.FFN)
	}
}
