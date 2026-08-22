package gemma

import (
	"math"
	"testing"
)

// Every tensor the tower reads is in the file, with the shape the geometry
// implies. A missing one is a sentence, not a nil pointer at the first block.
func TestAudioWeightsAreAllThere(t *testing.T) {
	g := openMMProj(t)
	cfg, err := LoadAudioConfig(g)
	if err != nil {
		t.Fatal(err)
	}
	w, err := LoadAudioWeights(g, cfg)
	if err != nil {
		t.Fatal(err)
	}
	if len(w.Blocks) != cfg.Blocks {
		t.Fatalf("%d blocks bound, file declares %d", len(w.Blocks), cfg.Blocks)
	}
	b := w.Blocks[0]
	if b.Q.Inputs != cfg.Dim || b.Q.Outputs != cfg.Dim {
		t.Errorf("the query is %dx%d, want %d square", b.Q.Outputs, b.Q.Inputs, cfg.Dim)
	}
	if b.FFNUp.Outputs != cfg.FFN {
		t.Errorf("the feed forward rises to %d, want %d", b.FFNUp.Outputs, cfg.FFN)
	}
	if len(b.ConvDW) != 5*cfg.Dim {
		t.Errorf("the depthwise convolution has %d weights, want 5 per channel", len(b.ConvDW))
	}
	if len(b.PerDimScale) != cfg.HeadDim {
		t.Errorf("the per-dimension scale has %d entries for a head of %d", len(b.PerDimScale), cfg.HeadDim)
	}
	if b.Q.Clamp == (Clamp{}) {
		t.Error("the query carries no clamp; the file has four scalars next to it")
	}
	if w.MMProj.Outputs != cfg.ProjDim {
		t.Errorf("the projection produces %d-wide tokens, want %d", w.MMProj.Outputs, cfg.ProjDim)
	}
	if len(w.OutProjBias) != w.OutProj.Outputs {
		t.Errorf("the output projection's bias has %d entries for %d outputs", len(w.OutProjBias), w.OutProj.Outputs)
	}
	// The two subsampling convolutions: 3x3, one channel in, then that many.
	if w.Conv[0].KW != 3 || w.Conv[0].KH != 3 || w.Conv[0].In != 1 {
		t.Errorf("the first convolution is %dx%d over %d channels, want 3x3 over 1", w.Conv[0].KW, w.Conv[0].KH, w.Conv[0].In)
	}
	if w.Conv[1].In != w.Conv[0].Out {
		t.Errorf("the second convolution takes %d channels and the first gives %d", w.Conv[1].In, w.Conv[0].Out)
	}
	// After two halvings of the frequency axis, the channels of the second
	// convolution times what is left is what the input projection reads.
	if want := w.Conv[1].Out * (cfg.MelBins / 4); w.InputProj.Inputs != want {
		t.Errorf("the input projection reads %d values, want %d", w.InputProj.Inputs, want)
	}
	if w.InputProj.Outputs != cfg.Dim {
		t.Errorf("the input projection gives %d, want the tower's %d", w.InputProj.Outputs, cfg.Dim)
	}
}

// The clamps are read, not invented: a weight the file gives no range for gets
// an infinite one, and every projection in this tower does have one.
func TestEveryAudioProjectionCarriesItsRange(t *testing.T) {
	g := openMMProj(t)
	cfg, err := LoadAudioConfig(g)
	if err != nil {
		t.Fatal(err)
	}
	w, err := LoadAudioWeights(g, cfg)
	if err != nil {
		t.Fatal(err)
	}
	for i, b := range w.Blocks {
		for name, l := range map[string]VisionLinear{
			"attn_q": b.Q, "attn_k": b.K, "attn_v": b.V, "attn_out": b.Out,
			"ffn_up": b.FFNUp, "ffn_down": b.FFNDown,
			"ffn_up_1": b.FFNUp1, "ffn_down_1": b.FFNDown1,
			"conv_pw1": b.ConvPW1, "conv_pw2": b.ConvPW2,
		} {
			if math.IsInf(float64(l.Clamp.InMin), -1) {
				t.Errorf("block %d: %s has no input range", i, name)
			}
		}
	}
}
