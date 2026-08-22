package gemma

import (
	"math"
	"testing"
)

// The depthwise convolution is causal: frame t reads t-4 through t, and a
// signal that changes only after t cannot change what came before it. That is
// a property worth a test of its own, because a symmetric padding here would
// still run and would still look plausible.
func TestDepthwiseConvolutionIsCausal(t *testing.T) {
	tower := openAudioTower(t)
	b := &tower.W.Blocks[0]
	dim := tower.Cfg.Dim
	n := 40
	x := make([]float32, n*dim)
	for i := range x {
		x[i] = float32(math.Sin(float64(i)))
	}
	first := make([]float32, len(x))
	tower.convModule(b, x, first, n, tower.takeScratch(n))

	// Change the second half; the first half must not move.
	for i := 20 * dim; i < len(x); i++ {
		x[i] += 1
	}
	second := make([]float32, len(x))
	tower.convModule(b, x, second, n, tower.takeScratch(n))
	for i := 0; i < 20*dim; i++ {
		if first[i] != second[i] {
			t.Fatalf("frame %d moved when a later frame changed; the convolution is not causal", i/dim)
		}
	}
	// And the second half did move, or the test above proves nothing.
	same := true
	for i := 20 * dim; i < len(x); i++ {
		if first[i] != second[i] {
			same = false
			break
		}
	}
	if same {
		t.Fatal("nothing moved at all; the module is not reading its input")
	}
}

func TestConvModuleMatchesTheReference(t *testing.T) {
	f := loadAudioFixture(t, "audio")
	tower := openAudioTower(t)
	in := f.tensor(t, "blk.0.conv_in")
	n := f.frames(t, "blk.0.conv_in")
	got := make([]float32, len(in))
	tower.convModule(&tower.W.Blocks[0], in, got, n, tower.takeScratch(n))
	closeRelative(t, "blk.0.conv_out", got, f.tensor(t, "blk.0.conv_out"), 5e-4)
}
