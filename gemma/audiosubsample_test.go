package gemma

import "testing"

// The two strided convolutions and the projection that follows them, against
// the nodes the recorder kept. A divergence here is a padding or a layout, and
// it is cheaper to find at this end than after twelve blocks.
func TestSubsamplingMatchesTheReference(t *testing.T) {
	f := loadAudioFixture(t, "audio")
	tower := openAudioTower(t)
	melIn, frames := f.melInput(t)
	n := audioPositions(frames)
	s := tower.takeScratchFor(frames, n)
	defer tower.putScratch(s)
	got := tower.preEncode(melIn, frames, s)
	if want := f.frames(t, "pre_encode"); n != want {
		t.Fatalf("the subsampling gave %d positions, the reference %d", n, want)
	}
	compare(t, "pre_encode", got[:n*tower.Cfg.Dim], f.tensor(t, "pre_encode"), 2e-3)
}

// Two halvings of the time axis, each rounding up, and the number of positions
// the mel becomes is what the whole prompt's length depends on.
func TestSubsamplingQuartersTheTimeAxis(t *testing.T) {
	for _, c := range []struct{ frames, want int }{
		{1743, 436}, {1, 1}, {4, 1}, {5, 2}, {8, 2}, {9, 3},
	} {
		if got := audioPositions(c.frames); got != c.want {
			t.Errorf("%d mel frames become %d positions, want %d", c.frames, got, c.want)
		}
	}
}
