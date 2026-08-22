package gemma

import (
	"fmt"
	"math"
	"sort"
	"testing"
)

// Every block of the tower, against the recording. A divergence names its
// block, which is the whole point of keeping all twelve.
//
// The bound is on the drift itself rather than on a fraction of each block's
// own range, and it is written as a table because the drift is worth seeing.
// This tower adds four unscaled branches per block and never normalizes the
// residual between them, so one block's bfloat16 rounding is carried into the
// next: a hundredth of a unit after block nought, a quarter after block ten,
// on a stream whose values run to three hundred. Block eleven is the odd one —
// its own ln2 divides the range by four, which would make the same drift read
// four times worse against its own scale and says nothing about the
// arithmetic.
func TestEveryConformerBlockMatchesTheReference(t *testing.T) {
	f := loadAudioFixture(t, "audio")
	tower := openAudioTower(t)
	melIn, frames := f.melInput(t)
	x, n := tower.preEncode(melIn, frames)
	s := tower.takeScratch(n)
	drift := []float32{0.05, 0.05, 0.05, 0.2, 0.2, 0.2, 0.2, 0.2, 0.3, 0.3, 0.4, 1.0}
	for i := range tower.W.Blocks {
		tower.block(&tower.W.Blocks[i], x, n, s)
		close(t, fmt.Sprintf("blk.%d.out", i), x[:n*tower.Cfg.Dim],
			f.tensor(t, fmt.Sprintf("blk.%d.out", i)), drift[i])
	}
}

// And the whole thing from the sound file, ending on the rows the language
// model is given. This is the one that says the front end and the tower agree
// about what a frame is.
//
// Two bounds, as the mel test uses, because one would have to be the loosest.
// This path carries the front end's own rounding as well as the tower's — the
// spectrogram is computed here rather than read from llama.cpp — and twelve
// blocks stand between the two. The bulk of the rows sits four thousandths
// from the reference on a range of twenty-four; a few hundred of the six
// hundred thousand reach a third of a unit. A padding or a layout wrong
// anywhere above would move the median, not the tail.
func TestEncodeMatchesTheProjectedTokens(t *testing.T) {
	f := loadAudioFixture(t, "audio")
	tower := openAudioTower(t)
	rows := tower.Encode(loadSpeech(t))
	if len(rows) != f.NAudioTokens {
		t.Fatalf("%d soft tokens, the recording has %d", len(rows), f.NAudioTokens)
	}
	want := f.tensor(t, "projected")
	got := make([]float32, 0, len(want))
	for _, r := range rows {
		got = append(got, r...)
	}
	if len(got) != len(want) {
		t.Fatalf("%d values against the recording's %d", len(got), len(want))
	}
	var scale float32
	gaps := make([]float64, len(want))
	for i := range want {
		gaps[i] = math.Abs(float64(got[i] - want[i]))
		if v := abs32(want[i]); v > scale {
			scale = v
		}
	}
	sort.Float64s(gaps)
	if p99, allowed := gaps[len(gaps)*99/100], float64(scale)*1e-3; p99 > allowed {
		t.Fatalf("a hundredth of the rows are more than %.5f from the reference, over the %.5f allowed on a range of %.2f",
			p99, allowed, scale)
	}
	if worst, allowed := gaps[len(gaps)-1], float64(scale)*2e-2; worst > allowed {
		t.Fatalf("one row is %.5f from the reference, over the %.5f allowed on a range of %.2f", worst, allowed, scale)
	}
}

// A recording longer than a chunk becomes the chunks' tokens laid end to end,
// and nothing about the count depends on where the cut fell.
func TestALongRecordingIsCutIntoChunks(t *testing.T) {
	tower := openAudioTower(t)
	rows := tower.Encode(make([]float32, 45*16000))
	if len(rows) < 700 {
		t.Fatalf("forty-five seconds became %d tokens, which is fewer than one chunk's worth", len(rows))
	}
	for i, r := range rows {
		if len(r) != tower.Cfg.ProjDim {
			t.Fatalf("row %d is %d wide, want %d", i, len(r), tower.Cfg.ProjDim)
		}
	}
}
