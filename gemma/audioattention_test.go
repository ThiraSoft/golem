package gemma

import (
	"math"
	"testing"
)

// The blocked geometry, stated twice: once here as arithmetic, once against
// llama.cpp below. With a chunk of twelve and a past horizon of twelve, query
// t of chunk j is offered keys [12j-12, 12j+11] and keeps those that are in
// the signal, not in its future, and less than twelve frames behind it. The
// window is twenty-four wide and no query uses more than twelve of it — the
// asymmetry is llama.cpp's, and it is the bug every other mistake in this file
// looks like.
func TestBlockedMaskSeesTwelveBack(t *testing.T) {
	const n, chunk, past = 29, 12, 12
	for b := 0; b < 3; b++ {
		for q := 0; q < chunk; q++ {
			pos := b*chunk + q
			seen := 0
			for k := 0; k < chunk+past; k++ {
				key := b*chunk - past + k
				want := key >= 0 && key < n && key <= pos && pos-key < past && pos < n
				if got := audioVisible(pos, key, n, past); got != want {
					t.Fatalf("position %d, key %d: visible=%v, want %v", pos, key, got, want)
				}
				if want {
					seen++
				}
			}
			if pos < n && seen > past {
				t.Fatalf("position %d sees %d keys, more than the horizon of %d", pos, seen, past)
			}
		}
	}
}

// The relative table's shift, worked out rather than transcribed. llama.cpp
// pads each of the thirteen-wide rows to twenty-five, reads the flattened
// result back twenty-four at a time and calls the result the scores' relative
// term. This says the entry that lands on a key the query can see is always
// that query's own row at the distance between them — which is the one line
// audioattention.go writes instead of the pad and the reshape.
func TestTheRelativeShiftLandsOnTheDistance(t *testing.T) {
	const chunk, past, context, rpe = 12, 12, 24, 13
	for qi := 0; qi < chunk; qi++ {
		for ki := 0; ki < context; ki++ {
			gq, gk := qi, ki-past // one chunk is enough: the shift is per chunk
			if !(gk <= gq && gq-gk < past) {
				continue
			}
			at := qi*context + ki
			row, column := at/(context+1), at%(context+1)
			if row != qi {
				t.Fatalf("query %d, key %d: the shift reads row %d", qi, ki, row)
			}
			if column >= rpe {
				t.Fatalf("query %d, key %d: the shift reads column %d, past the thirteen distances", qi, ki, column)
			}
			if want := past - (gq - gk); column != want {
				t.Fatalf("query %d, key %d: the shift reads distance row %d, want %d", qi, ki, column, want)
			}
		}
	}
}

// And the whole branch against the recording: what llama.cpp handed block 0's
// attention, and what it gave back before the residual.
func TestAttentionMatchesTheReference(t *testing.T) {
	f := loadAudioFixture(t, "audio")
	tower := openAudioTower(t)
	in := f.tensor(t, "blk.0.attn_in")
	n := f.frames(t, "blk.0.attn_in")
	got := make([]float32, len(in))
	tower.attention(&tower.W.Blocks[0], in, got, n, tower.takeScratch(n))
	// A tolerance scaled to what the tensor holds rather than an absolute one:
	// this branch's output runs to a hundred and thirty, and a thousandth of
	// that is not the same demand as a thousandth of a residual near one.
	closeRelative(t, "blk.0.attn_out", got, f.tensor(t, "blk.0.attn_out"), 1e-4)
}

// A query in the padded tail of the last chunk writes nothing anyone reads,
// and a signal whose length is not a multiple of twelve is the ordinary case.
func TestAttentionAcceptsAPartialLastChunk(t *testing.T) {
	f := loadAudioFixture(t, "audio")
	tower := openAudioTower(t)
	all := f.tensor(t, "blk.0.attn_in")
	dim := tower.Cfg.Dim
	n := 25 // two full chunks and one frame
	in := all[:n*dim]
	got := make([]float32, len(in))
	tower.attention(&tower.W.Blocks[0], in, got, n, tower.takeScratch(n))
	for i, v := range got {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatalf("element %d of a %d-frame signal is %v", i, n, v)
		}
	}
}
