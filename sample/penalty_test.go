package sample

import (
	"math"
	"math/rand/v2"
	"testing"
)

// The arithmetic is llama.cpp's, and it is not symmetric: a logit above zero is
// divided by the penalty, one at or below it is multiplied. Dividing a negative
// logit would make an unlikely token likelier, which is the fault that fix is
// there for.
func TestRepeatPenaltyLowersASeenTokenOnEitherSideOfZero(t *testing.T) {
	s := New(Params{Temperature: 1, PenaltyLastN: 64, PenaltyRepeat: 2})
	s.Seed([]int32{0, 1})

	row := []float32{4, -4, 0}
	got := s.penalised(row)
	want := []float32{2, -8, 0}
	for i := range want {
		if math.Abs(float64(got[i]-want[i])) > 1e-6 {
			t.Fatalf("penalised row is %v, expected %v", got, want)
		}
	}
}

// Frequency counts every appearance; presence counts the first one only.
func TestFrequencyAndPresenceSubtract(t *testing.T) {
	s := New(Params{Temperature: 1, PenaltyLastN: 64, PenaltyRepeat: 1,
		PenaltyFreq: 0.5, PenaltyPresent: 2})
	s.Seed([]int32{0, 0, 0, 1})

	got := s.penalised([]float32{10, 10, 10})
	want := []float32{6.5, 7.5, 10}
	for i := range want {
		if math.Abs(float64(got[i]-want[i])) > 1e-6 {
			t.Fatalf("penalised row is %v, expected %v", got, want)
		}
	}
}

// The window is the last PenaltyLastN identifiers and nothing older.
func TestTheWindowForgetsWhatLeavesIt(t *testing.T) {
	s := New(Params{Temperature: 1, PenaltyLastN: 2, PenaltyRepeat: 2})
	s.Seed([]int32{0, 1, 2})

	got := s.penalised([]float32{4, 4, 4})
	want := []float32{4, 2, 2}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("penalised row is %v, expected %v: the oldest identifier is still counted", got, want)
		}
	}
}

// A window of zero is the feature turned off, whatever the three weights say.
func TestALastNOfZeroPenalisesNothing(t *testing.T) {
	s := New(Params{Temperature: 1, PenaltyLastN: 0, PenaltyRepeat: 4, PenaltyPresent: 9})
	s.Seed([]int32{0, 1, 2})
	got := s.penalised([]float32{4, 4, 4})
	for i, v := range got {
		if v != 4 {
			t.Fatalf("identifier %d was penalised to %v with no window", i, v)
		}
	}
}

// The point of the whole thing: the chain never walks the row. It takes the
// k+u best raw candidates, penalises those, and lands on the identifiers a full
// pass over the row would have named. This is the proof.
func TestTheShortcutDrawsWhatAFullPassDraws(t *testing.T) {
	const vocab = 4096
	rng := rand.New(rand.NewPCG(11, 22))

	for _, p := range []Params{
		{Temperature: 1, TopK: 64, TopP: 0.95, PenaltyLastN: 64, PenaltyRepeat: 1.3},
		{Temperature: 0.7, TopK: 8, TopP: 1, PenaltyLastN: 64, PenaltyRepeat: 1, PenaltyFreq: 0.8},
		{Temperature: 1, TopK: 40, TopP: 0.9, PenaltyLastN: 32, PenaltyRepeat: 2, PenaltyFreq: 0.1, PenaltyPresent: 0.4},
		{Temperature: 0, TopK: 40, TopP: 0.9, PenaltyLastN: 64, PenaltyRepeat: 3},
		// A penalty below one raises a logit rather than lowering it, which the
		// shortcut cannot see coming. The chain is expected to notice and take
		// the slow road; the identifiers must be the same either way.
		{Temperature: 1, TopK: 64, TopP: 0.95, PenaltyLastN: 64, PenaltyRepeat: 0.5},
		{Temperature: 1, TopK: 64, TopP: 0.95, PenaltyLastN: 64, PenaltyRepeat: 1, PenaltyPresent: -1},
	} {
		p.Seed = 7
		fast, slow := New(p), New(p)

		// The same window on both, drawn from the tokens the row favours so
		// that the penalties actually bite.
		var seed []int32
		for i := 0; i < 200; i++ {
			seed = append(seed, int32(rng.IntN(128)))
		}
		fast.Seed(seed)
		slow.Seed(seed)

		for draw := 0; draw < 200; draw++ {
			row := make([]float32, vocab)
			for i := range row {
				row[i] = float32(rng.NormFloat64() * 3)
			}
			want := slow.pickOverFullRow(row)
			got := fast.Pick(row)
			if got != want {
				t.Fatalf("params %+v, draw %d: the shortcut named %d and a full pass named %d", p, draw, got, want)
			}
		}
	}
}

// pickOverFullRow is the reference: penalise a copy of the whole row, then run
// the ordinary chain over it. It is what the shortcut has to agree with.
func (s *Sampler) pickOverFullRow(logits []float32) int32 {
	row := append([]float32(nil), logits...)
	for i := range row {
		row[i] = s.pen.apply(int32(i), row[i])
	}
	id := s.plainDraw(row)
	s.Accept(id)
	return id
}

// The prompt is part of the window: llama-server feeds every prompt token to
// the sampler before the first draw.
func TestSeedFillsTheWindowFromThePrompt(t *testing.T) {
	s := New(Params{Temperature: 0, TopK: 8, PenaltyLastN: 64, PenaltyRepeat: 8})
	row := []float32{3, 2, 1}
	if got := s.Pick(row); got != 0 {
		t.Fatalf("an unseeded sampler drew %d, expected the highest logit", got)
	}

	s = New(Params{Temperature: 0, TopK: 8, PenaltyLastN: 64, PenaltyRepeat: 8})
	s.Seed([]int32{0})
	if got := s.Pick(row); got != 1 {
		t.Fatalf("a prompt holding identifier 0 still drew %d", got)
	}
}

// Whatever Pick returns joins the window, so a caller that hands Pick to a
// speculative step needs no second call.
func TestADrawnTokenJoinsTheWindow(t *testing.T) {
	s := New(Params{Temperature: 0, TopK: 8, PenaltyLastN: 64, PenaltyRepeat: 8})
	row := []float32{3, 2.9, 2.8}
	first := s.Pick(row)
	second := s.Pick(row)
	if first == second {
		t.Fatalf("both draws named %d: the first was not penalised", first)
	}
}
