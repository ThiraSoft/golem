package yue2

import (
	"cmp"
	"fmt"
	"math"
	"slices"
)

// Phase designates the autoregressive stage currently sampling from the LM head.
type Phase int

const (
	PhaseABC Phase = iota
	PhaseSemantic
)

// Sampling parameters controlling the autoregressive token generation.
type Sampling struct {
	Temperature       float32
	TopP              float32
	TopK              int
	RepetitionPenalty float32
	PenaltyWindow     int
	MinTokens         int
	MaxTokens         int
}

// Checkpoint native sampling defaults for each phase.
var (
	ABCSampling      = Sampling{Temperature: 0.7, TopP: 0.9, TopK: 30, RepetitionPenalty: 1.005, PenaltyWindow: 100, MinTokens: 32, MaxTokens: 4096}
	SemanticSampling = Sampling{Temperature: 1.0, TopP: 0.95, TopK: 100, RepetitionPenalty: 1.2, PenaltyWindow: 50, MinTokens: 200, MaxTokens: 9000}
)

// Valid checks that the sampling configuration stays within the bounds enforced by the protocol.
func (s Sampling) Valid() error {
	if s.Temperature < 0.0 || s.Temperature > 5.0 {
		return fmt.Errorf("temperature %g outside [0, 5]", s.Temperature)
	}
	if s.TopP <= 0.0 || s.TopP > 1.0 {
		return fmt.Errorf("top_p %g outside (0, 1]", s.TopP)
	}
	if s.TopK < 1 {
		return fmt.Errorf("top_k %d < 1", s.TopK)
	}
	if s.RepetitionPenalty <= 0.0 {
		return fmt.Errorf("repetition_penalty %g <= 0", s.RepetitionPenalty)
	}
	if s.PenaltyWindow < 1 || s.PenaltyWindow > 100 {
		return fmt.Errorf("penalty_window %d outside [1, 100]", s.PenaltyWindow)
	}
	if s.MinTokens < 0 || s.MinTokens > s.MaxTokens || s.MaxTokens < 1 {
		return fmt.Errorf("invalid token bounds: min %d, max %d", s.MinTokens, s.MaxTokens)
	}
	return nil
}

// phaseRange returns the content token range [lo, hi) and end token for a phase.
func phaseRange(p Phase) (lo, hi, end int) {
	if p == PhaseABC {
		return 0, EOD, ABCEnd
	}
	return CodecOffset, CodecOffset + CodecSize, MusicEnd
}

// PhaseRows returns the first LM head row index (row0) and row count for the phase.
func PhaseRows(p Phase) (row0, rows int) {
	lo, hi, end := phaseRange(p)
	row0 = lo
	if end < row0 {
		row0 = end
	}
	maxRow := hi
	if end >= maxRow {
		maxRow = end + 1
	}
	return row0, maxRow - row0
}

// Candidate holds a token identifier and its unnormalized logit score.
type Candidate struct {
	ID    int32
	Score float32
}

// Distribution applies phase masking, the end token floor, repetition penalties over
// the sliding window, temperature scaling, top-k truncation with ties, and nucleus
// (top-p) filtering to logits. The surviving candidates are returned in descending
// score order.
func Distribution(logits []float32, s Sampling, history []int32, step int, p Phase, out []Candidate) []Candidate {
	lo, hi, end := phaseRange(p)
	row0, _ := PhaseRows(p)

	out = out[:0]
	needed := (hi - lo) + 1
	if cap(out) < needed {
		out = make([]Candidate, 0, needed)
	}

	for id := lo; id < hi; id++ {
		out = append(out, Candidate{ID: int32(id), Score: logits[id-row0]})
	}
	if step >= s.MinTokens {
		out = append(out, Candidate{ID: int32(end), Score: logits[end-row0]})
	}

	// Repetition penalty over the sliding window. Scores are penalized at their
	// first occurrence in the window; negative logits multiply and positive divide.
	if s.RepetitionPenalty != 1.0 {
		first := 0
		if len(history) > s.PenaltyWindow {
			first = len(history) - s.PenaltyWindow
		}
		for i := first; i < len(history); i++ {
			id := history[i]
			seen := false
			for j := first; j < i; j++ {
				if history[j] == id {
					seen = true
					break
				}
			}
			if seen {
				continue
			}
			freq := 0
			for j := i; j < len(history); j++ {
				if history[j] == id {
					freq++
				}
			}
			alpha := float32(math.Pow(float64(s.RepetitionPenalty), float64(freq)))
			idx := int(id - int32(lo))
			if idx >= 0 && idx < len(out) {
				score := out[idx].Score
				if score < 0 {
					score *= alpha
				} else {
					score /= alpha
				}
				out[idx].Score = score
			}
		}
	}

	if s.Temperature != 0.0 && s.Temperature != 1.0 {
		for i := range out {
			out[i].Score /= s.Temperature
		}
	}

	if s.Temperature == 0.0 {
		best := 0
		for i := 1; i < len(out); i++ {
			if out[i].Score > out[best].Score {
				best = i
			}
		}
		winner := out[best]
		return append(out[:0], winner)
	}

	// Top-k truncation keeps all candidates at or above the k-th score, ties
	// included. Only the k-th score is needed to cut, so it is selected with a
	// heap of k scores rather than by sorting the whole window (151849 entries
	// in the score phase, 19 ms a step when sorted). The survivors are then a
	// few dozen and sorted alone, as yue2.cpp sorts its kept set.
	k := s.TopK
	if k > len(out) {
		k = len(out)
	}
	threshold := kthLargest(out, k)
	kept := 0
	for i := range out {
		if out[i].Score >= threshold {
			out[kept] = out[i]
			kept++
		}
	}
	out = out[:kept]
	// Equal scores go by id so the order, and with it the nucleus cut and the
	// draw, never depends on how the sort happens to break ties.
	slices.SortFunc(out, func(a, b Candidate) int {
		if a.Score != b.Score {
			if a.Score > b.Score {
				return -1
			}
			return 1
		}
		return cmp.Compare(a.ID, b.ID)
	})

	if s.TopP >= 1.0 {
		return out
	}

	// Nucleus (top-p) filtering over the surviving distribution. The highest-scoring
	// candidate is always preserved.
	top := out[0].Score
	var sum float64
	for i := range out {
		sum += math.Exp(float64(out[i].Score - top))
	}

	var cumulative float64
	for i := range out {
		if i > 0 && cumulative > float64(s.TopP) {
			out = out[:i]
			break
		}
		cumulative += math.Exp(float64(out[i].Score-top)) / sum
	}

	return out
}

// kthLargest returns the k-th largest score of c, counting equal scores
// separately, as out[k-1] of a descending sort would. It keeps the k largest
// scores seen in a min-heap, so most candidates cost one comparison with its
// root.
func kthLargest(c []Candidate, k int) float32 {
	var small [128]float32
	h := small[:0]
	if k > len(small) {
		h = make([]float32, 0, k)
	}
	for i := range c {
		x := c[i].Score
		if len(h) < k {
			h = append(h, x)
			for j := len(h) - 1; j > 0; {
				p := (j - 1) / 2
				if h[p] <= h[j] {
					break
				}
				h[p], h[j] = h[j], h[p]
				j = p
			}
			continue
		}
		if x <= h[0] {
			continue
		}
		h[0] = x
		for j := 0; ; {
			l, m := 2*j+1, j
			if l < k && h[l] < h[m] {
				m = l
			}
			if l+1 < k && h[l+1] < h[m] {
				m = l + 1
			}
			if m == j {
				break
			}
			h[j], h[m] = h[m], h[j]
			j = m
		}
	}
	return h[0]
}

const curand2Pow32Inv = float32(2.3283064365386963e-10)

func curandUniform(x uint32) float32 {
	return float32(x)*curand2Pow32Inv + (curand2Pow32Inv * 0.5)
}

// Draw samples one token from the candidate distribution using the Gumbel-max trick
// driven by the request lm_seed and Philox4x32-10 PRNG.
func Draw(candidates []Candidate, seed int64, index int64) int32 {
	if len(candidates) == 0 {
		return 0
	}
	top := candidates[0].Score
	var sum float64
	for i := range candidates {
		sum += math.Exp(float64(candidates[i].Score - top))
	}

	seedLo := uint32(seed)
	seedHi := uint32(uint64(seed) >> 32)
	best := candidates[0].ID
	ratio := float32(-1.0)

	for i := range candidates {
		ctr := [4]uint32{
			uint32(index),
			uint32(uint64(index) >> 32),
			uint32(candidates[i].ID),
			0,
		}
		r := Philox4x32_10(ctr, seedLo, seedHi)
		p := float32(math.Exp(float64(candidates[i].Score-top)) / sum)
		u := curandUniform(r[0])
		e := -float32(math.Log(float64(u)))
		c := p / e
		if c > ratio {
			ratio = c
			best = candidates[i].ID
		}
	}
	return best
}
