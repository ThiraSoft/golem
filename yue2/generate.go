package yue2

import (
	"errors"
	"fmt"
)

// Generation is what one autoregressive stage wrote: the content tokens, the
// end token excluded, and whether the budget ran out before the end token.
type Generation struct {
	Tokens    []int32
	Truncated bool
}

// ErrCancelled is what a stage returns, wrapped or as it is, when its cancel
// function says so: Generate between tokens, NAR.Solve between steps, the
// VAE between tiles and Pipeline.Generate between stages.
var ErrCancelled = errors.New("yue2: cancelled")

// Generate is yue2.cpp's yue2_generate (src/generate.h) for one song: prefill
// prefix into slot 0 (and negative into slot 1 when guidance is not one),
// then draw and decode a token a step until the end token or the budget.
//
// What it keeps from the reference, each on purpose there:
//
//   - The Philox key is seed for every stage, and the draw index is the step.
//   - With guidance, both slots decode the same token in one pass, and the
//     logits mix as uncond + guidance·(cond - uncond) before the distribution.
//   - The end token, once drawn, is fed once more, so the cache holds it; a
//     budget cut on a content token owes two forwards, the token and then the
//     end token. A slot's cache always ends on the end token.
//
// forced, when not nil, is the teacher-forced mode: the distribution is still
// computed at every step, but forced[step] is the token taken, and the end
// token after the last of them. cancel, when not nil, is asked before every
// step.
func Generate(ar *AR, prefix, negative []int32, guidance float32, s Sampling, seed int64, p Phase,
	forced []int32, cancel func() bool) (Generation, error) {
	if err := s.Valid(); err != nil {
		return Generation{}, fmt.Errorf("yue2: %w", err)
	}
	guided := guidance != 1
	if guided && len(negative) == 0 {
		return Generation{}, fmt.Errorf("yue2: guidance %g needs an unconditional prefix", guidance)
	}
	if guided && len(ar.pos) < 2 {
		return Generation{}, fmt.Errorf("yue2: guidance needs two slots, the AR has %d", len(ar.pos))
	}
	longest := len(prefix)
	if guided {
		longest = max(longest, len(negative))
	}
	if longest+s.MaxTokens+1 > ar.context {
		return Generation{}, fmt.Errorf("yue2: prefix %d + budget %d + end exceeds context %d", longest, s.MaxTokens, ar.context)
	}
	_, _, end := phaseRange(p)
	_, rows := PhaseRows(p)

	ar.Trim(0, 0)
	cond, err := ar.Prefill(0, prefix, p)
	if err != nil {
		return Generation{}, err
	}
	var uncond, mixed []float32
	if guided {
		ar.Trim(1, 0)
		if uncond, err = ar.Prefill(1, negative, p); err != nil {
			return Generation{}, err
		}
		mixed = make([]float32, rows)
	}

	g := Generation{Truncated: true}
	// The forwards still owed once drawing stops: one for the end token, two
	// when the budget cut on a content token. Negative while drawing.
	owed := -1
	tokens := []int32{0, 0}
	slots := []int{0, 1}
	var candidates []Candidate
	for step := 0; ; step++ {
		if cancel != nil && cancel() {
			return g, ErrCancelled
		}
		var token int32
		if owed < 0 {
			logits := cond
			if guided {
				for k := range mixed {
					mixed[k] = uncond[k] + guidance*(cond[k]-uncond[k])
				}
				logits = mixed
			}
			candidates = Distribution(logits, s, g.Tokens, step, p, candidates)
			switch {
			case forced == nil:
				token = Draw(candidates, seed, int64(step))
			case step < len(forced):
				token = forced[step]
			default:
				token = int32(end)
			}
			if token == int32(end) {
				g.Truncated = false
				owed = 1
			} else {
				g.Tokens = append(g.Tokens, token)
				if len(g.Tokens) >= s.MaxTokens {
					owed = 2
				}
			}
		} else {
			token = int32(end)
		}
		if owed == 0 {
			break
		}
		n := 1
		if guided {
			n = 2
		}
		tokens[0], tokens[1] = token, token
		outs, err := ar.Step(tokens[:n], slots[:n], p)
		if err != nil {
			return g, err
		}
		cond = outs[0]
		if guided {
			uncond = outs[1]
		}
		if owed > 0 {
			owed--
		}
	}
	return g, nil
}
