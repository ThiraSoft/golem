package main

// Weighing options rather than drawing an answer.
//
// A typed question asks which of a closed list of answers the model would
// give, and how sure it is. Nothing is drawn: the prompt is read, and each
// option is scored as the model would have scored writing it, the sum of the
// log-probabilities of its tokens, each read off the state before it.
//
// The state is read once for every question about it. A question's own words
// come after it, and so do its options, and all of them are written at the
// same positions in turn: the cache is rewound to where the state ends, the
// way a conversation is rewound to a prefix it shares. Nothing is copied, so a
// question costs no memory beyond what the slot already had, and the price is
// that the options of a question go through one after the other rather than
// side by side. An option of one token costs nothing past its question: the
// state its question's words end on scores the first token of every option.
// An option of several costs a pass for its tokens past the first, less what
// it shares with the option before it.

import (
	"context"
	"fmt"
	"math"
	"runtime"
	"slices"
	"sync"

	"github.com/ThiraSoft/golem/engine"
)

// weighing is one question as tokens: what follows the state, and its options.
type weighing struct {
	tail    []int32
	options [][]int32
}

// Weigh scores every option of every question about one state, and returns
// their log-probabilities in the order they came, and how many positions were
// fed. build turns a run of tokens that starts with the state into a prompt,
// which is where a picture in the state goes back in.
func (c *Context) Weigh(ctx context.Context, build func([]int32) (engine.Prompt, error),
	state []int32, qs []weighing, vocab int) ([][]float64, int, error) {
	if c.recurrent {
		return nil, 0, fmt.Errorf("serve: this model's cache is a recurrent state, which cannot be rewound to weigh one option after another")
	}
	longest := 0
	for _, q := range qs {
		if len(q.tail) == 0 || len(q.options) == 0 {
			return nil, 0, fmt.Errorf("serve: a question with no words or no options")
		}
		for _, o := range q.options {
			if len(o) == 0 {
				return nil, 0, fmt.Errorf("serve: an option with no tokens")
			}
			longest = max(longest, len(o))
		}
	}
	rows := make([][]float32, longest)
	for i := range rows {
		rows[i] = make([]float32, vocab)
	}

	fed := 0
	out := make([][]float64, len(qs))
	for qi, q := range qs {
		// Options that share a beginning go one after the other, so that the
		// second is fed only from where it parts from the first.
		order := make([]int, len(q.options))
		for i := range order {
			order[i] = i
		}
		slices.SortStableFunc(order, func(a, b int) int { return slices.Compare(q.options[a], q.options[b]) })

		lp := make([]float64, len(q.options))
		for n, i := range order {
			opt := q.options[i]
			// The first option rides with its question's words and scores the
			// first token of every option; the others score only what follows
			// their first token, and an option of one token needs no pass.
			from := 0
			if n > 0 {
				from = 1
				if len(opt) == 1 {
					continue
				}
			}
			ids := slices.Concat(state, q.tail, opt[:len(opt)-1])
			p, err := build(ids)
			if err != nil {
				return nil, fed, err
			}
			scored := rows[:len(opt)-from]
			k, err := c.PrefillScored(ctx, p, scored)
			fed += k
			if err != nil {
				return nil, fed, err
			}
			if n == 0 {
				z := logSum(scored[0])
				for j, o := range q.options {
					lp[j] = float64(scored[0][o[0]]) - z
				}
			}
			for t := 1; t < len(opt); t++ {
				lp[i] += float64(scored[t-from][opt[t]]) - logSum(scored[t-from])
			}
		}
		out[qi] = lp
	}
	return out, fed, nil
}

// logSum is the log of the sum of a row's exponentials, which a logit takes
// away to become a log-probability. It is summed in float64: a row is a
// quarter of a million entries, and a float32 sum loses the small ones.
//
// A quarter of a million exponentials is 3.4 ms on one core of an i7-9700K,
// and a question asks for a few rows, so the row is cut between the cores.
// Not on nn's workers: they belong to the runner, which may be using them.
func logSum(row []float32) float64 {
	m := math.Inf(-1)
	for _, v := range row {
		m = math.Max(m, float64(v))
	}
	parts := runtime.NumCPU()
	if len(row) < 1<<14 {
		parts = 1
	}
	sums := make([]float64, parts)
	var wg sync.WaitGroup
	for k := range parts {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for _, v := range row[k*len(row)/parts : (k+1)*len(row)/parts] {
				sums[k] += math.Exp(float64(v) - m)
			}
		}()
	}
	wg.Wait()
	var sum float64
	for _, s := range sums {
		sum += s
	}
	return m + math.Log(sum)
}
