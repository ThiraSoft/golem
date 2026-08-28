package qwen35

import (
	"fmt"
	"testing"
	"time"

	"github.com/ThiraSoft/golem/tensors"
	"github.com/ThiraSoft/golem/token/bytebpe"
)

// What a token costs on each path, and what the prediction block is worth on
// each. The draft is the checkpoint's own multi-token-prediction head, so a
// token it guesses right is a token the trunk never has to read its weights
// for — but only if the pass that verifies it is cheaper than the pass it
// saves, and that is not true everywhere.
//
// The numbers this prints are the ones in README.md.
func TestGenerationCost(t *testing.T) {
	g, err := tensors.OpenGGUF(qwen38)
	if err != nil {
		t.Skipf("open: %v", err)
	}
	vocab, err := bytebpe.Load(g)
	if err != nil {
		t.Fatal(err)
	}
	m, err := New(g, 1024)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	if !m.HasMTP() {
		t.Skip("the checkpoint carries no prediction block")
	}

	prompt := "<|im_start|>user\nWrite a short paragraph about the Mediterranean sea.<|im_end|>\n<|im_start|>assistant\n"
	toks := vocab.Encode(prompt, false, true)
	logits := make([]float32, m.Cfg.Vocab)

	// plain generates n tokens a token at a time, which is the rate to beat.
	plain := func(n int) (float64, time.Duration) {
		m.Reset()
		hs := m.ForwardBatch(toks, 0)
		hidden := hs[len(hs)-1]
		pos := len(toks)
		start := time.Now()
		drawn := 0
		for ; drawn < n; drawn++ {
			m.Logits(hidden, logits)
			id := argmax(logits)
			if vocab.IsEOG(id) {
				break
			}
			hidden = m.Forward(id, pos)
			pos++
		}
		d := time.Since(start)
		return float64(drawn) / d.Seconds(), d / time.Duration(max(drawn, 1))
	}

	// prefill is what reading the prompt costs, which the same pass width
	// decides.
	prefill := func() float64 {
		m.Reset()
		start := time.Now()
		m.ForwardBatch(toks, 0)
		return float64(len(toks)) / time.Since(start).Seconds()
	}

	// draftCost is what one call of the prediction block costs, and pairCost
	// what a pass of two columns costs — the two halves of a speculative step.
	costs := func() (draft, pair time.Duration) {
		m.Reset()
		hs := m.ForwardBatch(toks, 0)
		hidden := hs[len(hs)-1]
		pos := len(toks)
		out := make([]float32, m.Cfg.Vocab)
		const reps = 8
		m.ForwardMTP(toks[len(toks)-1], hidden, pos, out)
		start := time.Now()
		for i := 0; i < reps; i++ {
			m.ForwardMTP(toks[len(toks)-1], hidden, pos, out)
		}
		draft = time.Since(start) / reps
		pair2 := []int32{toks[0], toks[1]}
		m.ForwardBatch(pair2, pos)
		start = time.Now()
		for i := 0; i < reps; i++ {
			m.ForwardBatch(pair2, pos)
		}
		pair = time.Since(start) / reps
		return draft, pair
	}

	report := func(where string, n int) {
		pf := prefill()
		rate, each := plain(n)
		draft, pair := costs()
		fmt.Printf("\n== %s ==\n", where)
		fmt.Printf("prompt              %.1f positions/s\n", pf)
		fmt.Printf("a token at a time   %.2f t/s (%v a token)\n", rate, each.Round(100*time.Microsecond))
		fmt.Printf("one draft           %v (%.2f of a token)\n", draft.Round(100*time.Microsecond),
			draft.Seconds()/each.Seconds())
		fmt.Printf("a pass of two       %v (%.2f of a token)\n", pair.Round(100*time.Microsecond),
			pair.Seconds()/each.Seconds())
		// A step costs the draft and the pass of two, and returns one token
		// plus the accepted one. Break-even is where that beats a plain token.
		step := (draft + pair).Seconds() / each.Seconds()
		fmt.Printf("break-even needs    %.0f%% of drafts accepted\n", 100*(step-1))

		if m.Speculate() {
			s, err := m.NewSpeculator()
			if err != nil {
				t.Fatal(err)
			}
			m.Reset()
			hs := m.ForwardBatch(toks, 0)
			hidden := hs[len(hs)-1]
			pos := len(toks)
			m.Logits(hidden, logits)
			id := argmax(logits)
			drawn, stop := 1, false
			start := time.Now()
			for drawn < n && !stop {
				ids, h, err := s.Step(id, hidden, pos, argmax)
				if err != nil {
					t.Fatal(err)
				}
				for _, one := range ids {
					if vocab.IsEOG(one) {
						stop = true
						break
					}
				}
				drawn += len(ids)
				pos += len(ids)
				hidden = h
				id = ids[len(ids)-1]
			}
			d := time.Since(start)
			fmt.Printf("drafting            %.2f t/s (%d of %d drafts accepted, %.0f%%)\n",
				float64(drawn)/d.Seconds(), s.Accepted, s.Drafted,
				100*float64(s.Accepted)/float64(max(s.Drafted, 1)))
		} else {
			fmt.Printf("drafting            refused on this path\n")
		}
	}

	report("processor", 16)

	if err := m.UseVulkan(); err != nil {
		t.Skipf("no Vulkan: %v", err)
	}
	report("card", 64)
}
