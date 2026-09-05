package qwen35

import (
	"fmt"
	"os"
	"testing"
	"time"

	"github.com/ThiraSoft/golem/internal/heavy"
	"github.com/ThiraSoft/golem/tensors"
	"github.com/ThiraSoft/golem/token/bytebpe"
	"github.com/ThiraSoft/golem/vk"
)

// What a token costs on each path, and what the prediction block is worth on
// each. The draft is the checkpoint's own multi-token-prediction head, so a
// token it guesses right is a token the trunk never has to read its weights
// for — but only if the pass that verifies it is cheaper than the pass it
// saves, and that is not true everywhere.
//
// The numbers this prints are the ones in README.md.
// onCard skips the processor half of TestGenerationCost. Set GOLEM_CARD_ONLY
// for a checkpoint only the card can run at a useful rate.
var onCard = os.Getenv("GOLEM_CARD_ONLY") != ""

func TestGenerationCost(t *testing.T) {
	// A bench, not a test: everything below is fmt.Printf and the numbers go in
	// README.md. It asserts nothing, and it is 384 seconds of a 660-second
	// package — so it is the thing to skip when the question is whether the
	// code is still correct, and the thing to run when the question is what it
	// costs.
	heavy.Skip(t, "a bench: it reads a checkpoint of tens of gigabytes and times a run over it")
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

	// headCost is what one reading of the logit head costs on its own.
	//
	// It is here because a speculative step reads the head three times where a
	// plain token reads it once, so whether drafting pays turns on this number
	// and it was being inferred rather than measured.
	headCost := func() time.Duration {
		m.Reset()
		hs := m.ForwardBatch(toks, 0)
		hidden := hs[len(hs)-1]
		const reps = 16
		m.Logits(hidden, logits)
		start := time.Now()
		for i := 0; i < reps; i++ {
			m.Logits(hidden, logits)
		}
		return time.Since(start) / reps
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

	// specPair is the pass a speculative step actually takes, which is not the
	// pass of two above: it snapshots every delta net's state so that a refused
	// draft can be undone. The two are recorded from the same function and
	// differ by a push constant, so they ought to cost the same — and this is
	// here because they did not, and a step was paying for the difference
	// three times over before anyone timed the halves separately.
	specPair := func() time.Duration {
		if !m.Speculate() {
			return 0
		}
		m.Reset()
		hs := m.ForwardBatch(toks, 0)
		hidden := hs[len(hs)-1]
		pos := len(toks)
		e0 := make([]float32, m.Cfg.Dim)
		e1 := make([]float32, m.Cfg.Dim)
		m.W.TokenEmbd.Row(int(toks[0]), e0)
		m.W.TokenEmbd.Row(int(toks[1]), e1)
		at := Place{Pos: pos, T: pos, H: pos, W: pos}
		next := at.Next()
		const reps = 8
		run := func() {
			if _, err := m.gpuPipe.ForwardSpeculativeAt(
				[][]float32{e0, e1}, []vk.QwenPlace{at.gpu(), next.gpu()}); err != nil {
				t.Fatal(err)
			}
		}
		run()
		start := time.Now()
		for i := 0; i < reps; i++ {
			run()
		}
		_ = hidden
		return time.Since(start) / reps
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
		if sp := specPair(); sp > 0 {
			fmt.Printf("the speculative two %v (%.2f of a token, %.2f of a plain pass of two)\n",
				sp.Round(100*time.Microsecond), sp.Seconds()/each.Seconds(), sp.Seconds()/pair.Seconds())
		}
		head := headCost()
		fmt.Printf("the logit head      %v (%.2f of a token), read three times a step\n",
			head.Round(100*time.Microsecond), head.Seconds()/each.Seconds())
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

	// The processor half is skipped for a checkpoint the processor cannot run
	// in any reasonable time. A .golem 27B decodes a trellis for every weight
	// of every token on eight cores; sixteen tokens that way is half an hour,
	// and the question this file asks is what the card costs.
	if !onCard {
		report("processor", 16)
	}

	if err := m.UseVulkan(); err != nil {
		t.Skipf("no Vulkan: %v", err)
	}
	report("card", 64)
}

// What a pass costs by its width, which is where the prompt rate comes from.
//
// A pass reads every weight in the model once whatever it carries, so the
// first columns are nearly free. Measured on an RX 9070 XT:
//
//	positions   1       16      32      64      128      256      512
//	the pass    31.1ms  69.4ms  62.8ms  79.6ms  139.3ms  224.5ms  414.3ms
//	a second    32      231     510     804     919      1140     1236
//
// The widest is the fastest and the test says so, because that is the property
// vk/qwen_pipeline.go's qwenWidths depends on: a run takes the widest pass that
// fits what is left, and a width that has stopped paying should fail here
// rather than ship. It did stop paying once — a mat-vec at thirty-two columns
// is slower than one at sixteen, because the accumulator a thread carries a
// column in stops fitting in registers, which is why the projections that can
// go through the tiled product now do.
func TestPassWidthCost(t *testing.T) {
	heavy.Skip(t, "it puts a model on the card")
	g, err := tensors.OpenGGUF(qwen38)
	if err != nil {
		t.Skipf("open: %v", err)
	}
	m, err := New(g, 1024)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	if err := m.UseVulkan(); err != nil {
		t.Skipf("no Vulkan: %v", err)
	}
	emb := make([]float32, m.Cfg.Dim)
	m.W.TokenEmbd.Row(100, emb)

	best, at := 0.0, 0
	for _, w := range []int{1, 16, 32, 64, 128, 256, 512} {
		if w > m.gpuPipe.Columns() {
			continue
		}
		xs := make([][]float32, w)
		pos := make([]int, w)
		for i := range xs {
			xs[i], pos[i] = emb, 100+i
		}
		for i := 0; i < 2; i++ {
			if _, err := m.gpuPipe.ForwardColumns(xs, pos); err != nil {
				t.Fatal(err)
			}
		}
		reps := max(2, 64/w)
		start := time.Now()
		for i := 0; i < reps; i++ {
			if _, err := m.gpuPipe.ForwardColumns(xs, pos); err != nil {
				t.Fatal(err)
			}
		}
		each := time.Since(start) / time.Duration(reps)
		rate := float64(w) / each.Seconds()
		fmt.Printf("%3d columns: %v a pass (%.0f positions/s)\n", w, each.Round(100*time.Microsecond), rate)
		if rate > best {
			best, at = rate, w
		}
	}
	if at != m.gpuPipe.Columns() {
		t.Errorf("the widest pass is %d columns and the fastest is %d: qwenWidths wants moving",
			m.gpuPipe.Columns(), at)
	}

	// And a prompt read end to end, which is what the README quotes.
	toks := make([]int32, 512)
	for i := range toks {
		toks[i] = int32(100 + i)
	}
	m.Reset()
	m.ForwardBatch(toks[:64], 0)
	m.Reset()
	start := time.Now()
	m.ForwardBatch(toks, 0)
	d := time.Since(start)
	fmt.Printf("a prompt of %d: %v (%.0f positions/s)\n", len(toks), d.Round(time.Millisecond),
		float64(len(toks))/d.Seconds())
}
