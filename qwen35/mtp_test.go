package qwen35

import (
	"fmt"
	"strings"
	"testing"
	"time"

	"github.com/ThiraSoft/golem/tensors"
	"github.com/ThiraSoft/golem/token/bytebpe"
)

// TestMTPAcceptance measures how often the prediction block's draft is the
// token the model itself goes on to choose. That ratio is the whole of what
// speculative decoding can buy.
func TestMTPAcceptance(t *testing.T) {
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
	if err := m.UseVulkan(); err != nil {
		t.Fatalf("vulkan: %v", err)
	}
	if !m.HasMTP() {
		t.Fatalf("the checkpoint carries no prediction block")
	}

	prompt := "<|im_start|>user\nWrite a short paragraph about the Mediterranean sea.<|im_end|>\n<|im_start|>assistant\n"
	toks := vocab.Encode(prompt, false, true)
	hs := m.ForwardBatch(toks, 0)
	hidden := hs[len(hs)-1]

	logits := make([]float32, m.Cfg.Vocab)
	draft := make([]float32, m.Cfg.Vocab)
	pick := func(v []float32) int32 {
		best, bi := float32(-1e30), 0
		for i, x := range v {
			if x > best {
				best, bi = x, i
			}
		}
		return int32(bi)
	}

	pos := len(toks)
	accepted, tried := 0, 0
	var mtpTime, trunkTime time.Duration

	m.Logits(hidden, logits)
	id := pick(logits)

	for n := 0; n < 48; n++ {
		if vocab.IsEOG(id) {
			break
		}
		// Draft the token after `id`, from `id` and the state before it.
		t0 := time.Now()
		m.ForwardMTP(id, hidden, pos, draft)
		mtpTime += time.Since(t0)
		guess := pick(draft)

		t1 := time.Now()
		hidden = m.Forward(id, pos)
		pos++
		m.Logits(hidden, logits)
		trunkTime += time.Since(t1)
		truth := pick(logits)

		tried++
		if guess == truth {
			accepted++
		}
		id = truth
	}

	fmt.Printf("MTP draft accepted %d/%d (%.1f%%)\n", accepted, tried, 100*float64(accepted)/float64(tried))
	fmt.Printf("one draft: %v   one trunk token: %v\n", mtpTime/time.Duration(tried), trunkTime/time.Duration(tried))
}

// TestSpeculativeGenerate generates the same answer twice, once a token at a
// time and once with the prediction block drafting, and reports both rates.
func TestSpeculativeGenerate(t *testing.T) {
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
	if err := m.UseVulkan(); err != nil {
		t.Fatalf("vulkan: %v", err)
	}

	prompt := "<|im_start|>user\nWrite a short paragraph about the Mediterranean sea.<|im_end|>\n<|im_start|>assistant\n"
	toks := vocab.Encode(prompt, false, true)
	logits := make([]float32, m.Cfg.Vocab)
	pick := func(v []float32) int32 {
		best, bi := float32(-1e30), 0
		for i, x := range v {
			if x > best {
				best, bi = x, i
			}
		}
		return int32(bi)
	}

	const want = 64

	run := func(spec bool) (string, float64, string) {
		m.Reset()
		hs := m.ForwardBatch(toks, 0)
		hidden := hs[len(hs)-1]
		pos := len(toks)
		var out strings.Builder
		var sp *Speculator
		if spec {
			sp, err = m.NewSpeculator()
			if err != nil {
				t.Fatal(err)
			}
		}

		m.Logits(hidden, logits)
		id := pick(logits)
		n := 0
		start := time.Now()
		for n < want {
			if vocab.IsEOG(id) {
				break
			}
			out.WriteString(vocab.Piece(id, false))
			n++

			if !spec {
				hidden = m.Forward(id, pos)
				pos++
				m.Logits(hidden, logits)
				id = pick(logits)
				continue
			}
			next, h, err := sp.Step(id, hidden, pos, pick)
			if err != nil {
				t.Fatal(err)
			}
			for _, tok := range next[:len(next)-1] {
				if vocab.IsEOG(tok) {
					id = tok
					break
				}
				out.WriteString(vocab.Piece(tok, false))
				n++
			}
			id = next[len(next)-1]
			hidden = h
			pos += len(next)
		}
		rate := float64(n) / time.Since(start).Seconds()
		note := ""
		if spec {
			note = fmt.Sprintf("accepted %d/%d (%.1f%%)", sp.Accepted, sp.Drafted,
				100*float64(sp.Accepted)/float64(sp.Drafted))
		}
		return out.String(), rate, note
	}

	plainText, plainRate, _ := run(false)
	specText, specRate, note := run(true)

	fmt.Printf("plain:       %.2f t/s\n", plainRate)
	fmt.Printf("speculative: %.2f t/s  (%.2fx)  %s\n", specRate, specRate/plainRate, note)
	// An accepted draft emits two tokens at once, so the speculative run can
	// overshoot the count by one. Compare what both of them said.
	n := min(len(plainText), len(specText))
	if plainText[:n] != specText[:n] {
		fmt.Printf("--- plain ---\n%s\n--- speculative ---\n%s\n", plainText, specText)
		t.Errorf("speculation changed the answer")
	} else {
		fmt.Printf("=== same answer both ways ===\n%s\n", plainText)
	}
}
