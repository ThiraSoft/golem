package laya

import (
	"context"
	"encoding/json"
	"os"
	"strings"
	"testing"
	"time"
)

// benchRequest is the email ref/laya/bench.py times, and its four questions.
func benchRequest(tb testing.TB) (json.RawMessage, []Named) {
	state := json.RawMessage(`{"subject": "Refund not received", "body": "I cancelled my subscription two weeks ago and I still have not seen the refund on my card. Order #88213. Can someone please look into this? Thanks, Maria"}`)
	qs, err := ParseQuestions(json.RawMessage(`{
		"department": {"type": "choice", "instructions": "Which team should handle this email?", "criteria": {"billing": "payments, refunds, invoices", "support": "technical problems", "sales": "pricing and new purchases"}},
		"urgency": {"type": "score", "instructions": "How urgent is this message?", "criteria": ["not urgent", "somewhat urgent", "urgent", "critical"]},
		"is_refund": {"type": "noul", "instructions": "The customer is asking about a refund."},
		"tone": {"type": "choice", "instructions": "What is the tone of the writer?", "criteria": ["angry", "neutral", "polite", "sarcastic", "desperate", "happy"]}}`))
	if err != nil {
		tb.Fatal(err)
	}
	return state, qs
}

// BenchmarkDecide is one question of that request, the four, and the four
// asked eight times over, on the processor or, with GOLEM_LAYA_VULKAN set, on
// the card.
func BenchmarkDecide(b *testing.B) {
	dir := os.Getenv("GOLEM_MODEL_LAYA")
	if dir == "" {
		b.Skip("GOLEM_MODEL_LAYA is not set")
	}
	m, err := Open(dir)
	if err != nil {
		b.Fatal(err)
	}
	defer m.Close()
	if os.Getenv("GOLEM_LAYA_VULKAN") != "" {
		if err := m.UseVulkan(); err != nil {
			b.Fatal(err)
		}
	}
	state, qs := benchRequest(b)
	for _, c := range []struct {
		name string
		qs   []Named
	}{{"1q", qs[2:3]}, {"4q", qs}, {"32q", repeat(qs, 8)}} {
		b.Run(c.name, func(b *testing.B) {
			tokens := 0
			for i := 0; i < b.N; i++ {
				res, err := m.Decide(context.Background(), state, c.qs)
				if err != nil {
					b.Fatal(err)
				}
				tokens += res.InputTokens
			}
			b.ReportMetric(float64(tokens)/b.Elapsed().Seconds(), "tok/s")
		})
	}
}

// TestProfileVulkan prints where each request's pass spends the card's time.
// It is a measurement, not a check.
func TestProfileVulkan(t *testing.T) {
	if os.Getenv("GOLEM_LAYA_PROFILE") == "" {
		t.Skip("GOLEM_LAYA_PROFILE is not set")
	}
	m := openModel(t)
	if err := m.UseVulkan(); err != nil {
		t.Skip(err)
	}
	state, qs := benchRequest(t)
	for _, c := range []struct {
		name string
		qs   []Named
	}{{"1q", qs[2:3]}, {"4q", qs}, {"32q", repeat(qs, 8)}} {
		m.Decide(context.Background(), state, c.qs)
		if err := m.gpu.pipe.Profile(); err != nil {
			t.Fatal(err)
		}
		start := time.Now()
		if _, err := m.Decide(context.Background(), state, c.qs); err != nil {
			t.Fatal(err)
		}
		took := time.Since(start)
		report, err := m.gpu.pipe.Report(took)
		if err != nil {
			t.Fatal(err)
		}
		t.Logf("%s, %v\n%s", c.name, took, report)
	}
}

func repeat(qs []Named, n int) []Named {
	var out []Named
	for i := 0; i < n; i++ {
		for _, q := range qs {
			q.ID = q.ID + strings.Repeat("'", i)
			out = append(out, q)
		}
	}
	return out
}
