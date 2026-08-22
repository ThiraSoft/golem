package engine

import (
	"fmt"
	"os"
	"strings"
	"testing"
	"time"
)

// How the cost of reading a prompt grows with how much of it is already read.
// Batches are the 32 positions cmd/golem-server feeds together, timed one by
// one, so the shape of the curve is visible rather than an average over it.
func TestPrefillCostByDepth(t *testing.T) {
	path := os.Getenv("PREFILL_MODEL")
	if path == "" {
		t.Skip("PREFILL_MODEL is not set")
	}
	limit := 4096
	m, err := Open(path, 8192)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()

	// Real text, so the tokens are the sort a prompt actually carries.
	unit := "The scheduler keeps a run queue per core, and work stealing moves a goroutine across them when one core runs dry. "
	text := strings.Repeat(unit, 400)
	ids := m.Vocab.Encode(text, false, false)
	if len(ids) < limit {
		t.Fatalf("only %d tokens for a %d-token run", len(ids), limit)
	}
	ids = ids[:limit]

	const batch = 32
	const bucket = 512
	var sum [8]time.Duration
	var count [8]int

	out, err := os.Create(os.Getenv("PREFILL_OUT"))
	if err != nil {
		t.Fatal(err)
	}
	defer out.Close()
	say := func(format string, args ...any) {
		fmt.Fprintf(out, format+"\n", args...)
		t.Logf(format, args...)
	}
	say("%s, %d blocks — feeding %d tokens in batches of %d",
		m.Name, m.Blocks, limit, batch)
	whole := time.Now()
	for at := 0; at < len(ids); at += batch {
		to := min(at+batch, len(ids))
		start := time.Now()
		m.Forward.ForwardBatch(ids[at:to], at)
		took := time.Since(start)
		b := at / bucket
		if b < len(sum) {
			sum[b] += took
			count[b] += batch
		}
	}
	total := time.Since(whole)

	say("positions        per batch     rate")
	for b := range sum {
		if count[b] == 0 {
			continue
		}
		rate := float64(count[b]) / sum[b].Seconds()
		say("%5d-%5d   %9s   %6.1f tok/s",
			b*bucket, (b+1)*bucket-1,
			(sum[b] / time.Duration(count[b]/batch)).Round(time.Millisecond), rate)
	}
	say("whole prompt: %d tokens in %s (%.1f tok/s)",
		limit, total.Round(time.Millisecond), float64(limit)/total.Seconds())
}
