package qwen35

import (
	"os"
	"sort"
	"strconv"
	"testing"
	"time"

	"github.com/ThiraSoft/golem/internal/heavy"
)

// TestVulkanPassCost times one wide pass at a few widths and depths, and
// splits its card time into the attention and everything else. It is what
// WidthAt's budget is drawn from: a pass is one submission, and the driver
// resets the card past about two seconds.
//
// A pass whose predicted cost is past 1.4 s is skipped, not run: the
// prediction is the fixed part measured at the shallowest depth plus the
// attention's cost a column-position measured so far.
func TestVulkanPassCost(t *testing.T) {
	heavy.Skip(t, "a measurement, and it fills eight thousand positions")
	path := envOr("GOLEM_DEPTH_MODEL", bonsaiDir+"Ternary-Bonsai-2-27B-PQ2_0-mtp.gguf")
	m, err := Open(path, 32768)
	if err != nil {
		t.Skipf("open: %v", err)
	}
	defer m.Close()
	if err := m.UseVulkan(); err != nil {
		t.Skipf("no Vulkan: %v", err)
	}
	tok := func(i int) int32 { return int32(1000 + i*7919%60000) }
	const limit = 1400 * time.Millisecond

	widths := []int{64, 128, 256}
	if os.Getenv("GOLEM_COST_WIDE") != "" {
		widths = append(widths, 512)
	}
	fixed := map[int]time.Duration{}
	perCP := 0.0 // seconds of attention a column-position, the largest seen

	filled := 0
	for _, depth := range []int{1024, 4096, 8192} {
		for filled < depth {
			n := min(64, depth-filled)
			toks := make([]int32, n)
			for i := range toks {
				toks[i] = tok(filled + i)
			}
			m.ForwardBatch(toks, filled)
			filled += n
		}
		for _, w := range widths {
			if w > m.gpuPipe.Columns() {
				continue
			}
			cp := float64(w) * float64(depth+w)
			if f, ok := fixed[w]; ok {
				predicted := f + time.Duration(perCP*cp*1e9)
				if predicted > limit {
					t.Logf("position %d, %d columns: skipped, predicted %v", depth, w, predicted)
					continue
				}
			}
			xs := make([][]float32, w)
			pos := make([]int, w)
			for i := range xs {
				xs[i] = make([]float32, m.Cfg.Dim)
				m.W.TokenEmbd.Row(int(tok(depth+i)), xs[i])
				pos[i] = depth + i
			}
			run := func() time.Duration {
				start := time.Now()
				if _, err := m.gpuPipe.ForwardColumns(xs, pos); err != nil {
					t.Fatal(err)
				}
				return time.Since(start)
			}
			run() // records the program at this width
			walls := []time.Duration{run(), run(), run()}
			sort.Slice(walls, func(i, j int) bool { return walls[i] < walls[j] })

			tl, err := m.gpuPipe.NewTimeline()
			if err != nil {
				t.Fatal(err)
			}
			m.gpuPipe.Profile(tl)
			run() // recorded with the stamps
			run()
			m.gpuPipe.Profile(nil)
			spans, err := tl.Spans()
			if err != nil {
				t.Fatal(err)
			}
			tl.Close()
			var all, attn uint64
			for _, s := range spans {
				all += s.Ticks
				if s.Label == "attn gqa" {
					attn += s.Ticks
				}
			}
			// The pool's period on this card is ten nanoseconds.
			card := time.Duration(all * 10)
			att := time.Duration(attn * 10)
			rest := card - att
			if _, ok := fixed[w]; !ok {
				fixed[w] = rest
			}
			perCP = max(perCP, att.Seconds()/cp)
			t.Logf("position %5d, %3d columns: wall %v, card %v = attention %v + rest %v; %.3f µs a column-position, rest %.0f µs a column",
				depth, w, walls[1].Round(time.Millisecond), card.Round(time.Millisecond),
				att.Round(time.Millisecond), rest.Round(time.Millisecond),
				att.Seconds()/cp*1e6, rest.Seconds()/float64(w)*1e6)
		}
	}
}

// TestVulkanPrefillRate reads a prompt of GOLEM_PREFILL_TO positions (24576
// by default) in runs of GOLEM_DEPTH_FILL tokens (2048 by default), the
// widths left to WidthAt, and logs the rate of every run.
func TestVulkanPrefillRate(t *testing.T) {
	heavy.Skip(t, "a measurement, and it fills twenty-four thousand positions")
	path := envOr("GOLEM_DEPTH_MODEL", bonsaiDir+"Ternary-Bonsai-2-27B-PQ2_0-mtp.gguf")
	to, fill := 24576, 2048
	if s := os.Getenv("GOLEM_PREFILL_TO"); s != "" {
		to, _ = strconv.Atoi(s)
	}
	if s := os.Getenv("GOLEM_DEPTH_FILL"); s != "" {
		fill, _ = strconv.Atoi(s)
	}
	m, err := Open(path, 32768)
	if err != nil {
		t.Skipf("open: %v", err)
	}
	defer m.Close()
	if err := m.UseVulkan(); err != nil {
		t.Skipf("no Vulkan: %v", err)
	}
	tok := func(i int) int32 { return int32(1000 + i*7919%60000) }
	m.ForwardBatch([]int32{tok(0)}, 0) // records the narrow programs outside the clock
	var total time.Duration
	for filled := 0; filled < to; {
		n := min(fill, to-filled)
		toks := make([]int32, n)
		for i := range toks {
			toks[i] = tok(filled + i)
		}
		start := time.Now()
		m.ForwardBatch(toks, filled)
		took := time.Since(start)
		total += took
		t.Logf("%5d to %5d: widths %d then %d, %.1f t/s", filled, filled+n,
			m.gpuPipe.WidthAt(n, filled), m.gpuPipe.WidthAt(n, filled+n-1), float64(n)/took.Seconds())
		filled += n
	}
	t.Logf("%d positions in %v, %.1f t/s", to, total.Round(time.Millisecond), float64(to)/total.Seconds())
}
