package qwen35

import (
	"os"
	"sort"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/ThiraSoft/golem/internal/heavy"
	"github.com/ThiraSoft/golem/vk"
)

// TestVulkanDepthProfile is what one token costs as the conversation grows,
// and how much of it is the attention.
//
// The other two profiles run a pass at position 32, where the sixteen full
// attention blocks read a handful of keys and weigh nothing. Generation on
// this model collapses with the context — 61 t/s on a short prompt, 12.6 at
// ten thousand positions — and a profile at 32 cannot say why.
//
// GOLEM_DEPTH_MODEL names the checkpoint (Bonsai PQ2_0 with its grafted
// prediction block by default), GOLEM_DEPTH_CONTEXT the context.
func TestVulkanDepthProfile(t *testing.T) {
	heavy.Skip(t, "a profile is a measurement, and it fills twenty-four thousand positions")
	path := envOr("GOLEM_DEPTH_MODEL", bonsaiDir+"Ternary-Bonsai-2-27B-PQ2_0-mtp.gguf")
	maxContext := 32768
	if s := os.Getenv("GOLEM_DEPTH_CONTEXT"); s != "" {
		maxContext, _ = strconv.Atoi(s)
	}
	// GOLEM_DEPTH_FILL caps how many tokens the fill hands ForwardBatch at
	// once, and so the width of its passes. A .golem pass of 256 columns is
	// 1.2 s before any attention, and WidthAt's budget, measured on Bonsai,
	// let t3g reach 256 at position 10496 and reset the card there.
	fill := 2048
	if s := os.Getenv("GOLEM_DEPTH_FILL"); s != "" {
		fill, _ = strconv.Atoi(s)
	}
	m, err := Open(path, maxContext)
	if err != nil {
		t.Skipf("open: %v", err)
	}
	defer m.Close()
	if err := m.UseVulkan(); err != nil {
		t.Skipf("no Vulkan: %v", err)
	}
	logits := make([]float32, m.Cfg.Vocab)
	tok := func(i int) int32 { return int32(1000 + i*7919%60000) }

	median := func(n int, f func()) time.Duration {
		ds := make([]time.Duration, n)
		for i := range ds {
			start := time.Now()
			f()
			ds[i] = time.Since(start)
		}
		sort.Slice(ds, func(i, j int) bool { return ds[i] < ds[j] })
		return ds[n/2]
	}

	eh := make([]float32, 2*m.Cfg.Dim)
	for i := range eh {
		eh[i] = 0.01 * float32(i%17-8)
	}

	filled := 0
	for _, depth := range []int{1024, 4096, 8192, 16384, 24576} {
		if depth+4 > maxContext {
			break
		}
		for filled < depth {
			n := min(fill, depth-filled)
			toks := make([]int32, n)
			for i := range toks {
				toks[i] = tok(filled + i)
			}
			m.ForwardBatch(toks, filled)
			filled += n
		}

		one := median(7, func() { m.ForwardBatch([]int32{tok(depth)}, depth) })
		two := median(7, func() { m.ForwardBatch([]int32{tok(depth), tok(depth + 1)}, depth) })
		head := median(5, func() { m.Logits(m.x, logits) })
		draft := time.Duration(0)
		if m.gpuPipe.HasMTP() {
			draft = median(5, func() {
				if _, err := m.gpuPipe.DraftMTPAt(eh, vk.QwenPlace{Pos: depth, T: depth, H: depth, W: depth}); err != nil {
					t.Fatal(err)
				}
			})
		}

		tl, err := m.gpuPipe.NewTimeline()
		if err != nil {
			t.Fatal(err)
		}
		m.gpuPipe.Profile(tl)
		m.ForwardBatch([]int32{tok(depth)}, depth) // recorded with the stamps
		start := time.Now()
		m.ForwardBatch([]int32{tok(depth)}, depth)
		took := time.Since(start)
		m.gpuPipe.Profile(nil)
		report, err := tl.Report(took)
		if err != nil {
			t.Fatal(err)
		}
		tl.Close()

		t.Logf("position %d: one column %v, two %v, head %v, draft %v; a plain token %.1f t/s\n%s",
			depth, one, two, head, draft, 1/(one+head).Seconds(), indent(report))
	}
}

func indent(s string) string {
	return "    " + strings.ReplaceAll(strings.TrimRight(s, "\n"), "\n", "\n    ")
}
