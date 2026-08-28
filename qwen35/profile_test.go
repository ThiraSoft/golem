package qwen35

import (
	"testing"
	"time"
)

// TestVulkanPassProfile is not a test of anything; it is the instrument this
// model never had.
//
// Every performance number Qwen3.5 has ever been changed on came from
// ablation: take a kernel out, time the whole pass, call the difference that
// kernel's cost. That measures a difference and never a share, it cannot see
// time that belongs to no kernel at all, and it is why four structural
// rewrites of the delta net's scan — each correct, each measured — netted
// nothing between them. The card's own clock says where the pass goes.
//
// The tick count matters as much as the shares. The pool's period on this card
// is ten nanoseconds; if the sum of the spans times ten falls short of the
// wall clock around the submission, the difference is time the card spent
// inside no dispatch, which is the one cost no kernel rewrite can reach.
func TestVulkanPassProfile(t *testing.T) {
	m, err := Open(qwen38, 1024)
	if err != nil {
		t.Skipf("open: %v", err)
	}
	defer m.Close()
	if err := m.UseVulkan(); err != nil {
		t.Skipf("no Vulkan: %v", err)
	}

	for _, width := range []int{256, 512} {
		if width > m.gpuPipe.Columns() {
			continue
		}
		xs := make([][]float32, width)
		pos := make([]int, width)
		for i := range xs {
			xs[i] = make([]float32, m.Cfg.Dim)
			m.W.TokenEmbd.Row(1000+i, xs[i])
			pos[i] = i
		}

		tl, err := m.gpuPipe.NewTimeline()
		if err != nil {
			t.Fatal(err)
		}

		m.gpuPipe.ResetState()
		if _, err := m.gpuPipe.ForwardColumns(xs, pos); err != nil {
			t.Fatal(err)
		}

		m.gpuPipe.Profile(tl)
		m.gpuPipe.ResetState()
		start := time.Now()
		if _, err := m.gpuPipe.ForwardColumns(xs, pos); err != nil {
			t.Fatal(err)
		}
		took := time.Since(start)
		m.gpuPipe.Profile(nil)

		report, err := tl.Report(took)
		if err != nil {
			t.Fatal(err)
		}
		t.Logf("%d columns\n%s", width, report)
		tl.Close()
	}
}
