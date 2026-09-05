package qwen35

import (
	"testing"
	"time"

	"github.com/ThiraSoft/golem/internal/heavy"
)

// TestVulkanPassProfile is not a test of anything; it is the instrument this
// model never had.
//
// Every performance number Qwen3.8 has ever been changed on came from
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
	heavy.Skip(t, "a profile is a measurement, not a check, and this one runs longer than the test timeout allows")
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

// TestVulkanTokenProfile is the same instrument on a pass of one column, which
// is the pass generation runs and the one the wide profile above cannot see.
// The two are not the same problem: a prompt is bound by the tiled products
// and a token by everything that is one workgroup, and on the other two
// engines it was the token profile that found a fifth of the time sitting in
// an attention built for thirty-two columns.
func TestVulkanTokenProfile(t *testing.T) {
	heavy.Skip(t, "a profile is a measurement, not a check")
	m, err := Open(qwen38, 1024)
	if err != nil {
		t.Skipf("open: %v", err)
	}
	defer m.Close()
	if err := m.UseVulkan(); err != nil {
		t.Skipf("no Vulkan: %v", err)
	}

	x := make([]float32, m.Cfg.Dim)
	m.W.TokenEmbd.Row(1000, x)

	// One column and two, because a speculative step is one of each and the
	// second is meant to be almost free: the weights are read once for both.
	for _, columns := range []int{1, 2} {
		xs := make([][]float32, columns)
		pos := make([]int, columns)
		for i := range xs {
			xs[i], pos[i] = x, 32+i
		}
		tl, err := m.gpuPipe.NewTimeline()
		if err != nil {
			t.Fatal(err)
		}

		m.gpuPipe.ResetState()
		for p := 0; p < 32; p++ {
			if _, err := m.gpuPipe.ForwardColumns([][]float32{x}, []int{p}); err != nil {
				t.Fatal(err)
			}
		}
		if _, err := m.gpuPipe.ForwardColumns(xs, pos); err != nil {
			t.Fatal(err)
		}
		m.gpuPipe.Profile(tl)
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
		t.Logf("%d column(s) in %v\n%s", columns, took, report)
		tl.Close()
	}
}
