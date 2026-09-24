package vk

import (
	"testing"
	"time"
)

// A pass narrows as the position grows, so its attention stays under the
// budget, and never below one column.
func TestWidthAtNarrowsWithThePosition(t *testing.T) {
	const budget = 3 << 20
	cases := []struct{ n, pos, want int }{
		{600, 0, 512},
		{600, 5000, 512},
		{600, 6000, 256},
		{600, 12288, 128},
		{132, 13312, 128},
		{600, 30000, 64},
		{3, 0, 2},
		{1, 1 << 30, 1},
		{600, 1 << 30, 1},
	}
	for _, c := range cases {
		if got := widthAt(c.n, c.pos, 512, budget); got != c.want {
			t.Errorf("widthAt(%d, %d) = %d, want %d", c.n, c.pos, got, c.want)
		}
	}
	// A narrower pipeline keeps its own line.
	if got := widthAt(600, 0, 256, budget); got != 256 {
		t.Errorf("a pipeline of 256 gave %d", got)
	}
}

// The budget was measured on Qwen3.8-27B's attention; a model with less of it
// a column gets proportionally more.
func TestAttendBudgetScalesWithTheAttention(t *testing.T) {
	p := &QwenPipeline{shape: QwenShape{Heads: 24, HeadDim: 256}}
	for i := 0; i < 64; i++ {
		p.isSSM = append(p.isSSM, i%4 != 3)
	}
	if got := p.attendBudget(); got != 3<<20 {
		t.Fatalf("27B budget %d", got)
	}
	p.shape.Heads = 12
	if got := p.attendBudget(); got != 6<<20 {
		t.Fatalf("half the heads gave %d", got)
	}
}

// A .golem pass costs most of its time before any attention, which the
// budget alone did not count: on t3g-27b it let 256 columns through at
// position 10496, 1.36 s fixed and 0.63 s attending, and the driver reset the
// card. Every width WidthAt picks up to 32768 must keep the whole pass under
// passLine, with the attention priced as it was measured at each width.
func TestWidthAtCountsAGolemPassFixedPart(t *testing.T) {
	p := &QwenPipeline{shape: QwenShape{Dim: 5120, FFN: 17408, Heads: 24, HeadDim: 256}, golem: &GolemKernels{}}
	for i := 0; i < 64; i++ {
		p.isSSM = append(p.isSSM, i%4 != 3)
	}
	if got := p.passFixed(256); got < 1340*time.Millisecond || got > 1370*time.Millisecond {
		t.Fatalf("a pass of 256 costs %v before its attention, measured 1357 ms", got)
	}
	// Seconds a column-position, measured on the same passes.
	attend := map[int]float64{256: 0.228e-6, 128: 0.263e-6, 64: 0.351e-6}
	widest := 0
	for pos := 0; pos <= 32768; pos += 64 {
		w := widthWith(600, pos, golemWidestPass, p.roomFor)
		widest = max(widest, w)
		per, ok := attend[w]
		if !ok {
			per = attend[64]
		}
		took := p.passFixed(w) + time.Duration(per*float64(w*(pos+w))*1e9)
		if took > passLine+50*time.Millisecond {
			t.Fatalf("position %d: %d columns take %v", pos, w, took)
		}
	}
	if widest != 128 {
		t.Errorf("the widest pass picked is %d, want 128", widest)
	}
	for _, c := range []struct{ pos, want int }{{0, 128}, {10496, 128}, {16384, 64}, {32704, 64}} {
		if got := widthWith(600, c.pos, golemWidestPass, p.roomFor); got != c.want {
			t.Errorf("position %d: %d columns, want %d", c.pos, got, c.want)
		}
	}
	// A quantized pipeline of the same shape keeps the whole budget.
	p.golem = nil
	if got := p.roomFor(512); got != p.attendBudget() {
		t.Errorf("a quantized pass of 512 has room for %d, want %d", got, p.attendBudget())
	}
}
