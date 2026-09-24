package vk

import "testing"

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
