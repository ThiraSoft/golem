package qwen35

import (
	"testing"
	"time"
)

// TestVulkanWidthCost is what a pass costs at each width it may take. A second
// column is meant to be almost free — the weights are read once for both — and
// on the quantized path it is: 1.012 of one column. On the .golem path it is
// 1.20, and the shape of this curve says whether that is a fixed price for any
// width past one or a price a column.
func TestVulkanWidthCost(t *testing.T) {
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

	one := time.Duration(0)
	for _, w := range []int{1, 2, 4, 8, 16} {
		if w > m.gpuPipe.Columns() {
			continue
		}
		xs := make([][]float32, w)
		pos := make([]int, w)
		for i := range xs {
			xs[i], pos[i] = x, i
		}
		run := func(n int) time.Duration {
			start := time.Now()
			for i := 0; i < n; i++ {
				if _, err := m.gpuPipe.ForwardColumns(xs, pos); err != nil {
					t.Fatal(err)
				}
			}
			return time.Since(start) / time.Duration(n)
		}
		run(4)
		best := time.Hour
		for r := 0; r < 3; r++ {
			if d := run(12); d < best {
				best = d
			}
		}
		if w == 1 {
			one = best
		}
		t.Logf("%2d columns %8v  %.3fx one column  %6.2f us a column",
			w, best, best.Seconds()/one.Seconds(), float64(best.Microseconds())/float64(w))
	}
}
