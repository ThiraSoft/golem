package qwen35

import (
	"testing"

	"github.com/ThiraSoft/golem/internal/heavy"
)

// TestVulkanResidency says what the card is asked for and what it holds, which
// is the one number that decides whether the 27B draws at thirty tokens a
// second or at four. A checkpoint that overflows does not fail: the driver puts
// the last buffer in system memory and the model reads it across the bus every
// token, fluently and eighty per cent slower.
func TestVulkanResidency(t *testing.T) {
	heavy.Skip(t, "it puts a model on the card")
	for _, drafts := range []bool{true, false} {
		m, err := Open(qwen38, 2048)
		if err != nil {
			t.Skipf("open: %v", err)
		}
		if !drafts {
			m.SkipDraftBlock()
		}
		if err := m.UseVulkanStack(); err != nil {
			m.Close()
			t.Skipf("no Vulkan: %v", err)
		}
		d, err := m.device()
		if err != nil {
			m.Close()
			t.Fatal(err)
		}
		stack := m.gpuPipe.DeviceBytes()
		heap := d.DeviceLocalBytes()
		head := uint64(m.W.OutputHead.Rows) * uint64(m.W.OutputHead.Cols) / 256 * 212
		t.Logf("drafts=%v: heap %d MiB, stack %d, head %d, together %d, room left %d MiB",
			drafts, heap>>20, stack>>20, head>>20, (stack+head)>>20,
			(int64(heap)-int64(stack+head))>>20)
		m.Close()
	}
}
