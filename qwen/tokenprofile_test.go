package qwen

import (
	"testing"
	"time"
)

// TestVulkanTokenProfile is TestVulkanPromptProfile for a single token, which
// is the pass generation actually runs: one column through every block, where
// the weights are read once for one answer and the dispatches are as many as
// they are for five hundred. It says which of the two the gap against
// llama.cpp's generation is.
func TestVulkanTokenProfile(t *testing.T) {
	_, m := openQuantizedStack(t)
	tl, err := m.NewStackTimeline()
	if err != nil {
		t.Fatal(err)
	}
	defer tl.Close()

	m.Reset()
	for pos := 0; pos < 32; pos++ {
		m.Forward(int32(pos+100), pos)
	}
	m.ProfileStack(tl)
	start := time.Now()
	m.Forward(1000, 32)
	took := time.Since(start)
	m.ProfileStack(nil)

	report, err := tl.Report(took)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("one token in %v\n%s", took, report)
}
