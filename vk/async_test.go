package vk

import (
	"math"
	"testing"
)

// TestStartWait fills a buffer from a submission that does not block, and
// reads it after Wait. Reading before Wait would prove nothing on a card that
// happens to be fast, so the check is only that the answer is complete once
// Wait has returned.
func TestStartWait(t *testing.T) {
	d := open(t)
	defer d.Close()
	b, err := d.Readback(4096, UsageStorage|UsageTransferDst)
	if err != nil {
		t.Fatal(err)
	}
	defer b.Close()
	if err := d.Start(func(r *Recorder) { r.Fill(b, math.Float32bits(1.5)) }); err != nil {
		t.Fatal(err)
	}
	if err := d.Wait(); err != nil {
		t.Fatal(err)
	}
	for i, x := range b.Floats() {
		if x != 1.5 {
			t.Fatalf("float %d = %v after Wait", i, x)
		}
	}
}
