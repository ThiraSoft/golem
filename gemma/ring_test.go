package gemma

import "testing"

// A batch that crosses the end of a window ring gives what the same tokens give
// one at a time. Bit for bit, as ForwardBatch promises.
//
// It did not. A window block's ring held exactly its window, and a pass stores
// every key before it scores any of them, so the last positions of a pass wrote
// over the oldest keys its first positions still attend to: those queries read
// keys from their own future. The window fixture is 841 positions against
// E2B's window of 512, so each width below crosses the ring. On the 26B, whose
// window is 1024, it was a village conversation of 1358 positions, and the
// model answered by copying its previous line.
func TestABatchAcrossTheRingAgreesWithOneAtATime(t *testing.T) {
	w := loadFixture(t, "window")
	ids := w.Tokens

	one := openEngine(t, 4096)
	if len(ids) <= one.Cfg.Blocks[0].WindowSize {
		t.Skipf("the fixture is %d positions, which does not cross a window of %d",
			len(ids), one.Cfg.Blocks[0].WindowSize)
	}
	var want []float32
	for pos, id := range ids {
		want = one.Forward(id, pos)
	}
	want = append([]float32(nil), want...)
	one.Close() // four engines at once is more memory than a test should hold

	// The whole prompt at once, the card's widest pass, and the server's
	// processor pass.
	for _, width := range []int{len(ids), 512, 32} {
		m := openEngine(t, 4096)
		var got []float32
		for at := 0; at < len(ids); at += width {
			to := min(at+width, len(ids))
			hs := m.ForwardBatch(ids[at:to], at)
			got = hs[len(hs)-1]
		}
		if !same(want, got) {
			t.Errorf("%d positions in passes of %d: the last state differs from one at a time", len(ids), width)
		}
		m.Close()
	}
}
