package qwen35

import "testing"

// The same input twice must give the same output. A block keeps nothing
// between calls, and a scratch reused across patches is exactly where that
// quietly stops being true — a buffer left over from patch p read by p+1
// answers plausibly and drifts.
func TestVisionBlockIsAFunction(t *testing.T) {
	cfg, w := tinyTower()
	at := cfg.PatchPositions(2, 2)

	first := make([]float32, len(at)*cfg.Dim)
	for i := range first {
		first[i] = float32(i%13)*0.1 - 0.6
	}
	second := append([]float32(nil), first...)

	s1 := newVisionScratch(cfg, len(at))
	s2 := newVisionScratch(cfg, len(at))
	cfg.VisionBlockForward(w, 0, first, at, s1)
	cfg.VisionBlockForward(w, 0, second, at, s2)

	for i := range first {
		if first[i] != second[i] {
			t.Fatalf("element %d: %v against %v", i, first[i], second[i])
		}
	}
}

// A scratch reused across two calls must give what two fresh ones give.
func TestVisionBlockDoesNotCarryItsScratch(t *testing.T) {
	cfg, w := tinyTower()
	at := cfg.PatchPositions(2, 2)

	xs := make([]float32, len(at)*cfg.Dim)
	for i := range xs {
		xs[i] = float32(i%7)*0.2 - 0.5
	}
	reused := append([]float32(nil), xs...)
	fresh := append([]float32(nil), xs...)

	s := newVisionScratch(cfg, len(at))
	cfg.VisionBlockForward(w, 0, reused, at, s)
	cfg.VisionBlockForward(w, 0, reused, at, s)

	cfg.VisionBlockForward(w, 0, fresh, at, newVisionScratch(cfg, len(at)))
	cfg.VisionBlockForward(w, 0, fresh, at, newVisionScratch(cfg, len(at)))

	for i := range reused {
		if reused[i] != fresh[i] {
			t.Fatalf("element %d: reused %v against fresh %v", i, reused[i], fresh[i])
		}
	}
}
