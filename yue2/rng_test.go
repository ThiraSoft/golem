package yue2

import (
	"math"
	"testing"
)

func TestTorchRandn(t *testing.T) {
	f := loadFixtures(t)
	const seed = 9207606299741175932

	t.Run("normal_6400", func(t *testing.T) {
		want := f.read(t, "rng/normal_6400")
		got := make([]float32, len(want))
		TorchRandn(seed, got)
		for i := range want {
			if got[i] != want[i] {
				t.Fatalf("normal_6400 mismatch at index %d: got %v (%08x), want %v (%08x)",
					i, got[i], math.Float32bits(got[i]), want[i], math.Float32bits(want[i]))
			}
		}
	})

	t.Run("noise_100x64", func(t *testing.T) {
		want := f.read(t, "nar/noise")
		got := make([]float32, 100*64)
		TorchRandn(seed, got)
		for i := range want {
			if got[i] != want[i] {
				t.Fatalf("nar/noise mismatch at index %d: got %v (%08x), want %v (%08x)",
					i, got[i], math.Float32bits(got[i]), want[i], math.Float32bits(want[i]))
			}
		}
	})

	t.Run("normal_37", func(t *testing.T) {
		want := f.read(t, "rng/normal_37")
		got := make([]float32, len(want))
		TorchRandn(seed, got)
		for i := range want {
			if got[i] != want[i] {
				t.Fatalf("normal_37 mismatch at index %d: got %v (%08x), want %v (%08x)",
					i, got[i], math.Float32bits(got[i]), want[i], math.Float32bits(want[i]))
			}
		}
	})

	t.Run("tail_redraw", func(t *testing.T) {
		// As documented in task 6 brief, normal_37 matches normal_6400[:37] on the first
		// 16 elements (indices 0..15), while indices 16..36 differ because torch redraws
		// a non-multiple-of-16 tail.
		norm37 := f.read(t, "rng/normal_37")
		norm6400 := f.read(t, "rng/normal_6400")
		for i := 0; i < 16; i++ {
			if norm37[i] != norm6400[i] {
				t.Fatalf("expected norm37[%d] == norm6400[%d], got %v vs %v", i, i, norm37[i], norm6400[i])
			}
		}
		differ := false
		for i := 16; i < 37; i++ {
			if norm37[i] != norm6400[i] {
				differ = true
				break
			}
		}
		if !differ {
			t.Fatalf("expected norm37 to differ from norm6400 after index 16")
		}
	})
}
