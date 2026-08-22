package resample

import (
	"math"
	"testing"
)

func TestMonoAveragesTheChannels(t *testing.T) {
	got := Mono([]float32{1, 3, -1, 1}, 2)
	want := []float32{2, 0}
	for i := range want {
		if math.Abs(float64(got[i]-want[i])) > 1e-6 {
			t.Fatalf("Mono = %v, want %v", got, want)
		}
	}
}

func TestSameRateIsACopy(t *testing.T) {
	in := []float32{1, 2, 3}
	out := To(in, 16000, 16000)
	if &out[0] == &in[0] {
		t.Fatal("To returned the caller's own slice; a resampler owns what it gives back")
	}
	for i := range in {
		if out[i] != in[i] {
			t.Fatalf("To at the same rate changed the signal: %v", out)
		}
	}
}

// A sine keeps its frequency across a rate change: the zero crossings of the
// result must fall where the mathematics puts them.
func TestSineKeepsItsFrequency(t *testing.T) {
	const (
		from = 44100
		to   = 16000
		freq = 440.0
		secs = 0.25
	)
	in := make([]float32, int(from*secs))
	for i := range in {
		in[i] = float32(math.Sin(2 * math.Pi * freq * float64(i) / from))
	}
	out := To(in, from, to)
	if want := int(to * secs); math.Abs(float64(len(out)-want)) > 2 {
		t.Fatalf("resampled to %d samples, want about %d", len(out), want)
	}
	// Skip the edges, where the kernel runs off the signal.
	var worst float64
	for i := 64; i < len(out)-64; i++ {
		want := math.Sin(2 * math.Pi * freq * float64(i) / to)
		if d := math.Abs(float64(out[i]) - want); d > worst {
			worst = d
		}
	}
	if worst > 0.01 {
		t.Fatalf("the resampled sine is %.4f away from the analytic one", worst)
	}
}

// TestDownsamplingLeavesNoAlias: a tone above the new Nyquist frequency must
// come back as near silence, not as the lower tone it would fold onto. This is
// the whole reason the kernel is stretched when the rate drops, and a
// resampler that forgets it still passes every other test in this file.
func TestDownsamplingLeavesNoAlias(t *testing.T) {
	const (
		from = 16000
		to   = 4000
		freq = 3000.0 // above 2 kHz, the new Nyquist: it must not survive
	)
	in := make([]float32, from/2)
	for i := range in {
		in[i] = float32(math.Sin(2 * math.Pi * freq * float64(i) / from))
	}
	out := To(in, from, to)
	var energy float64
	for _, v := range out[64 : len(out)-64] {
		energy += float64(v) * float64(v)
	}
	if rms := math.Sqrt(energy / float64(len(out)-128)); rms > 0.05 {
		t.Fatalf("a 3 kHz tone came through a 4 kHz resampling at %.4f; it aliased", rms)
	}
}

// TestAConstantStaysConstant: the weights are normalized, so a flat signal
// comes back flat rather than scaled by whatever the kernel happened to sum to.
func TestAConstantStaysConstant(t *testing.T) {
	in := make([]float32, 4000)
	for i := range in {
		in[i] = 0.5
	}
	out := To(in, 44100, 16000)
	for i := 64; i < len(out)-64; i++ {
		if math.Abs(float64(out[i]-0.5)) > 1e-3 {
			t.Fatalf("sample %d of a constant signal came back %f", i, out[i])
		}
	}
}
