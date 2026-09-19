package krea2

import (
	"math/rand"
	"testing"
	"time"

	"github.com/ThiraSoft/golem/internal/heavy"
)

// How long each network takes at the mobile front's size, 768 × 1024.
func TestFullSizeTimings(t *testing.T) {
	heavy.Skip(t, "a measurement at full size")
	needFile(t, VAEPath())
	d := device(t)
	h, w := 128, 96
	r := rand.New(rand.NewSource(1))
	latent := make([]float32, 16*h*w)
	for i := range latent {
		latent[i] = float32(r.NormFloat64())
	}
	v, err := OpenVAE(d, VAEPath(), h*w)
	if err != nil {
		t.Fatal(err)
	}
	start := time.Now()
	if _, err := v.Decode(latent, h, w); err != nil {
		t.Fatal(err)
	}
	t.Logf("VAE decode %v", time.Since(start))
	v.Close()

	m, err := OpenDiT(d, DiTPath())
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	cond := make([]float32, 30*CondWidth)
	if err := m.SetText(0, cond, 30); err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 2; i++ {
		start = time.Now()
		if _, err := m.Step(0, latent, h, w, 0.5); err != nil {
			t.Fatal(err)
		}
		t.Logf("DiT step %v", time.Since(start))
	}
}
