package krea2

import (
	"math/rand"
	"os"
	"path/filepath"
	"strings"
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

// What a LoRA costs a step at 768 × 1024, and what putting one on costs: the
// one the fixtures were recorded with, and the files GOLEM_KREA2_LORAS names
// in the LoRA directory, separated by commas.
func TestLoRAStepTimings(t *testing.T) {
	heavy.Skip(t, "a measurement at full size")
	needFile(t, DiTPath())
	d := device(t)
	h, w := 128, 96
	r := rand.New(rand.NewSource(1))
	latent := make([]float32, 16*h*w)
	for i := range latent {
		latent[i] = float32(r.NormFloat64())
	}
	m, err := OpenDiT(d, DiTPath())
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	cond := make([]float32, 30*CondWidth)
	step := func(name string) {
		t.Helper()
		if err := m.SetText(0, cond, 30); err != nil {
			t.Fatal(err)
		}
		best := time.Hour
		for i := 0; i < 4; i++ {
			start := time.Now()
			if _, err := m.Step(0, latent, h, w, 0.5); err != nil {
				t.Fatal(err)
			}
			best = min(best, time.Since(start))
		}
		t.Logf("%s: a DiT step %v", name, best)
	}
	step("no LoRA")
	paths := []string{recordedLoRA(t)}
	for _, name := range strings.Split(os.Getenv("GOLEM_KREA2_LORAS"), ",") {
		if name != "" {
			paths = append(paths, filepath.Join(LoRADir(), name))
		}
	}
	for _, path := range paths {
		name := filepath.Base(path)
		start := time.Now()
		l, err := OpenLoRA(path)
		if err != nil {
			t.Fatal(err)
		}
		read := time.Since(start)
		start = time.Now()
		if err := m.SetLoRA(l, 1); err != nil {
			t.Logf("%s: %v", name, err)
			continue
		}
		t.Logf("%s: rank %d, %d MB, read in %v, put on the card in %v", name, l.Rank(), l.Bytes()>>20, read, time.Since(start))
		step(name)
		start = time.Now()
		if err := m.SetLoRA(l, 0.5); err != nil {
			t.Fatal(err)
		}
		t.Logf("%s: another strength in %v", name, time.Since(start))
		m.Trim()
	}
}
