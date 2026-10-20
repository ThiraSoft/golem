package yue2

import (
	"fmt"
	"math"
	"math/rand"
	"path/filepath"
	"testing"
	"time"

	"github.com/ThiraSoft/golem/internal/heavy"
	"github.com/ThiraSoft/golem/vk"
)

func loadVAEWeights(t testing.TB) *VAEWeights {
	t.Helper()
	w, err := OpenVAE(filepath.Join(weightsDir(), "YuE2-Vae-F32.gguf"))
	if err != nil {
		t.Skipf("no VAE weights (%v): see ref/yue2/README.md", err)
	}
	return w
}

func openDevice(t testing.TB) *vk.Device {
	t.Helper()
	d, err := vk.Open()
	if err != nil {
		t.Skipf("no Vulkan device: %v", err)
	}
	t.Cleanup(d.Close)
	if !d.Coopmat() {
		t.Skip("the VAE wants cooperative matrices")
	}
	return d
}

// TestVAEWaypoints decodes the recording's 64 frames untiled and compares
// every step with yue2.cpp's, block 0 piece by piece.
//
// The bar holds against vae_f32acc, the same graph with ggml's products at
// GGML_PREC_F32: at its default precision ggml-vulkan accumulates fp16
// products in fp16 on this card, which puts yue2.cpp's own conv1 0.15% away
// from the exact sum of its fp16 operands, where golem's lands. The distance
// to the default recording, vae, is logged beside it.
func TestVAEWaypoints(t *testing.T) {
	fix := loadFixtures(t)
	w := loadVAEWeights(t)
	v, err := NewVAE(openDevice(t), w, 64)
	if err != nil {
		t.Fatal(err)
	}
	defer v.Close()
	got, err := v.waypoints(fix.read(t, "vae/latent"), 64)
	if err != nil {
		t.Fatal(err)
	}
	names := []string{"conv1", "b0_ct", "b0_ru0", "b0_ru1"}
	for i := range 6 {
		names = append(names, fmt.Sprintf("block_%d", i))
	}
	// The audio is time-major on the card; waypoints made it planar.
	names = append(names, "audio")
	for _, n := range names {
		t.Logf("%s: against fp16 accumulation %.4f%%", n, 100*rmsError(got[n], fix.read(t, "vae/"+n)))
		compareRMS(t, n, got[n], fix.read(t, "vae_f32acc/"+n), 2e-3)
	}
}

// TestVAETiled decodes 600 frames in tiles of 512 with a halo of 16, two
// tiles, against yue2.cpp's tiled decode of the same latents.
func TestVAETiled(t *testing.T) {
	fix := loadFixtures(t)
	w := loadVAEWeights(t)
	v, err := NewVAE(openDevice(t), w, 600)
	if err != nil {
		t.Fatal(err)
	}
	defer v.Close()
	t.Logf("arena %.2f GB", float64(v.ArenaBytes())/1e9)
	got, err := v.DecodeTiled(fix.read(t, "vae_tiled/latent"), 600, 512, 16)
	if err != nil {
		t.Fatal(err)
	}
	snr := snrDB(got, fix.read(t, "vae_tiled/audio"))
	t.Logf("SNR %.2f dB", snr)
	if !(snr >= 40) {
		t.Errorf("SNR %.2f dB, under 40", snr)
	}
}

// TestVAESnakeKernel checks the operand kernel, snake and dilated taps, in
// float32 against a loop.
func TestVAESnakeKernel(t *testing.T) {
	d := openDevice(t)
	const (
		ic, tl, taps, dil = 64, 100, 7, 3
		pad               = 3 * dil
	)
	rng := rand.New(rand.NewSource(42))
	in := make([]float32, tl*ic)
	for i := range in {
		in[i] = (rng.Float32() - 0.5) * 10
	}
	ea, ieb := make([]float32, ic), make([]float32, ic)
	for c := range ic {
		ea[c] = float32(math.Exp(float64(rng.Float32()-0.5) * 2))
		ieb[c] = float32(1 / math.Exp(float64(rng.Float32()-0.5)*2))
	}
	want := make([]float32, tl*ic*taps)
	for t := range tl {
		for c := range ic {
			for k := range taps {
				ts := t + k*dil - pad
				if ts < 0 || ts >= tl {
					continue
				}
				x := in[ts*ic+c]
				s := float32(math.Sin(float64(ea[c] * x)))
				want[t*ic*taps+c*taps+k] = x + s*s*ieb[c]
			}
		}
	}
	k, err := vk.NewK2(d, len(in)+len(want), append(append([]float32{}, ea...), ieb...))
	if err != nil {
		t.Fatal(err)
	}
	defer k.Close()
	// Something other than zero where the padding goes, to see it written.
	junk := make([]float32, len(want))
	for i := range junk {
		junk[i] = 7
	}
	if err := k.Write(len(in), junk); err != nil {
		t.Fatal(err)
	}
	if err := k.Write(0, in); err != nil {
		t.Fatal(err)
	}
	p, err := k.Compile(func(r *vk.Recorder) {
		k.Yue2Im2Col(r, vk.K2Yue2Im2Col{Src: 0, Dst: uint32(len(in)), DstStride: ic * taps, T: tl, IC: ic,
			K: taps, Dil: dil, Pad: pad, SnakeA: 0, SnakeB: ic})
	})
	if err != nil {
		t.Fatal(err)
	}
	defer p.Close()
	if err := p.Run(); err != nil {
		t.Fatal(err)
	}
	got, err := k.Read(len(in), len(want))
	if err != nil {
		t.Fatal(err)
	}
	compareRMS(t, "snake", got, want, 1e-6)
}

// BenchmarkVAEDecode decodes 4870 frames, the Guerreros song's length, of
// random latents. yue2.cpp takes 1.47 s for the same decode.
func BenchmarkVAEDecode(b *testing.B) {
	heavy.Skip(b, "a bench is a measurement, not a check")
	w := loadVAEWeights(b)
	v, err := NewVAE(openDevice(b), w, 4870)
	if err != nil {
		b.Fatal(err)
	}
	defer v.Close()
	const frames = 4870
	rng := rand.New(rand.NewSource(1))
	latents := make([]float32, 64*frames)
	for i := range latents {
		latents[i] = float32(rng.NormFloat64())
	}
	// The first decode records the programs.
	if _, err := v.Decode(latents, frames); err != nil {
		b.Fatal(err)
	}
	b.ResetTimer()
	start := time.Now()
	for range b.N {
		if _, err := v.Decode(latents, frames); err != nil {
			b.Fatal(err)
		}
	}
	ms := float64(time.Since(start).Milliseconds()) / float64(b.N)
	b.ReportMetric(ms, "ms/decode")
	b.Logf("%.0f ms a decode of %d frames, arena %.2f GB", ms, frames, float64(v.ArenaBytes())/1e9)
}
