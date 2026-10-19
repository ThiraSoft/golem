package flux3

import (
	"math"
	"os"
	"testing"
	"time"
)

const defaultVAEPath = "/mnt/data/dev/ComfyUI/models/vae/flux2-vae.safetensors"

func vaePath() string {
	if p := os.Getenv("FLUX3_VAE"); p != "" {
		return p
	}
	return defaultVAEPath
}

func calcPSNR(a, b []float32) float64 {
	if len(a) != len(b) || len(a) == 0 {
		return 0
	}
	var mse float64
	for i := range a {
		d := float64(a[i] - b[i])
		mse += d * d
	}
	mse /= float64(len(a))
	if mse <= 1e-15 {
		return 100.0
	}
	return -10 * math.Log10(mse)
}

func TestVAEDecodeMatchesComfyUI(t *testing.T) {
	f := loadFixtures(t)
	if _, err := os.Stat(vaePath()); err != nil {
		t.Skipf("VAE file not found at %s: %v", vaePath(), err)
	}

	latent := f.read(t, "sample/latent")
	wantRGB := f.read(t, "sample/rgb")
	wantUp0 := f.read(t, "vae/up_0")
	wantUp1 := f.read(t, "vae/up_1")
	wantUp2 := f.read(t, "vae/up_2")
	wantUp3 := f.read(t, "vae/up_3")

	d := device(t)
	v, err := openVAEForTest(d, vaePath(), 16*16)
	if err != nil {
		t.Fatalf("OpenVAE: %v", err)
	}
	defer v.Close()

	rgb, wps, err := v.decodeWithWaypoints(latent, 16, 16)
	if err != nil {
		t.Fatalf("decodeWithWaypoints: %v", err)
	}

	// Compare per-block waypoints
	compareRMS(t, "vae/up_3", wps["vae/up_3"], wantUp3, 0.02)
	compareRMS(t, "vae/up_2", wps["vae/up_2"], wantUp2, 0.02)
	compareRMS(t, "vae/up_1", wps["vae/up_1"], wantUp1, 0.02)
	compareRMS(t, "vae/up_0", wps["vae/up_0"], wantUp0, 0.02)

	// Final RGB PSNR >= 40 dB
	p := calcPSNR(rgb, wantRGB)
	t.Logf("Decode PSNR: %.2f dB (want >= 40 dB)", p)
	if p < 40.0 {
		t.Errorf("Decode PSNR %.2f dB < 40 dB", p)
	}
}

func TestVAEEncodeMatchesComfyUI(t *testing.T) {
	f := loadFixtures(t)
	if _, err := os.Stat(vaePath()); err != nil {
		t.Skipf("VAE file not found at %s: %v", vaePath(), err)
	}

	encIn := f.read(t, "vae/enc.in")
	wantLatent := f.read(t, "vae/enc.latent")

	d := device(t)
	v, err := OpenVAE(d, vaePath(), 16*16)
	if err != nil {
		t.Fatalf("OpenVAE: %v", err)
	}
	defer v.Close()

	gotLatent, err := v.Encode(encIn, 209, 314)
	if err != nil {
		t.Fatalf("Encode: %v", err)
	}

	// enc.latent within 1% rms
	compareRMS(t, "vae/enc.latent", gotLatent, wantLatent, 0.01)
}

func TestVAEDecode1024MatchesComfyUI(t *testing.T) {
	f := loadFixtures(t)
	if _, err := os.Stat(vaePath()); err != nil {
		t.Skipf("VAE file not found at %s: %v", vaePath(), err)
	}

	latent := f.read(t, "accept/latent")
	wantRGB := f.read(t, "accept/rgb")

	d := device(t)
	v, err := OpenVAE(d, vaePath(), 64*64)
	if err != nil {
		t.Fatalf("OpenVAE: %v", err)
	}
	defer v.Close()

	// Warmup and measure decode time
	start := time.Now()
	rgb, err := v.Decode(latent, 64, 64)
	if err != nil {
		t.Fatalf("Decode: %v", err)
	}
	dur := time.Since(start)
	t.Logf("Decode 1024x1024 first run: %v", dur)

	// Measure second run (steady state)
	start = time.Now()
	rgb, err = v.Decode(latent, 64, 64)
	if err != nil {
		t.Fatalf("Decode: %v", err)
	}
	dur = time.Since(start)
	t.Logf("Decode 1024x1024 steady-state: %v", dur)

	p := calcPSNR(rgb, wantRGB)
	t.Logf("Decode 1024x1024 PSNR: %.2f dB (want >= 40 dB)", p)
	if p < 40.0 {
		t.Errorf("Decode 1024x1024 PSNR %.2f dB < 40 dB", p)
	}

	// Benchmark Encode for a 1 MP reference (1024 x 1024 RGB)
	oneMP := make([]float32, 3*1024*1024)
	copy(oneMP, wantRGB) // real 1 MP image
	start = time.Now()
	_, err = v.Encode(oneMP, 1024, 1024)
	if err != nil {
		t.Fatalf("Encode: %v", err)
	}
	dur = time.Since(start)
	t.Logf("Encode 1 MP reference first run: %v", dur)

	start = time.Now()
	_, err = v.Encode(oneMP, 1024, 1024)
	if err != nil {
		t.Fatalf("Encode: %v", err)
	}
	dur = time.Since(start)
	t.Logf("Encode 1 MP reference steady-state: %v", dur)
}

func TestVAEBoundsAndErrors(t *testing.T) {
	if _, err := os.Stat(vaePath()); err != nil {
		t.Skipf("VAE file not found at %s: %v", vaePath(), err)
	}
	d := device(t)

	// maxLatentPix <= 0 rejected
	if _, err := OpenVAE(d, vaePath(), 0); err == nil {
		t.Errorf("expected error opening VAE with maxLatentPix=0")
	}
	if _, err := OpenVAE(d, vaePath(), -1); err == nil {
		t.Errorf("expected error opening VAE with maxLatentPix=-1")
	}

	// maxLatentPix beyond arena capacity rejected
	if _, err := OpenVAE(d, vaePath(), 7000); err == nil {
		t.Errorf("expected error opening VAE with maxLatentPix=7000 (> 6553)")
	}

	// Product OpenVAE does not allocate waypoint buffers; decodeWithWaypoints must fail
	prodVAE, err := OpenVAE(d, vaePath(), 16*16)
	if err != nil {
		t.Fatalf("OpenVAE: %v", err)
	}
	defer prodVAE.Close()

	dummyLatent := make([]float32, 128*16*16)
	if _, _, err := prodVAE.decodeWithWaypoints(dummyLatent, 16, 16); err == nil {
		t.Errorf("expected error calling decodeWithWaypoints on product VAE")
	}

	// Decode with h*w > maxLatentPix fails
	if _, err := prodVAE.Decode(make([]float32, 128*17*16), 17, 16); err == nil {
		t.Errorf("expected error calling Decode with 17*16 > 16*16")
	}
}
