package flux3

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"strings"
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
	if _, err := OpenVAE(d, vaePath(), 40000); err == nil || !strings.Contains(err.Error(), "4 GiB") {
		t.Errorf("OpenVAE with maxLatentPix=40000: %v, want an error naming 4 GiB", err)
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

func TestMergeGroupStats(t *testing.T) {
	part := func(xs []float64) [3]float64 {
		var mean, m2 float64
		for _, x := range xs {
			mean += x
		}
		mean /= float64(len(xs))
		for _, x := range xs {
			m2 += (x - mean) * (x - mean)
		}
		return [3]float64{float64(len(xs)), mean, m2}
	}
	got := mergeGroupStats([][3]float64{part([]float64{1, 2, 3}), {}, part([]float64{4, 5, 6, 7, 8, 9, 10})})
	for i, want := range []float64{10, 5.5, 82.5} {
		if math.Abs(got[i]-want) > 1e-9 {
			t.Errorf("merged %v, want n 10, mean 5.5, m2 82.5", got)
		}
	}
}

func TestVAEBands(t *testing.T) {
	bands, err := vaeBands(64, 64, 64*23)
	if err != nil {
		t.Fatal(err)
	}
	want := []vaeBand{{0, 19, 0, 21}, {19, 38, 17, 40}, {38, 57, 36, 59}, {57, 64, 55, 64}}
	if len(bands) != len(want) {
		t.Fatalf("bands %v, want %v", bands, want)
	}
	for i := range want {
		if bands[i] != want[i] {
			t.Errorf("band %d is %v, want %v", i, bands[i], want[i])
		}
	}
	if _, err := vaeBands(16, 1000, 4096); err == nil || !strings.Contains(err.Error(), "819") {
		t.Errorf("a latent 1000 wide: %v, want an error naming 819", err)
	}
}

// TestVAEBandedMatchesFull decodes the 1 MP acceptance latent whole and in
// bands of 20 rows (16 their own) and of 23 (19, the last band short).
func TestVAEBandedMatchesFull(t *testing.T) {
	f := loadFixtures(t)
	needFile(t, vaePath())
	latent := f.read(t, "accept/latent")
	d := device(t)

	full, err := OpenVAE(d, vaePath(), 64*64)
	if err != nil {
		t.Fatal(err)
	}
	want, err := full.Decode(latent, 64, 64)
	full.Close()
	if err != nil {
		t.Fatal(err)
	}
	for _, rows := range []int{20, 23} {
		v, err := openVAEBands(d, vaePath(), 64*64, 64*rows)
		if err != nil {
			t.Fatal(err)
		}
		got, err := v.Decode(latent, 64, 64)
		v.Close()
		if err != nil {
			t.Fatal(err)
		}
		p := calcPSNR(got, want)
		t.Logf("bands of %d latent rows vs a whole decode: PSNR %.2f dB", rows, p)
		if p < 40 {
			t.Errorf("bands of %d rows: PSNR %.2f dB < 40 dB", rows, p)
		}
	}
}

func TestVAEDecode4MPMatchesComfyUI(t *testing.T) {
	f := loadFixtures(t)
	needFile(t, vaePath())
	latent := f.read(t, "t2i4/latent")
	var comfy struct {
		Decode     float64 `json:"decode"`
		DecodeMode string  `json:"decode_mode"`
	}
	if b, err := os.ReadFile(filepath.Join(f.dir, "t2i4", "timings.json")); err == nil {
		json.Unmarshal(b, &comfy)
	}
	d := device(t)
	v, err := OpenVAE(d, vaePath(), 4456448/256)
	if err != nil {
		t.Fatal(err)
	}
	defer v.Close()
	t.Logf("VAE arena %d MB", vaeArenaFloats(4456448/256, vaeBandLatentPix, false)*4>>20)
	var rgb []float32
	for i := 0; i < 2; i++ {
		start := time.Now()
		if rgb, err = v.Decode(latent, 128, 128); err != nil {
			t.Fatal(err)
		}
		t.Logf("decode 2048x2048 run %d: %v (ComfyUI %.2f s, %s)", i+1, time.Since(start), comfy.Decode, comfy.DecodeMode)
	}
	// ComfyUI on this card decodes 2048 x 2048 tiled, the mean of three
	// tilings, which is itself some 31 dB from a whole decode: the whole
	// decode, recorded on the CPU in fp32, is what the bands are held to.
	full := calcPSNR(rgb, f.read(t, "t2i4/rgb_full"))
	t.Logf("PSNR vs a whole decode (CPU, fp32): %.2f dB", full)
	if full < 35 {
		t.Errorf("PSNR vs t2i4/rgb_full %.2f dB < 35 dB", full)
	}
	t.Logf("PSNR vs ComfyUI's own decode (%s): %.2f dB", comfy.DecodeMode, calcPSNR(rgb, f.read(t, "t2i4/rgb")))
}
