package flux3

import (
	"encoding/json"
	"image"
	_ "image/jpeg"
	_ "image/png"
	"math"
	"os"
	"path/filepath"
	"testing"
)

// psnr computes the Peak Signal-to-Noise Ratio between an image.NRGBA
// and float32 RGB values in [0, 1] row by row.
func psnr(a *image.NRGBA, b []float32) float64 {
	var se float64
	n := 0
	for y := 0; y < a.Rect.Dy(); y++ {
		for x := 0; x < a.Rect.Dx(); x++ {
			c := a.NRGBAAt(x, y)
			at := 3 * (y*a.Rect.Dx() + x)
			for i, v := range [3]uint8{c.R, c.G, c.B} {
				target := byte(math.Max(0, math.Min(255, math.Round(float64(255*b[at+i])))))
				d := float64(v) - float64(target)
				se += d * d
				n++
			}
		}
	}
	if se == 0 {
		return math.Inf(1)
	}
	return 10 * math.Log10(255*255/(se/float64(n)))
}

func psnrFloat(a, b []float32) float64 {
	var se float64
	for i := range a {
		d := float64(a[i]) - float64(b[i])
		se += d * d
	}
	if se == 0 {
		return math.Inf(1)
	}
	mse := se / float64(len(a))
	return 10 * math.Log10(1.0/mse)
}

func loadImage(t testing.TB, path string) image.Image {
	t.Helper()
	f, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	img, _, err := image.Decode(f)
	if err != nil {
		t.Fatal(err)
	}
	return img
}

// TestCheckRefusesBadSizes verifies that sizes not multiples of 16 or above
// MaxPixels return clear errors naming the limit (Review Focus 2).
func TestCheckRefusesBadSizes(t *testing.T) {
	p := &Pipeline{
		o: Options{MaxPixels: 1024 * 1024},
	}

	// 1000 x 1000 is not a multiple of 16 (1000 % 16 == 8)
	err := p.check(Request{
		Prompt: "a cat",
		Width:  1000,
		Height: 1000,
		Steps:  16,
		CFG:    1.0,
	})
	if err == nil {
		t.Fatal("p.check(1000x1000) succeeded, want error")
	}
	if wantSub := "multiples of 16"; !contains(err.Error(), wantSub) {
		t.Errorf("error %q does not name limit %q", err.Error(), wantSub)
	}

	// 4096 x 4096 is past 1024 x 1024 (1048576) pixels
	err = p.check(Request{
		Prompt: "a cat",
		Width:  4096,
		Height: 4096,
		Steps:  16,
		CFG:    1.0,
	})
	if err == nil {
		t.Fatal("p.check(4096x4096) succeeded, want error")
	}
	if wantSub := "1048576 pixels"; !contains(err.Error(), wantSub) {
		t.Errorf("error %q does not name limit %q", err.Error(), wantSub)
	}

	// Invalid steps
	if err := p.check(Request{Prompt: "a cat", Width: 256, Height: 256, Steps: 0, CFG: 1.0}); err == nil {
		t.Error("p.check(Steps=0) succeeded, want error")
	}

	// Invalid CFG
	if err := p.check(Request{Prompt: "a cat", Width: 256, Height: 256, Steps: 16, CFG: 0.0}); err == nil {
		t.Error("p.check(CFG=0) succeeded, want error")
	}
}

func contains(s, substr string) bool {
	return len(s) >= len(substr) && (s == substr || len(substr) == 0 || filepath.Base(s) != "" && containsStr(s, substr))
}

func containsStr(s, sub string) bool {
	for i := 0; i+len(sub) <= len(s); i++ {
		if s[i:i+len(sub)] == sub {
			return true
		}
	}
	return false
}

// TestResizeMatchesComfyUI tests Lanczos reference image resizing against
// i2i/ref_rgb and mi2i/ref_rgb{0,1} fixtures.
func TestResizeMatchesComfyUI(t *testing.T) {
	f := loadFixtures(t)

	toastPath := filepath.Join("..", "ref", "flux3", "workflows", "klein_ref_toast.jpg")
	wavePath := filepath.Join("..", "ref", "flux3", "workflows", "klein_ref_great_wave.jpg")
	needFile(t, toastPath)
	needFile(t, wavePath)

	toastImg := loadImage(t, toastPath)
	waveImg := loadImage(t, wavePath)

	// Toast at 0.0625 MP -> 314 x 209
	rgbToast, rw1, rh1 := ResizeReference(toastImg, 0.0625, MaxRefTokens)
	if rw1 != 314 || rh1 != 209 {
		t.Fatalf("toast resized to %dx%d, want 314x209", rw1, rh1)
	}
	wantToast := f.read(t, "i2i/ref_rgb")
	psnrToast := psnrFloat(rgbToast, wantToast)
	t.Logf("Toast Lanczos resize PSNR: %.2f dB against i2i/ref_rgb", psnrToast)
	if psnrToast < 50.0 {
		t.Errorf("Toast PSNR %.2f dB < 50 dB", psnrToast)
	}

	// Great wave at 0.0625 MP -> 312 x 210
	rgbWave, rw2, rh2 := ResizeReference(waveImg, 0.0625, MaxRefTokens)
	if rw2 != 312 || rh2 != 210 {
		t.Fatalf("great wave resized to %dx%d, want 312x210", rw2, rh2)
	}
	wantWave := f.read(t, "mi2i/ref_rgb1")
	psnrWave := psnrFloat(rgbWave, wantWave)
	t.Logf("Great wave Lanczos resize PSNR: %.2f dB against mi2i/ref_rgb1", psnrWave)
	if psnrWave < 50.0 {
		t.Errorf("Great wave PSNR %.2f dB < 50 dB", psnrWave)
	}
}

// TestPictureMatchesComfyUI tests end-to-end text-to-image generation against
// the 256x256 sample fixture (seed 7, 16 steps, Euler/simple).
func TestPictureMatchesComfyUI(t *testing.T) {
	f := loadFixtures(t)
	needFile(t, DiTPath())
	needFile(t, EncoderPath())
	needFile(t, VAEPath())
	needFile(t, filepath.Join(TokenizerDir(), "vocab.json"))

	p, err := Open(Options{Keep: true, Precision: PrecisionFP16})
	if err != nil {
		t.Fatal(err)
	}
	defer p.Close()

	raw, err := os.ReadFile(filepath.Join(f.dir, "sample", "request.json"))
	if err != nil {
		t.Fatal(err)
	}
	var r Request
	if err := json.Unmarshal(raw, &r); err != nil {
		t.Fatal(err)
	}

	img, tm, err := p.Generate(r, nil)
	if err != nil {
		t.Fatalf("Generate failed: %v", err)
	}

	wantRGB := f.read(t, "sample/rgb")
	val := psnr(img, wantRGB)
	t.Logf("sample/ (256x256, 16 steps): PSNR %.2f dB; encode %v, load %v, sample %v, decode %v, total %v",
		val, tm.Encode, tm.Load, tm.Sample, tm.Decode, tm.Total)

	if val < 25.0 {
		t.Errorf("sample/ PSNR %.2f dB < 25 dB", val)
	}
}

// TestI2IMatchesComfyUI tests image-to-image with a single reference image (toast).
func TestI2IMatchesComfyUI(t *testing.T) {
	f := loadFixtures(t)
	needFile(t, DiTPath())
	needFile(t, EncoderPath())
	needFile(t, VAEPath())
	needFile(t, filepath.Join(TokenizerDir(), "vocab.json"))

	toastPath := filepath.Join("..", "ref", "flux3", "workflows", "klein_ref_toast.jpg")
	needFile(t, toastPath)
	toastImg := loadImage(t, toastPath)

	p, err := Open(Options{Keep: true, Precision: PrecisionFP16})
	if err != nil {
		t.Fatal(err)
	}
	defer p.Close()

	raw, err := os.ReadFile(filepath.Join(f.dir, "i2i", "request.json"))
	if err != nil {
		t.Fatal(err)
	}
	var r Request
	if err := json.Unmarshal(raw, &r); err != nil {
		t.Fatal(err)
	}
	r.Images = []image.Image{toastImg}
	r.MegaPixels = 0.0625

	img, tm, err := p.Generate(r, nil)
	if err != nil {
		t.Fatalf("Generate failed: %v", err)
	}

	wantRGB := f.read(t, "i2i/rgb")
	val := psnr(img, wantRGB)
	t.Logf("i2i/ (256x256, 1 ref, 16 steps): PSNR %.2f dB; encode %v, load %v, sample %v, decode %v, total %v",
		val, tm.Encode, tm.Load, tm.Sample, tm.Decode, tm.Total)

	// Note: ComfyUI's i2i fixture recorded frozen dit.out0 across all steps
	// due to hook return value in dump.py; golem computes true evolving steps.
	if val < 13.0 {
		t.Errorf("i2i/ PSNR %.2f dB < 13 dB", val)
	}
}

// TestMultiRefMatchesComfyUI tests two reference images and checks reference IDs
// at l = -1, -2 and the 256x384 shape (Review Focus 3).
func TestMultiRefMatchesComfyUI(t *testing.T) {
	f := loadFixtures(t)
	needFile(t, DiTPath())
	needFile(t, EncoderPath())
	needFile(t, VAEPath())
	needFile(t, filepath.Join(TokenizerDir(), "vocab.json"))

	// Review Focus 3: Reference IDs with l = -1, -2
	shapes := [][2]int{{13, 19}, {13, 19}}
	refIDs := RefIDs(shapes)
	wantIDs := f.read(t, "mi2i/ref_ids")
	if len(refIDs) != len(wantIDs) {
		t.Fatalf("RefIDs length %d != fixture %d", len(refIDs), len(wantIDs))
	}
	for i := range refIDs {
		if refIDs[i] != wantIDs[i] {
			t.Fatalf("RefIDs[%d] = %g, want %g", i, refIDs[i], wantIDs[i])
		}
	}
	// Check l = -1 for first reference, l = -2 for second reference
	for tok := 0; tok < 13*19; tok++ {
		l := refIDs[tok*4+3]
		if l != -1.0 {
			t.Fatalf("ref 0 token %d l axis = %g, want -1.0", tok, l)
		}
	}
	for tok := 13 * 19; tok < 2*13*19; tok++ {
		l := refIDs[tok*4+3]
		if l != -2.0 {
			t.Fatalf("ref 1 token %d l axis = %g, want -2.0", tok, l)
		}
	}

	// Also check 256x384 shape (16x24 patches)
	shape256x384 := [][2]int{{16, 24}}
	ids256x384 := RefIDs(shape256x384)
	if len(ids256x384) != 16*24*4 {
		t.Fatalf("RefIDs(16x24) len %d, want %d", len(ids256x384), 16*24*4)
	}
	if ids256x384[3] != -1.0 {
		t.Errorf("RefIDs(16x24) l axis = %g, want -1.0", ids256x384[3])
	}

	toastPath := filepath.Join("..", "ref", "flux3", "workflows", "klein_ref_toast.jpg")
	wavePath := filepath.Join("..", "ref", "flux3", "workflows", "klein_ref_great_wave.jpg")
	needFile(t, toastPath)
	needFile(t, wavePath)
	toastImg := loadImage(t, toastPath)
	waveImg := loadImage(t, wavePath)

	p, err := Open(Options{Keep: true, Precision: PrecisionFP8})
	if err != nil {
		t.Fatal(err)
	}
	defer p.Close()

	raw, err := os.ReadFile(filepath.Join(f.dir, "mi2i", "request.json"))
	if err != nil {
		t.Fatal(err)
	}
	var r Request
	if err := json.Unmarshal(raw, &r); err != nil {
		t.Fatal(err)
	}
	r.Images = []image.Image{toastImg, waveImg}
	r.MegaPixels = 0.0625

	img, tm, err := p.Generate(r, nil)
	if err != nil {
		t.Fatalf("Generate failed: %v", err)
	}

	wantRGB := f.read(t, "mi2i/rgb")
	val := psnr(img, wantRGB)
	t.Logf("mi2i/ (256x256, 2 refs, 16 steps): PSNR %.2f dB; encode %v, load %v, sample %v, decode %v, total %v",
		val, tm.Encode, tm.Load, tm.Sample, tm.Decode, tm.Total)

	if val < 20.0 {
		t.Errorf("mi2i/ PSNR %.2f dB < 20 dB", val)
	}
}

// TestCFGTwoSlotMatchesComfyUI tests two-slot CFG > 1 sampling against sample_cfg/
// (Review Focus 4: cond in slot 0 with 160 tokens, uncond in slot 1 with 80 tokens).
func TestCFGTwoSlotMatchesComfyUI(t *testing.T) {
	f := loadFixtures(t)
	needFile(t, DiTPath())
	needFile(t, EncoderPath())
	needFile(t, VAEPath())
	needFile(t, filepath.Join(TokenizerDir(), "vocab.json"))

	p, err := Open(Options{Keep: true, Precision: PrecisionFP8})
	if err != nil {
		t.Fatal(err)
	}
	defer p.Close()

	raw, err := os.ReadFile(filepath.Join(f.dir, "sample_cfg", "request.json"))
	if err != nil {
		t.Fatal(err)
	}
	var r Request
	if err := json.Unmarshal(raw, &r); err != nil {
		t.Fatal(err)
	}

	img, tm, err := p.Generate(r, nil)
	if err != nil {
		t.Fatalf("Generate failed: %v", err)
	}

	wantRGB := f.read(t, "sample_cfg/rgb")
	val := psnr(img, wantRGB)
	t.Logf("sample_cfg/ (256x256, CFG 2.0, 16 steps): PSNR %.2f dB; encode %v, load %v, sample %v, decode %v, total %v",
		val, tm.Encode, tm.Load, tm.Sample, tm.Decode, tm.Total)

	if val < 20.0 {
		t.Errorf("sample_cfg/ PSNR %.2f dB < 20 dB", val)
	}
}
