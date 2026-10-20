package flux3

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"image"
	"image/color"
	_ "image/jpeg"
	_ "image/png"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/ThiraSoft/golem/token/bytebpe"
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
	if wantSub := "multiples of 16"; !strings.Contains(err.Error(), wantSub) {
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
	if wantSub := "1048576 pixels"; !strings.Contains(err.Error(), wantSub) {
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
	if err := p.check(Request{Prompt: "a cat", Width: 256, Height: 256, Steps: 16, CFG: -1.0}); err == nil {
		t.Error("p.check(CFG=-1) succeeded, want error")
	}
}

func TestPromptOverLimitFailsBeforeDiTOpen(t *testing.T) {
	tokDir := TokenizerDir()
	needFile(t, filepath.Join(tokDir, "vocab.json"))
	tok, err := bytebpe.LoadFiles(tokDir)
	if err != nil {
		t.Fatal(err)
	}
	p := &Pipeline{
		tok: tok,
		o:   Options{MaxPixels: 1024 * 1024},
	}
	longPrompt := strings.Repeat("unbelievable extraordinary fantastic magnificent ", 250)
	_, _, err = p.Generate(Request{
		Prompt: longPrompt,
		Width:  256,
		Height: 256,
		Steps:  16,
		CFG:    1.0,
	}, nil)
	if err == nil {
		t.Fatal("Generate with long prompt succeeded, want error")
	}
	if !strings.Contains(err.Error(), "960") {
		t.Errorf("error %q does not name limit 960", err.Error())
	}
	if p.dit != nil {
		t.Errorf("p.dit was opened (%v), want nil", p.dit)
	}
}

func TestOpenRefusesExcessMaxPixels(t *testing.T) {
	_, err := Open(Options{
		MaxPixels: 4096 * 4096,
	})
	if err == nil {
		t.Fatal("Open with MaxPixels past DefaultMaxPixels succeeded, want error")
	}
	if !strings.Contains(err.Error(), "4456448") {
		t.Errorf("error %q does not name the 4456448 pixel limit", err.Error())
	}
}

// TestPNGCanHideThePrompt asserts that when HidePrompt is true, neither the prompt
// nor the negative prompt appears in the PNG parameters, JSON chunk, or any chunk.
func TestPNGCanHideThePrompt(t *testing.T) {
	img := image.NewNRGBA(image.Rect(0, 0, 64, 64))
	r := Request{
		Prompt:     "secret-super-confidential-prompt",
		Negative:   "secret-negative-prompt",
		HidePrompt: true,
		Width:      64,
		Height:     64,
		Steps:      16,
		CFG:        1.0,
		Seed:       42,
	}
	var buf bytes.Buffer
	if err := WritePNG(&buf, img, r, "test-model"); err != nil {
		t.Fatalf("WritePNG failed: %v", err)
	}
	data := buf.Bytes()
	if bytes.Contains(data, []byte("secret-super-confidential-prompt")) {
		t.Errorf("PNG data contains prompt despite HidePrompt")
	}
	if bytes.Contains(data, []byte("secret-negative-prompt")) {
		t.Errorf("PNG data contains negative prompt despite HidePrompt")
	}

	const sig = 8
	if len(data) < sig || string(data[:sig]) != "\x89PNG\r\n\x1a\n" {
		t.Fatalf("invalid PNG signature")
	}
	offset := sig
	for offset+8 <= len(data) {
		length := int(binary.BigEndian.Uint32(data[offset : offset+4]))
		chunkType := string(data[offset+4 : offset+8])
		offset += 8
		if offset+length+4 > len(data) {
			t.Fatalf("chunk %s overflows PNG data", chunkType)
		}
		chunkData := data[offset : offset+length]
		offset += length + 4
		if bytes.Contains(chunkData, []byte("secret-super-confidential-prompt")) {
			t.Errorf("chunk %s contains prompt", chunkType)
		}
		if bytes.Contains(chunkData, []byte("secret-negative-prompt")) {
			t.Errorf("chunk %s contains negative prompt", chunkType)
		}
	}
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

func TestResizeReferenceAlphaPreservesStraightRGB(t *testing.T) {
	// 32x32 image with transparent red (R=255, G=0, B=0, A=0)
	nrgba := image.NewNRGBA(image.Rect(0, 0, 32, 32))
	for y := 0; y < 32; y++ {
		for x := 0; x < 32; x++ {
			nrgba.SetNRGBA(x, y, color.NRGBA{R: 255, G: 0, B: 0, A: 0})
		}
	}
	rgb, rw, rh := ResizeReference(nrgba, 0.001, MaxRefTokens)
	if rw <= 0 || rh <= 0 {
		t.Fatalf("invalid resized dimensions: %dx%d", rw, rh)
	}
	// Straight RGB preserves R=1.0 despite A=0
	if rgb[0] < 0.95 {
		t.Errorf("R channel is %f, want ~1.0 (straight RGB preserved)", rgb[0])
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

	if val < 25.0 {
		t.Errorf("i2i/ PSNR %.2f dB < 25 dB", val)
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

	toastPath := filepath.Join("..", "ref", "flux3", "workflows", "klein_ref_toast.jpg")
	wavePath := filepath.Join("..", "ref", "flux3", "workflows", "klein_ref_great_wave.jpg")
	needFile(t, toastPath)
	needFile(t, wavePath)
	toastImg := loadImage(t, toastPath)
	waveImg := loadImage(t, wavePath)

	// Review Focus 3: Reference IDs with l = -1, -2, shapes computed from ResizeReference
	_, rw1, rh1 := ResizeReference(toastImg, 0.0625, MaxRefTokens)
	_, rw2, rh2 := ResizeReference(waveImg, 0.0625, MaxRefTokens)
	shapes := [][2]int{{rh1 / 16, rw1 / 16}, {rh2 / 16, rw2 / 16}}
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
	n0 := shapes[0][0] * shapes[0][1]
	n1 := shapes[1][0] * shapes[1][1]
	for tok := 0; tok < n0; tok++ {
		l := refIDs[tok*4+3]
		if l != -1.0 {
			t.Fatalf("ref 0 token %d l axis = %g, want -1.0", tok, l)
		}
	}
	for tok := n0; tok < n0+n1; tok++ {
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

	if val < 25.0 {
		t.Errorf("mi2i/ PSNR %.2f dB < 25 dB", val)
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

	var stepLatents [][]float32
	p.saveLatent = true
	p.onStep = func(step int, x []float32) {
		stepLatents = append(stepLatents, append([]float32(nil), x...))
	}

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

	// Slot check: slot 0 ran with 160 tokens, slot 1 with 80 tokens
	if p.lastTokens[0] != 160 || p.lastTokens[1] != 80 {
		t.Fatalf("slot tokens: slot 0 has %d, slot 1 has %d, want 160 and 80",
			p.lastTokens[0], p.lastTokens[1])
	}
	t.Logf("slot tokens confirmed: slot 0 = %d tokens, slot 1 = %d tokens",
		p.lastTokens[0], p.lastTokens[1])

	// Compare per-step latents x_n against sample_cfg/x_n
	var worstRMS float64
	var worstStep int
	for step := 0; step < len(stepLatents); step++ {
		key := fmt.Sprintf("sample_cfg/x_%d", step)
		if !f.has(key) {
			continue
		}
		wantX := f.read(t, key)
		e := rmsError(stepLatents[step], wantX)
		t.Logf("sample_cfg/x_%d: rms error %.4f%%", step, e*100)
		if e > worstRMS {
			worstRMS = e
			worstStep = step
		}
	}
	t.Logf("worst per-step latent rms error: %.4f%% at step %d", worstRMS*100, worstStep)
	// In CFG mode (scale 2.0), errors accumulate across 16 steps with two DiT passes
	// per step. Intermediate latents x_step drift up to ~7% RMS against the bf16 reference;
	// 8.0% provides a tight bound that catches algorithmic regressions.
	if worstRMS > 0.08 {
		t.Errorf("worst per-step latent rms %.4f%% > 8.0%%", worstRMS*100)
	}

	// Compare final latent against sample_cfg/latent
	// The final unscaled latent differences accumulate further after 16 extrapolation steps
	// (denoised = uncond + cfg * (cond - uncond)) reaching ~18-20% RMS; 22.0% bounds this
	// while the decoded RGB image still achieves >26 dB PSNR.
	wantLatent := f.read(t, "sample_cfg/latent")
	latErr := rmsError(p.lastLatent, wantLatent)
	t.Logf("sample_cfg/latent: rms error %.4f%%", latErr*100)
	if latErr > 0.22 {
		t.Errorf("final latent rms %.4f%% > 22.0%%", latErr*100)
	}

	wantRGB := f.read(t, "sample_cfg/rgb")
	val := psnr(img, wantRGB)
	t.Logf("sample_cfg/ (256x256, CFG 2.0, 16 steps): PSNR %.2f dB; encode %v, load %v, sample %v, decode %v, total %v",
		val, tm.Encode, tm.Load, tm.Sample, tm.Decode, tm.Total)

	if val < 25.0 {
		t.Errorf("sample_cfg/ PSNR %.2f dB < 25 dB", val)
	}
}
