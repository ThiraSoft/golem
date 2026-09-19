package krea2

import (
	"encoding/json"
	"image"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/ThiraSoft/golem/internal/heavy"
)

// psnr is the peak signal-to-noise ratio of two pictures of the same size,
// over the three colours, in dB.
func psnr(a *image.NRGBA, b []float32) float64 {
	var se float64
	n := 0
	for y := 0; y < a.Rect.Dy(); y++ {
		for x := 0; x < a.Rect.Dx(); x++ {
			c := a.NRGBAAt(x, y)
			at := 3 * (y*a.Rect.Dx() + x)
			for i, v := range [3]uint8{c.R, c.G, c.B} {
				d := float64(v) - float64(byte(255*b[at+i]))
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

// The whole of it against a picture ComfyUI drew: the same prompt, the same
// seed, the same sampler, from the three files.
//
// ComfyUI's DiT runs in bf16 on ROCm's products, which on one call is 1.8%
// (rms) from the arithmetic and, by the middle of the schedule, 6%; this is
// 0.7% and 1.5%. Its rounding is not torch's own on the CPU either, so it is
// not something to imitate, and the pictures part by what it does: at the
// front's 768 × 1024 they are the same to the eye, above 30 dB; at 256 × 256,
// a size the model was not made for, the same cat with other whiskers.
func TestPictureMatchesComfyUI(t *testing.T) {
	heavy.Skip(t, "draws pictures with the twelve-gigabyte DiT")
	f := loadFixtures(t)
	needFile(t, DiTPath())
	p, err := Open(Options{Keep: true})
	if err != nil {
		t.Fatal(err)
	}
	defer p.Close()
	for _, run := range []struct {
		tag string
		min float64
	}{{"sample", 20}, {"accept", 28}} {
		raw, err := os.ReadFile(filepath.Join(f.dir, run.tag, "request.json"))
		if err != nil {
			t.Logf("%s: not recorded", run.tag)
			continue
		}
		var r Request
		if err := json.Unmarshal(raw, &r); err != nil {
			t.Fatal(err)
		}
		img, tm, err := p.Generate(r, nil)
		if err != nil {
			t.Fatal(err)
		}
		want := f.read(t, run.tag+"/image")
		got := psnr(img, want)
		t.Logf("%s: %d × %d, %d steps: PSNR %.2f dB; encode %v, load %v, sample %v, decode %v", run.tag, r.Width, r.Height, r.Steps,
			got, tm.Encode, tm.Load, tm.Sample, tm.Decode)
		if out := os.Getenv("GOLEM_KREA2_OUT"); out != "" {
			writePNG(t, filepath.Join(out, run.tag+".png"), img)
		}
		if got < run.min {
			t.Errorf("%s: PSNR %.2f dB against ComfyUI's picture, want %.0f", run.tag, got, run.min)
		}
	}
}
