package flux3

import (
	"encoding/json"
	"image"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"
	"time"
)

func TestScaleToTotalPixelsSizes(t *testing.T) {
	for _, c := range []struct{ w, h, ww, wh int }{
		{1000, 1000, 2048, 2048},
		{1000, 1500, 1680, 2512},
		{900, 1200, 1776, 2368},
		{900, 1600, 1536, 2736},
		{1500, 1000, 2512, 1680},
		{1200, 900, 2368, 1776},
		{1600, 900, 2736, 1536},
		{2100, 900, 3136, 1344},
	} {
		got := ScaleToTotalPixels(image.NewRGBA(image.Rect(0, 0, c.w, c.h)), 4.0, 16).Bounds()
		if got.Dx() != c.ww || got.Dy() != c.wh {
			t.Errorf("%dx%d at 4 MP: %dx%d, want %dx%d", c.w, c.h, got.Dx(), got.Dy(), c.ww, c.wh)
		}
	}
}

func decodeFile(t testing.TB, path string) image.Image {
	t.Helper()
	f, err := os.Open(path)
	if err != nil {
		t.Skipf("%s: %v", path, err)
	}
	defer f.Close()
	img, err := DecodeImage(f)
	if err != nil {
		t.Fatal(err)
	}
	return img
}

func TestMatchImageUsesOrientedSize(t *testing.T) {
	f := loadFixtures(t)
	req := readStageRequest(t, f, "i2i4")
	img := decodeFile(t, filepath.Join(f.dir, "i2i4", "ref0.jpg"))
	if b := img.Bounds(); b.Dx() != 3120 || b.Dy() != 4160 {
		t.Fatalf("ref0.jpg decodes %dx%d, want the oriented 3120x4160", b.Dx(), b.Dy())
	}
	p := &Pipeline{o: Options{MaxPixels: DefaultMaxPixels}}
	r := Request{Prompt: "x", MatchImage: true, Images: []image.Image{img}}
	if err := p.Check(r); err != nil {
		t.Fatal(err)
	}
	if err := p.resolve(&r); err != nil {
		t.Fatal(err)
	}
	if r.Width != req.Width || r.Height != req.Height {
		t.Errorf("matched %dx%d, ComfyUI %dx%d", r.Width, r.Height, req.Width, req.Height)
	}
	if b := r.Images[0].Bounds(); b.Dx() != r.Width || b.Dy() != r.Height {
		t.Errorf("first reference %dx%d, want the matched %dx%d", b.Dx(), b.Dy(), r.Width, r.Height)
	}
}

func TestCheckRefusesPastSideAndPixels(t *testing.T) {
	p := &Pipeline{o: Options{MaxPixels: DefaultMaxPixels}}
	refuse := func(name string, r Request, limit string) {
		t.Helper()
		r.Prompt, r.Steps, r.CFG = "x", 16, 1
		err := p.Check(r)
		if err == nil || !strings.Contains(err.Error(), limit) {
			t.Errorf("%s: %v, want an error naming %s", name, err, limit)
		}
	}
	refuse("8208x512", Request{Width: 8208, Height: 512}, "8192")
	refuse("2560x2560", Request{Width: 2560, Height: 2560}, "4456448")
	refuse("match_image without images", Request{MatchImage: true}, "match_image")
	five := make([]image.Image, 5)
	for i := range five {
		five[i] = image.NewRGBA(image.Rect(0, 0, 64, 64))
	}
	refuse("five references", Request{Width: 1024, Height: 1024, Images: five}, "4")
	if err := p.Check(Request{Prompt: "x", Width: 1680, Height: 2512}); err != nil {
		t.Errorf("1680x2512: %v", err)
	}
	tiny := Request{Prompt: "x", MatchImage: true, Images: []image.Image{image.NewRGBA(image.Rect(0, 0, 300, 300))}}
	if err := p.Check(tiny); err != nil {
		t.Errorf("a 300x300 reference matched: %v", err)
	}
	if err := p.resolve(&tiny); err != nil || tiny.Width != 2048 || tiny.Height != 2048 {
		t.Errorf("a 300x300 reference matched to %dx%d (%v), want 2048x2048", tiny.Width, tiny.Height, err)
	}
}

// comfyTimings is a stage's timings.json.
type comfyTimings struct {
	Steps      []float64 `json:"steps"`
	Decode     float64   `json:"decode"`
	DecodeMode string    `json:"decode_mode"`
	Total      float64   `json:"total"`
}

func (c comfyTimings) medianStep() float64 {
	if len(c.Steps) == 0 {
		return math.NaN()
	}
	s := append([]float64(nil), c.Steps...)
	sort.Float64s(s)
	return s[len(s)/2]
}

// run4MP draws a 4 MP stage and holds it to minDB against ComfyUI's.
func run4MP(t *testing.T, stage string, refs []string, minDB float64) {
	f := loadFixtures(t)
	req := readStageRequest(t, f, stage)
	needFile(t, DiTPath())
	needFile(t, EncoderPath())
	needFile(t, VAEPath())
	var comfy comfyTimings
	if b, err := os.ReadFile(filepath.Join(f.dir, stage, "timings.json")); err == nil {
		json.Unmarshal(b, &comfy)
	}
	r := Request{Prompt: req.Prompt, Seed: req.Seed, Steps: req.Steps, CFG: 1, Width: req.Width, Height: req.Height}
	for _, name := range refs {
		r.Images = append(r.Images, decodeFile(t, filepath.Join(f.dir, stage, name)))
	}
	if len(refs) > 0 {
		r.MatchImage = true
		r.Width, r.Height = 0, 0
	}

	p, err := Open(Options{Keep: true})
	if err != nil {
		t.Fatal(err)
	}
	defer p.Close()
	img, tm, err := p.Generate(r, nil)
	if err != nil {
		t.Fatal(err)
	}
	if b := img.Bounds(); b.Dx() != req.Width || b.Dy() != req.Height {
		t.Fatalf("%s: drew %dx%d, ComfyUI %dx%d", stage, b.Dx(), b.Dy(), req.Width, req.Height)
	}
	// ComfyUI's rgb is its tiled decode, some 31 dB from a whole one
	// (TestVAEDecode4MPMatchesComfyUI): the picture is held to ComfyUI's
	// latent decoded whole, by the VAE whose bands are 53 dB from that.
	want, err := p.vae.Decode(f.read(t, stage+"/latent"), req.Height/16, req.Width/16)
	if err != nil {
		t.Fatal(err)
	}
	val := psnr(img, want)
	t.Logf("%s (%dx%d, %d refs): PSNR %.2f dB vs ComfyUI's latent, %.2f dB vs its tiled rgb; golem encode %v, load %v, sample %v (%v a step), decode %v, total %v",
		stage, req.Width, req.Height, len(refs), val, psnr(img, f.read(t, stage+"/rgb")), tm.Encode, tm.Load, tm.Sample, tm.Sample/time.Duration(req.Steps), tm.Decode, tm.Total)
	t.Logf("%s ComfyUI: median step %.2f s, decode %.2f s (%s), total %.1f s", stage, comfy.medianStep(), comfy.Decode, comfy.DecodeMode, comfy.Total)
	if dir := os.Getenv("FLUX3_SAVE"); dir != "" {
		if out, err := os.Create(filepath.Join(dir, stage+".png")); err == nil {
			p.WritePNG(out, img, r)
			out.Close()
		}
	}
	if val < minDB {
		t.Errorf("%s: PSNR %.2f dB < %g dB", stage, val, minDB)
	}
}

// Without a reference to hold it, the 2048 x 2048 trajectory drifts more than
// the 1 MP one (34 dB, TestAcceptMatchesComfyUI) over 16 steps: the same
// picture, the same pose and light, details of the mouth and the ball's
// pattern apart, at 25 dB. The first DiT call is 0.68% from ComfyUI's
// (TestDiT4MPMatchesComfyUI).
func TestPicture4MPMatchesComfyUI(t *testing.T) { run4MP(t, "t2i4", nil, 24) }

func TestI2I4MPMatchesComfyUI(t *testing.T) {
	f := loadFixtures(t)
	img := decodeFile(t, filepath.Join(f.dir, "i2i4", "ref0.jpg"))
	rgb, rw, rh := ResizeReference(ScaleToTotalPixels(img, 4, 16), 1.0, RefTokens)
	want := f.read(t, "i2i4/ref_rgb0")
	if len(want) != len(rgb) {
		t.Fatalf("first reference %dx%d (%d floats), ComfyUI %d floats", rw, rh, len(rgb), len(want))
	}
	p := psnrFloat(rgb, want)
	t.Logf("first reference %dx%d vs ComfyUI's: PSNR %.2f dB", rw, rh, p)
	if p < 35 {
		t.Errorf("first reference PSNR %.2f dB < 35 dB", p)
	}
	run4MP(t, "i2i4", []string{"ref0.jpg"}, 30)
}

func TestMultiRef4MPMatchesComfyUI(t *testing.T) {
	run4MP(t, "mi2i4", []string{"ref0.jpg", "ref1.jpg", "ref2.png", "ref3.png"}, 30)
}
