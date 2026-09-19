package tha3

import (
	"errors"
	"image"
	"image/color"
	"image/draw"
	"io/fs"
	"testing"

	"github.com/ThiraSoft/golem/internal/kyutai/reference"
	"github.com/ThiraSoft/golem/vk"
)

// End to end: the recorded picture and pose in, every network's inputs and
// outputs compared on the way, the frame last.
func TestPoser(t *testing.T) {
	for _, run := range runs {
		t.Run(run, func(t *testing.T) {
			f := loadFixtures(t, run)
			p, err := Open(Dir())
			if err != nil {
				if errors.Is(err, fs.ErrNotExist) {
					t.Skipf("weights not found (%v): see ref/tha3/README.md", err)
				}
				t.Fatal(err)
			}
			defer p.Close()
			p.trace = f.checker(t, "")
			if err := p.SetImage(f.tensor(t, "image")); err != nil {
				t.Fatal(err)
			}
			var pose [NumParams]float32
			copy(pose[:], f.Pose)
			frame, err := p.Pose(pose)
			if err != nil {
				t.Fatal(err)
			}
			reference.Compare(t, "frame", frame.Data, f.tensor(t, netEditor+".out.0").Data, tolerance)
		})
	}
}

func TestPoseBeforeImage(t *testing.T) {
	if _, err := (&Poser{}).Pose([NumParams]float32{}); err == nil {
		t.Fatal("Pose without a picture did not fail")
	}
}

func TestPoseReusesTheLastFrame(t *testing.T) {
	f := loadFixtures(t, "neutral")
	p := openPoserOrSkip(t)
	defer p.Close()
	img := f.tensor(t, "image")
	if err := p.SetImage(img); err != nil {
		t.Fatal(err)
	}
	var pose [NumParams]float32
	a, err := p.Pose(pose)
	if err != nil {
		t.Fatal(err)
	}
	b, err := p.Pose(pose)
	if err != nil {
		t.Fatal(err)
	}
	if p.renders != 1 {
		t.Fatalf("two identical poses rendered %d times, want 1", p.renders)
	}
	for i := range a.Data {
		if a.Data[i] != b.Data[i] {
			t.Fatalf("the reused frame differs at %d", i)
		}
	}
	b.Data[0] = 42
	if c, _ := p.Pose(pose); c.Data[0] == 42 {
		t.Fatal("the reused frame is shared with the caller")
	}
	pose[MouthAaa] = 1
	if _, err := p.Pose(pose); err != nil {
		t.Fatal(err)
	}
	if p.renders != 2 {
		t.Fatalf("a new pose rendered %d times in total, want 2", p.renders)
	}
	if err := p.SetImage(img); err != nil {
		t.Fatal(err)
	}
	if _, err := p.Pose(pose); err != nil {
		t.Fatal(err)
	}
	if p.renders != 3 {
		t.Fatalf("a new picture did not invalidate the last frame: %d renders, want 3", p.renders)
	}
}

func TestClosedPoserFails(t *testing.T) {
	p := openPoserOrSkip(t)
	p.Close()
	if err := p.SetImage(NewTensor(4, Size, Size)); err == nil {
		t.Fatal("SetImage after Close did not fail")
	}
	if _, err := p.Pose([NumParams]float32{}); err == nil {
		t.Fatal("Pose after Close did not fail")
	}
	if err := p.UseVulkan(); err == nil {
		t.Fatal("UseVulkan after Close did not fail")
	}
}

// A pose that keeps the eyebrows, or the eyebrows and the face, of the last
// one starts at a later stage and must give the frame a full pass gives, on
// the processor and on the card.
func TestPoseStartsAtTheFirstChange(t *testing.T) {
	f := loadFixtures(t, "neutral")
	img := f.tensor(t, "image")
	var base [NumParams]float32
	copy(base[:], f.Pose)
	steps := []struct {
		name   string
		change func(*[NumParams]float32)
		want   stage
	}{
		{"first", func(*[NumParams]float32) {}, stageEyebrows},
		{"breathing", func(p *[NumParams]float32) { p[Breathing] = 0.8 }, stageBody},
		{"mouth", func(p *[NumParams]float32) { p[MouthAaa] = 0.6 }, stageFace},
		{"head and mouth", func(p *[NumParams]float32) { p[HeadX] = -0.3; p[MouthAaa] = 0.2 }, stageFace},
		{"eyebrows", func(p *[NumParams]float32) { p[EyebrowHappyLeft] = 1 }, stageEyebrows},
		{"neck", func(p *[NumParams]float32) { p[NeckZ] = 0.2 }, stageBody},
	}
	check := func(t *testing.T, p, full *Poser) {
		for _, x := range []*Poser{p, full} {
			if err := x.SetImage(img); err != nil {
				t.Fatal(err)
			}
		}
		pose := base
		for _, s := range steps {
			s.change(&pose)
			got, err := p.Pose(pose)
			if err != nil {
				t.Fatal(err)
			}
			if p.started != s.want {
				t.Fatalf("%s: started at stage %d, want %d", s.name, p.started, s.want)
			}
			full.haveLast = false
			want, err := full.Pose(pose)
			if err != nil {
				t.Fatal(err)
			}
			reference.Compare(t, s.name, got.Data, want.Data, tolerance)
		}
	}
	t.Run("cpu", func(t *testing.T) {
		p, full := openPoserOrSkip(t), openPoserOrSkip(t)
		defer p.Close()
		defer full.Close()
		check(t, p, full)
	})
	t.Run("card", func(t *testing.T) {
		dev, err := vk.Open()
		if err != nil {
			t.Skipf("no Vulkan device: %v", err)
		}
		dev.Close()
		p, full := openPoserOrSkip(t), openPoserOrSkip(t)
		defer p.Close()
		defer full.Close()
		for _, x := range []*Poser{p, full} {
			if err := x.UseVulkan(); err != nil {
				t.Fatal(err)
			}
		}
		check(t, p, full)
	})
}

// A view is the same floats as that part of the whole frame, on the
// processor, and on the card whether the view came before or after it.
func TestView(t *testing.T) {
	f := loadFixtures(t, "neutral")
	img := f.tensor(t, "image")
	var pose [NumParams]float32
	copy(pose[:], f.Pose)
	view := image.Rect(144, 32, 368, 256)
	check := func(t *testing.T, card bool, before bool) {
		p, full := openPoserOrSkip(t), openPoserOrSkip(t)
		defer p.Close()
		defer full.Close()
		if card {
			if err := full.UseVulkan(); err != nil {
				t.Fatal(err)
			}
		}
		if before {
			if err := p.SetView(view); err != nil {
				t.Fatal(err)
			}
		}
		if card {
			if err := p.UseVulkan(); err != nil {
				t.Fatal(err)
			}
		}
		for _, x := range []*Poser{p, full} {
			if err := x.SetImage(img); err != nil {
				t.Fatal(err)
			}
		}
		if !before {
			if _, err := p.Pose(pose); err != nil { // a frame of the old view, cached
				t.Fatal(err)
			}
			if err := p.SetView(view); err != nil {
				t.Fatal(err)
			}
		}
		got, err := p.Pose(pose)
		if err != nil {
			t.Fatal(err)
		}
		if got.C != 4 || got.H != view.Dy() || got.W != view.Dx() {
			t.Fatalf("frame is %dx%dx%d, want 4x%dx%d", got.C, got.H, got.W, view.Dy(), view.Dx())
		}
		want, err := full.Pose(pose)
		if err != nil {
			t.Fatal(err)
		}
		want = want.Crop(view.Min.Y, view.Min.X, view.Dy(), view.Dx())
		for i := range got.Data {
			if got.Data[i] != want.Data[i] {
				t.Fatalf("float %d is %v, want %v", i, got.Data[i], want.Data[i])
			}
		}
	}
	t.Run("cpu", func(t *testing.T) { check(t, false, true) })
	dev, err := vk.Open()
	if err != nil {
		t.Skipf("no Vulkan device: %v", err)
	}
	dev.Close()
	t.Run("card, view first", func(t *testing.T) { check(t, true, true) })
	t.Run("card, view after", func(t *testing.T) { check(t, true, false) })
	t.Run("outside", func(t *testing.T) {
		p := openPoserOrSkip(t)
		defer p.Close()
		if err := p.SetView(image.Rect(400, 0, 600, 100)); err == nil {
			t.Fatal("a view past the frame was accepted")
		}
	})
}

// The larger picture at the scale of one is the picture itself: applying
// the networks' decisions again to it must give the frame they made.
func TestHighAtOneIsPose(t *testing.T) {
	f := loadFixtures(t, "neutral")
	p := openPoserOrSkip(t)
	defer p.Close()
	img := f.tensor(t, "image")
	var pose [NumParams]float32
	copy(pose[:], f.Pose)
	pose[MouthAaa], pose[HeadY] = 1, 0.5
	if err := p.SetImage(img); err != nil {
		t.Fatal(err)
	}
	want, err := p.Pose(pose)
	if err != nil {
		t.Fatal(err)
	}
	if err := p.SetImage(img); err != nil {
		t.Fatal(err)
	}
	p.high = img.Clone()
	p.hiEyebrow, p.hiBackground = p.decomposed.apply(img.Crop(64, 192, 128, 128))
	got, err := p.Pose(pose)
	if err != nil {
		t.Fatal(err)
	}
	reference.Compare(t, "frame", got.Data, want.Data, 1e-6)
}

// A finished frame at the scale of one, not zoomed, is the frame ToNRGBA
// makes, laid on the background.
func TestFinishIsFrameOnBackground(t *testing.T) {
	f := loadFixtures(t, "mix")
	img := f.tensor(t, "image")
	var pose [NumParams]float32
	copy(pose[:], f.Pose)
	view := image.Rect(144, 32, 368, 256)
	bg := color.RGBA{0x1f, 0x1a, 0x28, 0xff}
	p := openPoserOrSkip(t)
	defer p.Close()
	if err := p.SetView(view); err != nil {
		t.Fatal(err)
	}
	if err := p.SetImage(img); err != nil {
		t.Fatal(err)
	}
	frame, err := p.Pose(pose)
	if err != nil {
		t.Fatal(err)
	}
	want := image.NewRGBA(image.Rect(0, 0, view.Dx(), view.Dy()))
	draw.Draw(want, want.Rect, &image.Uniform{C: bg}, image.Point{}, draw.Src)
	draw.Draw(want, want.Rect, ToNRGBA(frame), image.Point{}, draw.Over)
	if err := p.SetBackground(bg); err != nil {
		t.Fatal(err)
	}
	got, err := p.PoseRGBA(pose, Zoom{})
	if err != nil {
		t.Fatal(err)
	}
	worst := 0
	for i := range got.Pix {
		worst = max(worst, abs(int(got.Pix[i])-int(want.Pix[i])))
	}
	if worst > 2 {
		t.Fatalf("a byte differs by %d", worst)
	}
}
