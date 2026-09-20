package tha3

import (
	"errors"
	"fmt"
	"image"
	"image/color"
	"io/fs"
	"testing"

	"github.com/ThiraSoft/golem/internal/kyutai/reference"
	"github.com/ThiraSoft/golem/vk"
)

func openPoserOrSkip(t *testing.T) *Poser {
	t.Helper()
	p, err := Open(Dir())
	if errors.Is(err, fs.ErrNotExist) {
		t.Skipf("weights not found (%v): see ref/tha3/README.md", err)
	}
	if err != nil {
		t.Fatal(err)
	}
	return p
}

// The card against the processor, which is itself held to PyTorch: every
// waypoint the processor's tracer emits, on every recorded pose, in the
// order it was emitted, so the first one out of tolerance says where.
func TestCardMatchesCPU(t *testing.T) {
	dev, err := vk.Open()
	if err != nil {
		t.Skipf("no Vulkan device: %v", err)
	}
	dev.Close()
	cpu := openPoserOrSkip(t)
	defer cpu.Close()
	card := openPoserOrSkip(t)
	defer card.Close()
	if err := card.useVulkan(true); err != nil {
		t.Fatal(err)
	}
	for _, run := range runs {
		t.Run(run, func(t *testing.T) {
			f := loadFixtures(t, run)
			var names []string
			want := map[string]Tensor{}
			cpu.trace = func(name string, x Tensor) {
				names = append(names, name)
				want[name] = x.Clone()
			}
			img := f.tensor(t, "image")
			if err := cpu.SetImage(img); err != nil {
				t.Fatal(err)
			}
			if err := card.SetImage(img); err != nil {
				t.Fatal(err)
			}
			var pose [NumParams]float32
			copy(pose[:], f.Pose)
			cpuFrame, err := cpu.Pose(pose)
			if err != nil {
				t.Fatal(err)
			}
			cardFrame, err := card.Pose(pose)
			if err != nil {
				t.Fatal(err)
			}
			for _, name := range names {
				got := card.gpu.run.Waypoint(name)
				if got == nil {
					t.Fatalf("%s: the card traced nothing under that name", name)
				}
				reference.Compare(t, name, got, want[name].Data, tolerance)
				if t.Failed() {
					t.FailNow()
				}
			}
			reference.Compare(t, "frame", cardFrame.Data, cpuFrame.Data, tolerance)
		})
	}
}

// The plan UseVulkan builds has no traces, so its tensors live shorter and
// share more of the arena than in the traced plan TestCardMatchesCPU checks.
// This runs the shipped plan: the picture set before the card is taken, two
// poses a picture, SetImage again for every run, frames compared.
func TestCardShippedPlan(t *testing.T) {
	dev, err := vk.Open()
	if err != nil {
		t.Skipf("no Vulkan device: %v", err)
	}
	dev.Close()
	cpu := openPoserOrSkip(t)
	defer cpu.Close()
	card := openPoserOrSkip(t)
	defer card.Close()
	for i, run := range runs {
		f := loadFixtures(t, run)
		img := f.tensor(t, "image")
		if err := cpu.SetImage(img); err != nil {
			t.Fatal(err)
		}
		if err := card.SetImage(img); err != nil {
			t.Fatal(err)
		}
		if i == 0 {
			if err := card.UseVulkan(); err != nil {
				t.Fatal(err)
			}
		}
		var first, second [NumParams]float32
		copy(first[:], f.Pose)
		second[MouthOoo], second[HeadX] = 0.7, -0.4
		for _, pose := range [][NumParams]float32{first, second} {
			want, err := cpu.Pose(pose)
			if err != nil {
				t.Fatal(err)
			}
			got, err := card.Pose(pose)
			if err != nil {
				t.Fatal(err)
			}
			reference.Compare(t, run+" frame", got.Data, want.Data, tolerance)
		}
	}
}

// The larger picture on the card against the processor, at a scale of two
// and the view the avatar shows: every stage's decisions resized and applied
// again, the last one only over the view. The larger picture carries a
// pattern the picture has not, so that a pixel read at the wrong place
// shows. The second pose keeps the eyebrows and the face, so the passes that
// start later are checked too.
func TestCardHighMatchesCPU(t *testing.T) {
	dev, err := vk.Open()
	if err != nil {
		t.Skipf("no Vulkan device: %v", err)
	}
	dev.Close()
	view := image.Rect(144, 32, 368, 256)
	cpu := openPoserOrSkip(t)
	defer cpu.Close()
	card := openPoserOrSkip(t)
	defer card.Close()
	if err := card.UseVulkan(); err != nil {
		t.Fatal(err)
	}
	for _, p := range []*Poser{cpu, card} {
		if err := p.SetView(view); err != nil {
			t.Fatal(err)
		}
		if err := p.SetScale(2); err != nil {
			t.Fatal(err)
		}
		// With the sharpening on, so that the two extra resamplings
		// and the two sharpen passes of the high path are compared
		// too; the other scaled tests leave it off.
		if err := p.SetSharpen(0.6); err != nil {
			t.Fatal(err)
		}
		// And with the eyes toned, so that the gain the card reads
		// from the pose buffer is compared to the one the processor
		// multiplies in.
		if err := p.SetEyeTone(1); err != nil {
			t.Fatal(err)
		}
	}
	for _, run := range runs {
		f := loadFixtures(t, run)
		img := f.tensor(t, "image")
		high := ResizeBilinear(img, 2*Size, 2*Size)
		for c := 0; c < 3; c++ {
			plane := high.Plane(c)
			for i := range plane {
				if (i/high.W+i%high.W)%3 == 0 {
					plane[i] = -plane[i] * 0.5
				}
			}
		}
		// A mask of what the picture keeps in front of the face, so
		// that the composite the card does for it is compared too.
		front := NewTensor(1, Size, Size)
		for y := 120; y < 170; y++ {
			for x := 220; x < 300; x += 9 {
				for w := 0; w < 4; w++ {
					front.Plane(0)[y*Size+x+w] = 1
				}
			}
		}
		for _, p := range []*Poser{cpu, card} {
			if err := p.SetImageHigh(img, high); err != nil {
				t.Fatal(err)
			}
			if p.EyeTone() == noTone {
				t.Fatalf("%s: no gain measured, the tone is not being compared", run)
			}
			if err := p.SetFront(front); err != nil {
				t.Fatal(err)
			}
			if p.frontFace.Data == nil || p.hiFront.Data == nil {
				t.Fatalf("%s: the front mask was not refined", run)
			}
		}
		var first [NumParams]float32
		copy(first[:], f.Pose)
		second := first
		second[HeadX], second[Breathing] = -0.4, 0.8
		for _, pose := range [][NumParams]float32{first, second} {
			want, err := cpu.Pose(pose)
			if err != nil {
				t.Fatal(err)
			}
			got, err := card.Pose(pose)
			if err != nil {
				t.Fatal(err)
			}
			if got.H != 2*view.Dy() || got.W != 2*view.Dx() {
				t.Fatalf("%s: frame is %dx%d, want %dx%d", run, got.W, got.H, 2*view.Dx(), 2*view.Dy())
			}
			reference.Compare(t, run+" frame", got.Data, want.Data, tolerance)
		}
	}
}

// Finished frames, bytes on a background through a zoom, from the card and
// from the processor, at the scales of one and two. The card's pow and the
// processor's differ in the last bits, which may round a byte either way.
func TestCardFinishMatchesCPU(t *testing.T) {
	dev, err := vk.Open()
	if err != nil {
		t.Skipf("no Vulkan device: %v", err)
	}
	dev.Close()
	f := loadFixtures(t, "mix")
	img := f.tensor(t, "image")
	var pose [NumParams]float32
	copy(pose[:], f.Pose)
	for _, k := range []int{1, 2} {
		cpu := openPoserOrSkip(t)
		card := openPoserOrSkip(t)
		if err := card.UseVulkan(); err != nil {
			t.Fatal(err)
		}
		for _, p := range []*Poser{cpu, card} {
			for _, err := range []error{p.SetView(image.Rect(144, 32, 368, 256)), p.SetScale(k),
				p.SetBackground(color.RGBA{0x1f, 0x1a, 0x28, 0xff}), p.SetImage(img)} {
				if err != nil {
					t.Fatal(err)
				}
			}
		}
		for _, z := range []Zoom{{}, {1.04, 0.5, 0.48}, {1.1, 0.47, 0.52}} {
			want, err := cpu.PoseRGBA(pose, z)
			if err != nil {
				t.Fatal(err)
			}
			got, err := card.PoseRGBA(pose, z)
			if err != nil {
				t.Fatal(err)
			}
			if got.Rect != want.Rect || got.Rect.Dx() != 224*k {
				t.Fatalf("scale %d: card %v, processor %v", k, got.Rect, want.Rect)
			}
			worst := 0
			for i := range got.Pix {
				worst = max(worst, abs(int(got.Pix[i])-int(want.Pix[i])))
			}
			t.Logf("scale %d zoom %v: worst byte gap %d", k, z, worst)
			if worst > 2 {
				t.Errorf("scale %d zoom %v: a byte differs by %d", k, z, worst)
			}
		}
		if _, err := card.Pose(pose); err == nil {
			t.Error("Pose after SetBackground did not fail")
		}
		cpu.Close()
		card.Close()
	}
}

func abs(x int) int {
	if x < 0 {
		return -x
	}
	return x
}

// With the body held, a pose that only opens the mouth runs neither the
// rotator nor the editor. The card and the processor must still agree, and
// what they make must be close to the frame the two networks would have
// made, which the same poser without the hold gives.
func TestCardHeldBodyMatchesCPU(t *testing.T) {
	dev, err := vk.Open()
	if err != nil {
		t.Skipf("no Vulkan device: %v", err)
	}
	dev.Close()
	f := loadFixtures(t, "head") // the head turned, where the editor repaints most
	img := f.tensor(t, "image")
	var first [NumParams]float32
	copy(first[:], f.Pose)
	second := first
	second[MouthAaa], second[MouthDelta] = 1, 0.6
	bg := color.RGBA{0x1f, 0x1a, 0x28, 0xff}

	frames := map[string]*image.RGBA{}
	for _, held := range []bool{false, true} {
		for _, card := range []bool{false, true} {
			p := openPoserOrSkip(t)
			defer p.Close()
			if card {
				if err := p.UseVulkan(); err != nil {
					t.Fatal(err)
				}
			}
			for _, err := range []error{p.SetView(image.Rect(144, 32, 368, 256)), p.SetScale(2),
				p.SetBackground(bg), p.SetHeldBody(held), p.SetImage(img)} {
				if err != nil {
					t.Fatal(err)
				}
			}
			if _, err := p.PoseRGBA(first, Zoom{}); err != nil {
				t.Fatal(err)
			}
			got, err := p.PoseRGBA(second, Zoom{})
			if err != nil {
				t.Fatal(err)
			}
			frames[fmt.Sprint(held, "/", card)] = got
			if held && card && p.Timings()[netEditor] != 0 {
				t.Error("the editor ran on a held body")
			}
		}
	}
	gap := func(a, b *image.RGBA) int {
		worst := 0
		for i := range a.Pix {
			worst = max(worst, abs(int(a.Pix[i])-int(b.Pix[i])))
		}
		return worst
	}
	if g := gap(frames["true/true"], frames["true/false"]); g > 2 {
		t.Errorf("held: the card and the processor differ by %d", g)
	}
	t.Logf("held against whole: %d on the card, %d on the processor",
		gap(frames["true/true"], frames["false/true"]), gap(frames["true/false"], frames["false/false"]))
}
