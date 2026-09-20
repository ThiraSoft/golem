package tha3

import (
	"math"
	"testing"
)

// paintedPicture is a flat skin with one dark strand running down it, and
// the mask a hand would draw over the strand: a little wider than it is.
func paintedPicture() (picture, mask Tensor, strandAt func(y, x int) bool) {
	const n = Size
	skin := [3]float32{0.95, 0.82, 0.74}
	strand := [3]float32{0.18, 0.14, 0.22}
	picture = NewTensor(4, n, n)
	mask = NewTensor(1, n, n)
	// The window the face morpher reads, so that the mask lands where the
	// poser looks for it.
	y0, y1 := 32+60, 32+150
	x0 := 160 + 96
	strandAt = func(y, x int) bool { return y >= y0 && y < y1 && x >= x0-2 && x < x0+3 }
	for y := 0; y < n; y++ {
		for x := 0; x < n; x++ {
			i := y*n + x
			c := skin
			if strandAt(y, x) {
				c = strand
			}
			for k := range 3 {
				picture.Plane(k)[i] = float32(srgbToLinear(float64(c[k])))*2 - 1
			}
			picture.Plane(3)[i] = 1
			if strandAt(y, x) || nearStrand(y, x) {
				mask.Plane(0)[i] = 1
			}
		}
	}
	return picture, mask, strandAt
}

// nearStrand is the pixel beside the strand, which a hand drawing over it
// catches too.
func nearStrand(y, x int) bool {
	return y >= 32+60 && y < 32+150 && x >= 160+96-4 && x < 160+96+5
}

// A mask is used as it was drawn: what it covers is kept, and the window of
// the face is what the poser reads of it.
func TestFrontCutToTheWindow(t *testing.T) {
	picture, mask, strandAt := paintedPicture()
	p := &Poser{timings: Timings{}, view: whole, scale: 1, eyeTone: noTone}
	if err := p.SetFront(mask); err != nil {
		t.Fatal(err)
	}
	if p.frontFace.Data != nil {
		t.Error("the mask was cut before there was a picture to cut it for")
	}
	p.image = picture
	p.cutFront()
	if p.frontFace.H != 192 || p.frontFace.W != 192 {
		t.Fatalf("the window is %dx%d", p.frontFace.H, p.frontFace.W)
	}
	for i, v := range p.frontFace.Plane(0) {
		y, x := i/192+32, i%192+160
		if want := boolFloat(strandAt(y, x) || nearStrand(y, x)); v != want {
			t.Fatalf("at %d,%d the mask is %v, want %v", y, x, v, want)
		}
	}
	// A mask of nothing is no mask at all.
	if err := p.SetFront(NewTensor(1, Size, Size)); err != nil {
		t.Fatal(err)
	}
	if p.frontFace.Data != nil {
		t.Error("an empty mask was kept")
	}
}

func boolFloat(b bool) float32 {
	if b {
		return 1
	}
	return 0
}

func TestSetFrontChecksTheMask(t *testing.T) {
	p := &Poser{timings: Timings{}, view: whole, scale: 1, eyeTone: noTone}
	if err := p.SetFront(NewTensor(1, 256, 256)); err == nil {
		t.Error("a 256 mask was taken")
	}
	if err := p.SetFront(NewTensor(4, Size, Size)); err == nil {
		t.Error("a four channel mask was taken")
	}
	mask := NewTensor(1, Size, Size)
	if err := p.SetFront(mask); err != nil {
		t.Fatal(err)
	}
	// The mask is copied, so the caller cannot change what the poser holds.
	mask.Data[0] = 1
	if p.Front().Data[0] != 0 {
		t.Error("the poser kept the caller's own mask")
	}
	if err := p.SetFront(Tensor{}); err != nil {
		t.Fatal(err)
	}
	if p.Front().Data != nil {
		t.Error("an empty mask did not clear it")
	}
}

// The whole thing on the real networks: a strand drawn across a shut eye
// stays where the mask says it is in front, and goes without it.
func TestFrontKeepsTheStrandOverAShutEye(t *testing.T) {
	f := loadFixtures(t, "neutral")
	p := openPoserOrSkip(t)
	defer p.Close()
	img := f.tensor(t, "image")

	// A strand of the character's own hair, laid across both eyes: the
	// darkest colour of the picture, over the window the morpher repaints.
	strand := darkest(img)
	picture := img.Clone()
	stroke := NewTensor(1, Size, Size)
	const (
		top, bottom = 130, 155
		left, right = 225, 295
	)
	for y := top; y < bottom; y++ {
		for x := left; x < right; x += 12 {
			for w := 0; w < 3; w++ {
				i := y*Size + x + w
				for k := range 3 {
					picture.Plane(k)[i] = strand[k]
				}
				stroke.Plane(0)[i] = 1
			}
		}
	}
	if err := p.SetImage(picture); err != nil {
		t.Fatal(err)
	}
	var shut [NumParams]float32
	shut[EyeWinkLeft], shut[EyeWinkRight] = 1, 1
	painted, err := p.Pose(shut)
	if err != nil {
		t.Fatal(err)
	}
	if err := p.SetFront(stroke); err != nil {
		t.Fatal(err)
	}
	if p.frontFace.Data == nil {
		t.Fatal("the mask was not refined")
	}
	kept, err := p.Pose(shut)
	if err != nil {
		t.Fatal(err)
	}
	// Under the stroke, the frame with the mask is far closer to the
	// picture than the frame without.
	with, without := gapTo(kept, picture, stroke), gapTo(painted, picture, stroke)
	if with >= without/2 {
		t.Errorf("under the mask the frame is %.4f from the picture, %.4f without it", with, without)
	}
}

// darkest is the darkest colour of a picture, as the tensors hold it.
func darkest(t Tensor) [3]float32 {
	best, at := float32(4), 0
	for i := range t.Plane(0) {
		if t.Plane(3)[i] < 0 {
			continue
		}
		var lit float32
		for k := range 3 {
			lit += t.Plane(k)[i]
		}
		if lit < best {
			best, at = lit, i
		}
	}
	var c [3]float32
	for k := range 3 {
		c[k] = t.Plane(k)[at]
	}
	return c
}

// gapTo is the mean distance between a frame and the picture where the mask
// is.
func gapTo(frame, picture, mask Tensor) float64 {
	var sum float64
	n := 0
	for i, v := range mask.Plane(0) {
		if v <= 0.5 {
			continue
		}
		n++
		for k := range 3 {
			d := float64(frame.Plane(k)[i] - picture.Plane(k)[i])
			sum += math.Abs(d)
		}
	}
	if n == 0 {
		return 0
	}
	return sum / float64(3*n)
}
