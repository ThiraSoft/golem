package tha3

import (
	"math"
	"testing"
)

// fieldsAt makes a morpher's decision by hand: a square of alpha in the
// middle of a 64×64 field, the eyes painted one colour and everything under
// them another, both in light.
func fieldsAt(paint, skin [3]float32) (faceFields, Tensor) {
	const n = 64
	f := faceFields{
		grid:       NewTensor(2, n, n),
		mouthAlpha: NewTensor(1, n, n),
		mouthColor: NewTensor(4, n, n),
		eyeAlpha:   NewTensor(1, n, n),
		eyeColor:   NewTensor(4, n, n),
	}
	under := NewTensor(4, n, n)
	for c := range 4 {
		for i := range under.Plane(c) {
			v := float32(1)
			if c < 3 {
				v = skin[c]
			}
			under.Plane(c)[i] = v*2 - 1
			p := paint[min(c, 2)]
			if c == 3 {
				p = 1
			}
			f.eyeColor.Plane(c)[i] = p*2 - 1
		}
	}
	for y := 16; y < 48; y++ {
		for x := 16; x < 48; x++ {
			f.eyeAlpha.Plane(0)[y*n+x] = 1
		}
	}
	return f, under
}

func TestEyeToneMatchesTheSkin(t *testing.T) {
	paint := [3]float32{0.8, 0.6, 0.5}
	skin := [3]float32{0.8, 0.66, 0.6}
	f, under := fieldsAt(paint, skin)
	g := eyeTone(f, under, 1)
	for c := range 3 {
		want := skin[c] / paint[c]
		if math.Abs(float64(g[c]-want)) > 1e-3 {
			t.Errorf("channel %d: gain %v, want %v", c, g[c], want)
		}
	}
	// The colour it corrects lands on the skin.
	got := toned(f.eyeColor, g)
	for c := range 3 {
		v := (got.Plane(c)[32*64+32] + 1) / 2
		if math.Abs(float64(v-skin[c])) > 1e-3 {
			t.Errorf("channel %d: painted %v, want the skin's %v", c, v, skin[c])
		}
	}
}

func TestEyeToneAmountGoesPartWay(t *testing.T) {
	paint := [3]float32{0.8, 0.6, 0.5}
	skin := [3]float32{0.8, 0.66, 0.6}
	f, under := fieldsAt(paint, skin)
	full, half := eyeTone(f, under, 1), eyeTone(f, under, 0.5)
	for c := range 3 {
		want := 1 + (full[c]-1)/2
		if math.Abs(float64(half[c]-want)) > 1e-3 {
			t.Errorf("channel %d: half the gain is %v, want %v", c, half[c], want)
		}
	}
	if off := eyeTone(f, under, 0); off != noTone {
		t.Errorf("an amount of zero gives %v, want %v", off, noTone)
	}
}

func TestEyeToneHeldInRange(t *testing.T) {
	// A picture whose skin is nothing like what is painted: the gain stops
	// at the range rather than dragging the colour there.
	f, under := fieldsAt([3]float32{0.9, 0.9, 0.9}, [3]float32{0.1, 0.1, 0.1})
	g := eyeTone(f, under, 1)
	for c := range 3 {
		if g[c] != toneLow {
			t.Errorf("channel %d: gain %v, want it held at %v", c, g[c], toneLow)
		}
	}
}

func TestEyeToneKeepsAPictureItCannotMeasure(t *testing.T) {
	f, under := fieldsAt([3]float32{0.8, 0.6, 0.5}, [3]float32{0.8, 0.66, 0.6})
	// Nothing painted: no lid to read.
	for i := range f.eyeAlpha.Data {
		f.eyeAlpha.Data[i] = 0
	}
	if g := eyeTone(f, under, 1); g != noTone {
		t.Errorf("gain %v on a picture with no lid, want %v", g, noTone)
	}
	// Everything painted: no skin around it to read.
	for i := range f.eyeAlpha.Data {
		f.eyeAlpha.Data[i] = 1
	}
	if g := eyeTone(f, under, 1); g != noTone {
		t.Errorf("gain %v on a picture with no skin around the lid, want %v", g, noTone)
	}
}

func TestTonedKeepsTheAlpha(t *testing.T) {
	f, _ := fieldsAt([3]float32{0.5, 0.5, 0.5}, [3]float32{1, 1, 1})
	got := toned(f.eyeColor, [3]float32{1.2, 1.2, 1.2})
	if v := got.Plane(3)[0]; v != f.eyeColor.Plane(3)[0] {
		t.Errorf("alpha %v, want %v", v, f.eyeColor.Plane(3)[0])
	}
	if v := (got.Plane(0)[0] + 1) / 2; math.Abs(float64(v-0.6)) > 1e-6 {
		t.Errorf("colour %v, want 0.6", v)
	}
	// A gain that would take the light past one stops at one.
	if v := (toned(f.eyeColor, [3]float32{4, 4, 4}).Plane(0)[0] + 1) / 2; v != 1 {
		t.Errorf("colour %v, want 1", v)
	}
	if got := toned(f.eyeColor, noTone); &got.Data[0] != &f.eyeColor.Data[0] {
		t.Error("a gain of one copied the colour")
	}
}

// lidOffset is how far the skin a frame shows over a closed eye sits from the
// skin around it, channel by channel: what the eye tone is there to shrink.
func lidOffset(frame Tensor, f faceFields) [3]float32 {
	k := frame.H / Size
	a := resize(f.eyeAlpha, 192*k, 192*k)
	lid, ring := maskRegions(a)
	at := func(r []int) []int {
		out := make([]int, 0, len(r))
		for _, i := range r {
			y, x := i/a.W+32*k, i%a.W+160*k
			out = append(out, y*frame.W+x)
		}
		return out
	}
	painted, skin := brightMean(frame, at(lid)), brightMean(frame, at(ring))
	var off [3]float32
	for c := range 3 {
		off[c] = painted[c] - skin[c]
	}
	return off
}

// On a real picture, with the real networks: the lid a shut eye shows must
// come closer to the skin around it than the networks left it.
func TestEyeToneBringsTheLidOntoTheSkin(t *testing.T) {
	f := loadFixtures(t, "neutral")
	p := openPoserOrSkip(t)
	defer p.Close()
	img := f.tensor(t, "image")
	var shut [NumParams]float32
	shut[EyeWinkLeft], shut[EyeWinkRight] = 1, 1

	if err := p.SetImage(img); err != nil {
		t.Fatal(err)
	}
	plain, err := p.Pose(shut)
	if err != nil {
		t.Fatal(err)
	}
	before := lidOffset(plain, p.faced)

	if err := p.SetEyeTone(1); err != nil {
		t.Fatal(err)
	}
	if g := p.EyeTone(); g == noTone {
		t.Fatalf("no gain measured on %s", "neutral")
	}
	corrected, err := p.Pose(shut)
	if err != nil {
		t.Fatal(err)
	}
	after := lidOffset(corrected, p.faced)

	for c := range 3 {
		a, b := math.Abs(float64(after[c])), math.Abs(float64(before[c]))
		if a > b {
			t.Errorf("channel %d: the lid is %v from the skin, was %v", c, a, b)
		}
	}
	t.Logf("gain %v, offset %v then %v", p.EyeTone(), before, after)
}
