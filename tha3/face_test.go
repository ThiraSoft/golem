package tha3

import (
	"math"
	"math/rand"
	"testing"
)

func TestFaceMorpher(t *testing.T) {
	for _, run := range runs {
		t.Run(run, func(t *testing.T) {
			f := loadFixtures(t, run)
			w := openNetwork(t, netFaceMorpher)
			m := newFaceMorpher(w)
			if err := w.err(); err != nil {
				t.Fatal(err)
			}
			n := netFaceMorpher
			m.forward(f.tensor(t, n+".in.0"), f.pose(t, n+".in.1"), noTone, f.checker(t, n))
		})
	}
}

// midTensor is a picture in the middle of the range, light between 0.05 and
// 0.30, so that a gain of up to sharpenGain cannot reach the ceiling where
// the light clips and the ratios between the channels do move.
func midTensor(r *rand.Rand, c, h, w int) Tensor {
	t := randomTensor(r, c, h, w)
	for ch := range 3 {
		for i, v := range t.Plane(ch) {
			t.Plane(ch)[i] = v*0.125 - 0.775
		}
	}
	return t
}

// sharpFields is a face morpher's decision that repaints nothing but a box
// in the middle, the way an open mouth is a patch and the rest of the face
// is left alone.
func sharpFields(r *rand.Rand, n, box int) (faceFields, func(y, x int) bool) {
	f := faceFields{
		grid:       NewTensor(2, n, n),
		mouthAlpha: NewTensor(1, n, n),
		mouthColor: midTensor(r, 4, n, n),
		eyeAlpha:   NewTensor(1, n, n),
		eyeColor:   midTensor(r, 4, n, n),
	}
	lo, hi := (n-box)/2, (n+box)/2
	inside := func(y, x int) bool { return y >= lo && y < hi && x >= lo && x < hi }
	for y := range n {
		for x := range n {
			if inside(y, x) {
				f.mouthAlpha.Plane(0)[y*n+x] = 1
			}
		}
	}
	return f, inside
}

func TestApplySharpAtZeroIsApply(t *testing.T) {
	r := rand.New(rand.NewSource(3))
	const n = 24
	f, _ := sharpFields(r, n, 8)
	image := midTensor(r, 4, n, n)
	want, got := f.apply(image), f.applySharp(image, 0)
	for i := range want.Data {
		if want.Data[i] != got.Data[i] {
			t.Fatalf("at %d: %v, want %v", i, got.Data[i], want.Data[i])
		}
	}
}

// Sharpening touches what the morpher painted and nothing else: outside the
// alpha the frame is the one apply makes, to the bit, and the alpha channel
// is untouched everywhere, so the silhouette cannot grow a halo.
func TestApplySharpKeepsTheUnpainted(t *testing.T) {
	r := rand.New(rand.NewSource(4))
	const n = 24
	f, inside := sharpFields(r, n, 8)
	image := midTensor(r, 4, n, n)
	plain, sharp := f.apply(image), f.applySharp(image, 0.6)
	changed := 0
	for y := range n {
		for x := range n {
			p := y*n + x
			if plain.Plane(3)[p] != sharp.Plane(3)[p] {
				t.Fatalf("the alpha changed at %d,%d", y, x)
			}
			for c := range 3 {
				d := plain.Plane(c)[p] - sharp.Plane(c)[p]
				switch {
				case !inside(y, x) && d != 0:
					t.Fatalf("channel %d changed at %d,%d, outside the mask", c, y, x)
				case inside(y, x) && d != 0:
					changed++
				}
			}
		}
	}
	if changed == 0 {
		t.Fatal("nothing changed inside the mask")
	}
}

// What it does inside the mask is raise the local contrast of the
// brightness: the distance of a pixel's luma from the luma of the blur
// around it grows by the amount asked for.
func TestApplySharpRaisesContrast(t *testing.T) {
	r := rand.New(rand.NewSource(5))
	const n = 32
	f, inside := sharpFields(r, n, 12)
	image := midTensor(r, 4, n, n)
	plain, sharp := f.apply(image), f.applySharp(image, 0.6)
	blurred := blurWide(plain)
	luma := func(x Tensor, p int) float64 {
		var v float64
		for c := range 3 {
			v += float64(lumaWeights[c] * x.Plane(c)[p])
		}
		return v
	}
	var before, after float64
	for y := range n {
		for x := range n {
			if !inside(y, x) {
				continue
			}
			p := y*n + x
			before += math.Abs(luma(plain, p) - luma(blurred, p))
			after += math.Abs(luma(sharp, p) - luma(blurred, p))
		}
	}
	if want := before * 1.6; math.Abs(after-want) > want*1e-3 {
		t.Fatalf("the distance from the blur is %v, want %v", after, want)
	}
}

// And it does it without touching the colour: every channel of a pixel is
// multiplied by one number, so the ratios between them, which is what the hue
// and the saturation are, come out unchanged. This is what keeps a mouth from
// going vivid as the amount rises. Checked as a cross product, so that it
// holds whatever the gain was.
func TestApplySharpKeepsTheColour(t *testing.T) {
	r := rand.New(rand.NewSource(6))
	const n = 24
	f, inside := sharpFields(r, n, 8)
	image := midTensor(r, 4, n, n)
	light := func(t Tensor, c, p int) float64 { return float64(t.Plane(c)[p]+1) / 2 }
	plain, sharp := f.apply(image), f.applySharp(image, 2)
	var touched int
	for y := range n {
		for x := range n {
			p := y*n + x
			for _, pair := range [][2]int{{0, 1}, {1, 2}, {0, 2}} {
				a, b := pair[0], pair[1]
				was := light(plain, a, p) * light(sharp, b, p)
				is := light(sharp, a, p) * light(plain, b, p)
				if math.Abs(was-is) > 1e-6 {
					t.Fatalf("channels %d and %d changed ratio at %d,%d: %v against %v", a, b, y, x, was, is)
				}
			}
			if inside(y, x) && plain.Plane(0)[p] != sharp.Plane(0)[p] {
				touched++
			}
		}
	}
	if touched == 0 {
		t.Fatal("nothing changed inside the mask")
	}
}

// Steepening narrows the band where a mask is neither nothing nor
// everything: that band is the outline of what the morpher painted, and its
// width is what reads as a soft edge around the mouth.
func TestSteepenNarrowsTheRamp(t *testing.T) {
	const n = 64
	ramp := NewTensor(1, 1, n)
	for i := range n {
		ramp.Data[i] = float32(i) / float32(n-1)
	}
	band := func(x Tensor) int {
		wide := 0
		for _, v := range x.Data {
			if v > 0.01 && v < 0.99 {
				wide++
			}
		}
		return wide
	}
	was := band(ramp)
	for _, amount := range []float32{0.6, 1.2, 3} {
		got := band(steepen(ramp, amount))
		if want := int(float32(was) / steepenBy(amount)); got > want+2 || got < want-2 {
			t.Errorf("at %v the ramp is %d wide, want about %d", amount, got, want)
		}
	}
	// And it stops: an amount past the cap steepens no further, so the
	// edge cannot turn into steps.
	if a, b := band(steepen(ramp, 3)), band(steepen(ramp, 30)); a != b {
		t.Errorf("the ramp went on narrowing past the cap: %d then %d", a, b)
	}
}
