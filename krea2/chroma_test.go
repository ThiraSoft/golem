package krea2

import (
	"math"
	"testing"
)

func TestParseColour(t *testing.T) {
	for _, s := range []string{"#00ff7f", "00ff7f"} {
		col, err := ParseColour(s)
		if err != nil {
			t.Fatalf("%s: %v", s, err)
		}
		want := Colour{0, 1, 127.0 / 255}
		for c := range want {
			if math.Abs(float64(col[c]-want[c])) > 1e-6 {
				t.Fatalf("%s: %v, want %v", s, col, want)
			}
		}
	}
	for _, s := range []string{"", "#fff", "#00ff7g", "green"} {
		if _, err := ParseColour(s); err == nil {
			t.Fatalf("%q was read as a colour", s)
		}
	}
}

// The shortest shift lands on the colour it was asked for, and is shorter
// than a shift that lands on it by moving one channel alone.
func TestShortestShiftLandsOnTheColour(t *testing.T) {
	var slope [3][16]float32
	for c := 0; c < 16; c++ {
		slope[0][c] = float32(c%3) / 3
		slope[1][c] = float32((c+1)%5) / 5
		slope[2][c] = float32(c%7)/7 - 0.2
	}
	want := Colour{0.1, -0.4, 0.25}
	d, err := shortestShift(slope, want)
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 3; i++ {
		var got float32
		for c := 0; c < 16; c++ {
			got += slope[i][c] * d[c]
		}
		if math.Abs(float64(got-want[i])) > 1e-4 {
			t.Fatalf("channel %d lands on %g, want %g", i, got, want[i])
		}
	}
}

// A decoder whose channels all say the same thing about colour cannot be
// asked for one, and says so.
func TestShortestShiftNeedsThreeColours(t *testing.T) {
	var slope [3][16]float32
	for c := 0; c < 16; c++ {
		slope[0][c], slope[1][c], slope[2][c] = 1, 1, 1
	}
	if _, err := shortestShift(slope, Colour{1, 0, 0}); err == nil {
		t.Fatal("a colourblind decoder was accepted")
	}
}

// The noise keeps its Gaussian mean in the middle, where the subject is
// drawn, and takes the whole shift in the corners.
func TestChromaNoiseSparesTheMiddle(t *testing.T) {
	const h, w = 32, 32
	x := make([]float32, 16*h*w)
	var d [16]float32
	d[5] = 2
	ChromaNoise(x, h, w, d, 1, 0)
	at := func(y, i int) float32 { return x[5*h*w+y*w+i] }
	if mid := at(h/2, w/2); math.Abs(float64(mid)) > 0.05 {
		t.Fatalf("the middle moved by %g", mid)
	}
	// The corner is a spread and a half out, so it takes nearly all of it.
	if corner := at(0, 0); corner < 0.9*d[5] || corner > d[5] {
		t.Fatalf("the corner moved by %g, want most of %g", corner, d[5])
	}
	// A channel with no shift of its own stays where it was.
	if v := x[0]; v != 0 {
		t.Fatalf("an unshifted channel moved by %g", v)
	}
}

// Half the strength moves the noise half as far.
func TestChromaNoiseStrength(t *testing.T) {
	const h, w = 16, 16
	var d [16]float32
	d[0] = 1
	full := make([]float32, 16*h*w)
	half := make([]float32, 16*h*w)
	ChromaNoise(full, h, w, d, 1, 0)
	ChromaNoise(half, h, w, d, 0.5, 0)
	for i := range full {
		if math.Abs(float64(full[i]/2-half[i])) > 1e-6 {
			t.Fatalf("at %d: %g and %g", i, full[i], half[i])
		}
	}
}

// The shift the decoder gives for a colour decodes back to that colour: the
// straight line the probe assumes holds as far as a saturated green.
func TestChromaLatentDecodesToTheColour(t *testing.T) {
	needFile(t, VAEPath())
	v, err := OpenVAE(device(t), VAEPath(), chromaProbe*chromaProbe)
	if err != nil {
		t.Fatal(err)
	}
	defer v.Close()
	want, err := ParseColour("#00c05a")
	if err != nil {
		t.Fatal(err)
	}
	d, err := v.ChromaLatent(want)
	if err != nil {
		t.Fatal(err)
	}
	got, err := v.flatColour(d)
	if err != nil {
		t.Fatal(err)
	}
	for c := range want {
		if math.Abs(float64(got[c]-want[c])) > 0.06 {
			t.Fatalf("the shift %v decodes to %v, want %v", d, got, want)
		}
	}
}
