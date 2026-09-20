package krea2

import (
	"fmt"
	"math"
	"strconv"
	"strings"
)

// A picture drawn on a colour of one's choosing, to be keyed out afterwards:
// TKG-DM's channel mean shift (arXiv 2411.15580). The initial noise is
// Gaussian everywhere, but away from the middle of the frame its per-channel
// mean is moved towards the latent of the wanted colour, and the model draws
// the background it was nudged towards. Nothing is trained and no weight is
// touched: only the noise the sampler starts from changes.

// chromaSpread is how wide the middle the shift spares is, as a share of the
// half-diagonal: the subject is drawn in it and keeps plain Gaussian noise.
const chromaSpread = 0.4

// chromaProbe is the side, in latent pixels, of the picture the decoder is
// asked for when the colour's latent is looked for: large enough that the
// middle is clear of the edges the decoder makes.
const chromaProbe = 8

// chromaStep is how far a channel is moved when the decoder's answer is
// measured, in the sampler's units, where the latent is a standard normal.
const chromaStep = 1

// chromaSteps is how many times the shift is corrected, and chromaClose how
// near the colour it must land for the correcting to stop.
const (
	chromaSteps = 6
	chromaClose = 0.01
)

// Colour is a colour in [0, 1], the decoder's own range.
type Colour [3]float32

// ParseColour reads #rrggbb, with or without the hash.
func ParseColour(s string) (Colour, error) {
	h := strings.TrimPrefix(strings.TrimSpace(s), "#")
	if len(h) != 6 {
		return Colour{}, fmt.Errorf("krea2: %q is not a colour: want #rrggbb", s)
	}
	var col Colour
	for i := range col {
		v, err := strconv.ParseUint(h[2*i:2*i+2], 16, 8)
		if err != nil {
			return Colour{}, fmt.Errorf("krea2: %q is not a colour: %w", s, err)
		}
		col[i] = float32(v) / 255
	}
	return col, nil
}

// ChromaLatent is the shift, in the sampler's units, whose decode is col. Of
// the many shifts that land on the colour it answers the shortest: the noise
// is moved as little as the colour allows. The decoder is not quite a
// straight line, so the slope one probe a channel gives is used to step
// towards the colour rather than to reach it in one go.
func (v *VAE) ChromaLatent(col Colour) ([16]float32, error) {
	flat, err := v.flatColour([16]float32{})
	if err != nil {
		return [16]float32{}, err
	}
	var slope [3][16]float32
	for c := 0; c < 16; c++ {
		var at [16]float32
		at[c] = chromaStep
		got, err := v.flatColour(at)
		if err != nil {
			return [16]float32{}, err
		}
		for i := 0; i < 3; i++ {
			slope[i][c] = (got[i] - flat[i]) / chromaStep
		}
	}
	var d [16]float32
	got := flat
	for step := 0; step < chromaSteps; step++ {
		var miss Colour
		worst := float32(0)
		for i := range miss {
			miss[i] = col[i] - got[i]
			worst = max(worst, abs(miss[i]))
		}
		if worst < chromaClose {
			break
		}
		towards, err := shortestShift(slope, miss)
		if err != nil {
			return [16]float32{}, err
		}
		for c := range d {
			d[c] += towards[c]
		}
		if got, err = v.flatColour(d); err != nil {
			return [16]float32{}, err
		}
	}
	return d, nil
}

func abs(x float32) float32 {
	if x < 0 {
		return -x
	}
	return x
}

// flatColour is what the decoder makes of a latent that holds one value a
// channel: the colour in the middle of the picture, where the edges the
// decoder draws do not reach.
func (v *VAE) flatColour(at [16]float32) (Colour, error) {
	n := chromaProbe * chromaProbe
	latent := make([]float32, 16*n)
	for c := 0; c < 16; c++ {
		for i := c * n; i < (c+1)*n; i++ {
			latent[i] = at[c]
		}
	}
	LatentOut(latent)
	rgb, err := v.Decode(latent, chromaProbe, chromaProbe)
	if err != nil {
		return Colour{}, err
	}
	return middle(rgb, 8*chromaProbe), nil
}

// middle is the mean colour of the middle quarter of a square picture.
func middle(rgb []float32, side int) Colour {
	var sum [3]float64
	var n float64
	for y := side / 4; y < 3*side/4; y++ {
		for x := side / 4; x < 3*side/4; x++ {
			at := 3 * (y*side + x)
			for c := 0; c < 3; c++ {
				sum[c] += float64(rgb[at+c])
			}
			n++
		}
	}
	var col Colour
	for c := range col {
		col[c] = float32(sum[c] / n)
	}
	return col
}

// shortestShift is the smallest d with slope · d = want: dᵀ(ddᵀ)⁻¹ want,
// three equations for sixteen unknowns. The three colours a channel moves
// are never quite the same, so the little matrix is invertible, but a
// decoder that ignored colour would make it singular and that is an error
// rather than a silent zero.
func shortestShift(slope [3][16]float32, want Colour) ([16]float32, error) {
	var g [3][3]float64 // slope · slopeᵀ
	for i := 0; i < 3; i++ {
		for j := 0; j < 3; j++ {
			var s float64
			for c := 0; c < 16; c++ {
				s += float64(slope[i][c]) * float64(slope[j][c])
			}
			g[i][j] = s
		}
	}
	det := g[0][0]*(g[1][1]*g[2][2]-g[1][2]*g[2][1]) -
		g[0][1]*(g[1][0]*g[2][2]-g[1][2]*g[2][0]) +
		g[0][2]*(g[1][0]*g[2][1]-g[1][1]*g[2][0])
	if math.Abs(det) < 1e-12 {
		return [16]float32{}, fmt.Errorf("krea2: the decoder's slopes do not tell the three colours apart")
	}
	inv := [3][3]float64{}
	inv[0][0] = (g[1][1]*g[2][2] - g[1][2]*g[2][1]) / det
	inv[0][1] = (g[0][2]*g[2][1] - g[0][1]*g[2][2]) / det
	inv[0][2] = (g[0][1]*g[1][2] - g[0][2]*g[1][1]) / det
	inv[1][0] = (g[1][2]*g[2][0] - g[1][0]*g[2][2]) / det
	inv[1][1] = (g[0][0]*g[2][2] - g[0][2]*g[2][0]) / det
	inv[1][2] = (g[0][2]*g[1][0] - g[0][0]*g[1][2]) / det
	inv[2][0] = (g[1][0]*g[2][1] - g[1][1]*g[2][0]) / det
	inv[2][1] = (g[0][1]*g[2][0] - g[0][0]*g[2][1]) / det
	inv[2][2] = (g[0][0]*g[1][1] - g[0][1]*g[1][0]) / det
	var y [3]float64
	for i := 0; i < 3; i++ {
		for j := 0; j < 3; j++ {
			y[i] += inv[i][j] * float64(want[j])
		}
	}
	var d [16]float32
	for c := 0; c < 16; c++ {
		var s float64
		for i := 0; i < 3; i++ {
			s += float64(slope[i][c]) * y[i]
		}
		d[c] = float32(s)
	}
	return d, nil
}

// ChromaNoise moves the initial latent's background towards the colour whose
// shift is d, by strength of it. The middle of the frame keeps the noise it
// was given, and the corners take the whole shift, with a Gaussian of
// spread between them; a spread of zero means chromaSpread.
func ChromaNoise(x []float32, h, w int, d [16]float32, strength, spread float32) {
	if spread <= 0 {
		spread = chromaSpread
	}
	plane := h * w
	cy, cx := float64(h-1)/2, float64(w-1)/2
	// The spread is a share of the half-diagonal, so that a wide picture
	// spares as much of its middle as a tall one.
	half := math.Hypot(cy, cx)
	den := 2 * math.Pow(float64(spread)*half, 2)
	for y := 0; y < h; y++ {
		for i := 0; i < w; i++ {
			r2 := math.Pow(float64(y)-cy, 2) + math.Pow(float64(i)-cx, 2)
			back := float32(1 - math.Exp(-r2/den))
			at := y*w + i
			for c := 0; c < 16; c++ {
				x[c*plane+at] += back * strength * d[c]
			}
		}
	}
}
