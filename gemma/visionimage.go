package gemma

// From a picture to the numbers the tower reads.
//
// Gemma 4 has no fixed input size. clip.vision.image_size says 224 and means
// nothing here: the image is scaled so that the number of tokens it becomes
// lands inside the range the model was trained for, rounded to a whole number
// of pooled patches so the 3x3 pooling divides, and its shape is preserved
// exactly — what the rounding leaves over becomes a black border rather than a
// stretch.
//
// The rule is llama.cpp's, in two parts: calc_size_preserved_ratio in
// mtmd-image.cpp decides the canvas, and img_tool::resize under the PAD_CEIL
// style, which gemma4v does not override, fills it. This is the step it is
// easiest to get quietly wrong, and being wrong here moves every activation
// after it, so it is compared against the reference's own pixels before any
// block is.

import (
	"math"

	"github.com/ThiraSoft/golem/imageio"
)

// padColour is what the border is filled with: clip's image_pad_color, which
// gemma4v leaves at black.
var padColour = [3]uint8{0, 0, 0}

// TargetSize is the canvas an image of w by h is resized onto: a multiple of
// PatchSize*Merge on each side, holding between MinTokens and MaxTokens pooled
// patches.
func (cfg *VisionConfig) TargetSize(w, h int) (int, int) {
	unit := cfg.PatchSize * cfg.Merge
	if w < 1 {
		w = 1
	}
	if h < 1 {
		h = 1
	}
	area := unit * unit
	minPixels, maxPixels := cfg.MinTokens*area, cfg.MaxTokens*area

	// Align up first, then pull the whole canvas back inside the budget it
	// overshot. The two branches are exclusive: a canvas cannot be both too
	// large and too small.
	wBar := maxInt(unit, roundTo(float64(w), unit))
	hBar := maxInt(unit, roundTo(float64(h), unit))
	switch {
	case wBar*hBar > maxPixels:
		beta := math.Sqrt(float64(w) * float64(h) / float64(maxPixels))
		wBar = maxInt(unit, floorTo(float64(w)/beta, unit))
		hBar = maxInt(unit, floorTo(float64(h)/beta, unit))
	case wBar*hBar < minPixels:
		beta := math.Sqrt(float64(minPixels) / (float64(w) * float64(h)))
		wBar = ceilTo(float64(w)*beta, unit)
		hBar = ceilTo(float64(h)*beta, unit)
	}
	return wBar, hBar
}

// Fit resizes onto that canvas the way the reference does: the shape is kept
// exactly, the scaled image is centred, and what is left over is the border.
func (cfg *VisionConfig) Fit(im *imageio.Image) *imageio.Image {
	tw, th := cfg.TargetSize(im.W, im.H)
	scale := math.Min(float64(tw)/float64(im.W), float64(th)/float64(im.H))
	w := minInt(int(math.Ceil(float64(im.W)*scale)), tw)
	h := minInt(int(math.Ceil(float64(im.H)*scale)), th)
	scaled := im.ResizeBilinear(w, h)
	if w == tw && h == th {
		return scaled
	}
	return scaled.PadInto(tw, th, padColour)
}

// Prepare lays the fitted image out as the tower reads it: the red plane, then
// the green, then the blue, each value in 0..1. The scaling to -1..1 belongs to
// the forward pass, which is where llama.cpp does it.
//
// cols and rows are the patch grid before pooling.
func (cfg *VisionConfig) Prepare(im *imageio.Image) (pixels []float32, cols, rows int) {
	fitted := cfg.Fit(im)
	pixels = make([]float32, 3*fitted.W*fitted.H)
	fitted.PlanarRGB(pixels)
	return pixels, fitted.W / cfg.PatchSize, fitted.H / cfg.PatchSize
}

func roundTo(x float64, unit int) int { return int(math.Round(x/float64(unit))) * unit }
func floorTo(x float64, unit int) int { return int(math.Floor(x/float64(unit))) * unit }
func ceilTo(x float64, unit int) int  { return int(math.Ceil(x/float64(unit))) * unit }

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}
