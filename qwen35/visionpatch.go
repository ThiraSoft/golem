package qwen35

import (
	"math"

	"github.com/ThiraSoft/golem/imageio"
	"github.com/ThiraSoft/golem/nn"
)

// TargetSize is the size an image is resized to before it is cut into patches.
//
// It is the "smart resize" of the reference implementation, transcribed from
// img_tool::calc_size_preserved_ratio in llama.cpp's mtmd-image.cpp: both sides
// are rounded to a whole number of merged patches, and if the area then falls
// outside the token budget the whole thing is scaled by the square root of how
// far out it is and aligned again — down by flooring, up by ceiling.
//
// The aspect ratio is held throughout, which is the point: this tower has no
// fixed input size, and a letterboxed square would hand it a border it was
// never trained to ignore.
func (cfg *VisionConfig) TargetSize(w, h int) (int, int) {
	align := cfg.Patch * cfg.Merge
	area := cfg.Patch * cfg.Patch * cfg.Merge * cfg.Merge
	minPixels := cfg.MinTokens * area
	maxPixels := cfg.MaxTokens * area

	round := func(x float64) int { return int(math.Round(x/float64(align))) * align }
	ceil := func(x float64) int { return int(math.Ceil(x/float64(align))) * align }
	floor := func(x float64) int { return int(math.Floor(x/float64(align))) * align }

	// Always align first, then correct if that put the area out of bounds.
	wBar := max(align, round(float64(w)))
	hBar := max(align, round(float64(h)))

	switch {
	case hBar*wBar > maxPixels:
		beta := math.Sqrt(float64(h) * float64(w) / float64(maxPixels))
		wBar = max(align, floor(float64(w)/beta))
		hBar = max(align, floor(float64(h)/beta))
	case hBar*wBar < minPixels:
		beta := math.Sqrt(float64(minPixels) / (float64(h) * float64(w)))
		wBar = ceil(float64(w) * beta)
		hBar = ceil(float64(h) * beta)
	}
	return wBar, hBar
}

// PatchOrder maps each slot of the reordered grid to the raster index of the
// patch that goes in it.
//
// Four consecutive slots are one square of Merge by Merge, which is what makes
// the merger's four consecutive rows a neighbourhood rather than a row of the
// image. The graph does this with reshapes and a permute; an index map is the
// same arithmetic said once.
func (cfg *VisionConfig) PatchOrder(cols, rows int) []int {
	out := make([]int, 0, cols*rows)
	for y := 0; y < rows; y += cfg.Merge {
		for x := 0; x < cols; x += cfg.Merge {
			for dy := 0; dy < cfg.Merge; dy++ {
				for dx := 0; dx < cfg.Merge; dx++ {
					out = append(out, (y+dy)*cols+(x+dx))
				}
			}
		}
	}
	return out
}

// PatchPositions is each slot's row and column, which is what the rotation
// turns it by. It walks the same loop clip.cpp fills its position tensor from,
// so the two orders cannot drift apart.
func (cfg *VisionConfig) PatchPositions(cols, rows int) [][2]int {
	out := make([][2]int, 0, cols*rows)
	for y := 0; y < rows; y += cfg.Merge {
		for x := 0; x < cols; x += cfg.Merge {
			for dy := 0; dy < cfg.Merge; dy++ {
				for dx := 0; dx < cfg.Merge; dx++ {
					out = append(out, [2]int{y + dy, x + dx})
				}
			}
		}
	}
	return out
}

// Positions is the learned table resized to this grid and put in the slot
// order, which is what a patch's embedding has added to it.
//
// A grid that is already the table's own side takes the table untouched —
// clip.cpp short-circuits there, and so does this, because an interpolation
// that should be the identity is not exactly one.
func (cfg *VisionConfig) Positions(w *VisionWeights, cols, rows int) []float32 {
	table := w.PosEmbd
	if cols != cfg.PosSide || rows != cfg.PosSide {
		// The table is stored row-major by position — entry p at p*Dim — and
		// the resize wants it channel-major, one plane a channel.
		planes := make([]float32, cfg.Dim*cfg.PosSide*cfg.PosSide)
		for p := 0; p < cfg.PosSide*cfg.PosSide; p++ {
			for d := 0; d < cfg.Dim; d++ {
				planes[d*cfg.PosSide*cfg.PosSide+p] = table[p*cfg.Dim+d]
			}
		}
		resized := nn.ResizeBilinearAntialias(planes, cfg.PosSide, cfg.PosSide, cols, rows, cfg.Dim)
		table = make([]float32, cfg.Dim*cols*rows)
		for p := 0; p < cols*rows; p++ {
			for d := 0; d < cfg.Dim; d++ {
				table[p*cfg.Dim+d] = resized[d*cols*rows+p]
			}
		}
	}
	order := cfg.PatchOrder(cols, rows)
	out := make([]float32, len(order)*cfg.Dim)
	for slot, raster := range order {
		copy(out[slot*cfg.Dim:(slot+1)*cfg.Dim], table[raster*cfg.Dim:(raster+1)*cfg.Dim])
	}
	return out
}

// Patches cuts the image into patches, projects each of them, and adds the
// bias and the positions. What comes back is the grid in slot order, flat:
// cols*rows rows of Dim.
//
// The projection is two convolutions summed. llama.cpp's tensor is a Conv3d of
// temporal depth two split into two Conv2d weights, and a still image is the
// same frame twice — so both meet the same pixels and the sum is what a video
// of one frame would give.
func (cfg *VisionConfig) Patches(w *VisionWeights, im *imageio.Image) []float32 {
	cols, rows := im.W/cfg.Patch, im.H/cfg.Patch
	taps := cfg.Patch * cfg.Patch * 3
	order := cfg.PatchOrder(cols, rows)

	// The pixels, planar and scaled the way the projector's mean and deviation
	// ask — both are 0.5 here, so this is 2x-1 over 0..1.
	pixels := make([]float32, 3*im.W*im.H)
	im.PlanarRGB(pixels)
	for i := range pixels {
		pixels[i] = pixels[i]*2 - 1
	}

	out := make([]float32, len(order)*cfg.Dim)
	patch := nn.NewBatch(taps, 1)
	gathered := make([]float32, taps)
	rowA := make([]float32, cfg.Dim)
	rowB := make([]float32, cfg.Dim)

	for slot, raster := range order {
		py, px := raster/cols, raster%cols
		gatherPatch(gathered, pixels, im.W, im.H, px, py, cfg.Patch)
		patch.Set(0, gathered)
		w.PatchA.MatVec(patch, rowA)
		w.PatchB.MatVec(patch, rowB)
		dst := out[slot*cfg.Dim : (slot+1)*cfg.Dim]
		for d := range dst {
			dst[d] = rowA[d] + rowB[d] + w.PatchBias[d]
		}
	}

	positions := cfg.Positions(w, cols, rows)
	for i := range out {
		out[i] += positions[i]
	}
	return out
}

// gatherPatch lays one patch out in the order the weight was written: x
// fastest, then y, then the channel — which is ggml's im2col, and what its
// convolution reads.
func gatherPatch(dst, pixels []float32, w, h, px, py, size int) {
	plane := w * h
	i := 0
	for c := 0; c < 3; c++ {
		for y := 0; y < size; y++ {
			row := (py*size+y)*w + px*size
			for x := 0; x < size; x++ {
				dst[i] = pixels[c*plane+row+x]
				i++
			}
		}
	}
}
