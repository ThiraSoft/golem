package qwen35

import (
	"bytes"
	"fmt"
	"math"

	"github.com/ThiraSoft/golem/imageio"
	"github.com/ThiraSoft/golem/nn"
	"github.com/ThiraSoft/golem/tensors"
)

// VisionTower is a projector bound and ready to run.
type VisionTower struct {
	Cfg *VisionConfig
	W   *VisionWeights

	file *tensors.GGUF

	// trace keeps the intermediates a reference test stands on, under the
	// names models/qwen3vl.cpp gives its nodes. It is nil unless Trace was
	// called, because keeping them costs a copy of the grid per block.
	trace map[string][]float32
}

// Trace makes the next Encode keep its waypoints, under llama.cpp's own names
// for them. It is for the reference tests and costs a copy a block.
func (v *VisionTower) Trace() { v.trace = map[string][]float32{} }

// Waypoint is what the last traced Encode left under that name, or nil.
func (v *VisionTower) Waypoint(name string) []float32 { return v.trace[name] }

func (v *VisionTower) keep(name string, xs []float32) {
	if v.trace != nil {
		v.trace[name] = append([]float32(nil), xs...)
	}
}

// NewVisionTower binds a configuration to its weights.
func NewVisionTower(cfg *VisionConfig, w *VisionWeights) *VisionTower {
	return &VisionTower{Cfg: cfg, W: w}
}

// Close releases the projector's file.
func (v *VisionTower) Close() error {
	if v.file == nil {
		return nil
	}
	return v.file.Close()
}

// Encode runs the whole tower over one image and returns one row per output
// token, each as wide as the language model's embedding.
//
// The image is resized first, to a grid that holds its aspect ratio and stays
// inside the token budget; a caller that wants to know how many rows it will
// get can ask TargetSize and Tokens without running anything.
func (v *VisionTower) Encode(im *imageio.Image) [][]float32 {
	cfg, w := v.Cfg, v.W

	width, height := cfg.TargetSize(im.W, im.H)
	im = cfg.Fit(im, width, height)
	cols, rows := width/cfg.Patch, height/cfg.Patch

	at := cfg.PatchPositions(cols, rows)
	xs := cfg.PatchesTraced(w, im, v)
	s := newVisionScratch(cfg, len(at))
	for i := 0; i < cfg.Blocks; i++ {
		cfg.VisionBlockForward(w, i, xs, at, s)
		v.keep(fmt.Sprintf("layer_out-%d", i), xs)
	}

	// The tower's own last norm, then the merger.
	for p := 0; p < len(at); p++ {
		nn.LayerNormGGML(xs[p*cfg.Dim:(p+1)*cfg.Dim], w.PostLN.Gain, w.PostLN.Bias, cfg.Eps)
	}
	return v.merge(xs, len(at), s)
}

// Fit puts an image on the canvas the tower reads, the way img_tool::resize
// does under PAD_CEIL — which is the padding style every Qwen-VL projector
// leaves at its default, because its branch of clip.cpp sets the algorithm and
// not the padding.
//
// It is not a stretch to the target. The image is scaled by whichever of the
// two ratios is smaller, so its own shape is kept, and what is left over
// becomes a black border with the image centred in it. Stretching instead
// moves every pixel of a picture whose aspect ratio does not already match,
// which is most of them.
//
// The scale is float32 because it is float32 there, and the ceiling is taken
// of the product: at a ratio that lands within a rounding of a whole pixel the
// two widths differ by one, and one column of border is one column of patches.
func (cfg *VisionConfig) Fit(im *imageio.Image, width, height int) *imageio.Image {
	if im.W == width && im.H == height {
		return im
	}
	scale := float32(width) / float32(im.W)
	if s := float32(height) / float32(im.H); s < scale {
		scale = s
	}
	inner := min(int(math.Ceil(float64(float32(im.W)*scale))), width)
	tall := min(int(math.Ceil(float64(float32(im.H)*scale))), height)
	if im.W != inner || im.H != tall {
		im = im.ResizeBilinear(inner, tall)
	}
	if inner == width && tall == height {
		return im
	}
	return im.PadInto(width, height, [3]uint8{0, 0, 0})
}

// merge folds each square of Merge by Merge patches into one row of the
// language model's width.
//
// Four consecutive rows are that square — which is what the patch order was
// arranged for — so the merger is a concatenation and two projections with a
// GELU between them, and no gathering at all.
func (v *VisionTower) merge(xs []float32, patches int, s *visionScratch) [][]float32 {
	cfg, w := v.Cfg, v.W
	group := cfg.Merge * cfg.Merge
	wide := cfg.Dim * group

	out := make([][]float32, patches/group)
	mid := make([]float32, wide)
	for t := range out {
		copy(s.merged, xs[t*wide:(t+1)*wide])
		w.MM0.Apply(s.merged, mid)
		nn.GELUTable(mid)
		row := make([]float32, cfg.ProjDim)
		w.MM2.Apply(mid, row)
		out[t] = row
	}
	return out
}

// OpenProjector maps a projector file and binds its tower to this model.
//
// It is an error when the file declares no vision encoder, when its projector
// is not the one this engine implements, or when its projection does not land
// on the model's own width — each said at open rather than left to a wrong
// answer twenty-seven blocks later.
func (m *Model) OpenProjector(path string) error {
	g, err := tensors.OpenGGUF(path)
	if err != nil {
		return err
	}
	cfg, err := LoadVisionConfig(g)
	if err != nil {
		g.Close()
		return err
	}
	if cfg.ProjDim != m.Cfg.Dim {
		g.Close()
		return fmt.Errorf("qwen35: the projector lands on %d, and this model is %d wide", cfg.ProjDim, m.Cfg.Dim)
	}
	w, err := LoadVisionWeights(g, cfg)
	if err != nil {
		g.Close()
		return err
	}
	m.vision = &VisionTower{Cfg: cfg, W: w, file: g}
	return nil
}

// HasVision reports whether a projector has been opened.
func (m *Model) HasVision() bool { return m.vision != nil }

// Vision is the tower, for a caller that wants its geometry.
func (m *Model) Vision() *VisionTower { return m.vision }

// EncodeImage decodes one encoded image and runs the tower over it. What comes
// back is one row per output token, each as wide as the model's embedding.
//
// The grid each picture came from is kept beside the rows, because a row count
// does not say what shape it was — two hundred and sixty rows is twenty by
// thirteen or thirteen by twenty, and the positions differ. GridOf reads them
// back in the order they were encoded, and BuildPrompt clears the list once it
// has used them.
func (m *Model) EncodeImage(data []byte) ([][]float32, error) {
	if m.vision == nil {
		return nil, fmt.Errorf("qwen35: this model was opened without a projector")
	}
	im, err := imageio.Decode(bytes.NewReader(data))
	if err != nil {
		return nil, err
	}
	cfg := m.vision.Cfg
	width, height := cfg.TargetSize(im.W, im.H)
	rows := m.vision.Encode(im)
	m.grids = append(m.grids, [2]int{width / cfg.Patch / cfg.Merge, height / cfg.Patch / cfg.Merge})
	return rows, nil
}

// GridOf is the merged grid the i-th picture encoded since the last prompt was
// built came from.
func (m *Model) GridOf(i int) ([2]int, bool) {
	if i < 0 || i >= len(m.grids) {
		return [2]int{}, false
	}
	return m.grids[i], true
}
