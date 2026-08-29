package qwen35

import (
	"bytes"
	"fmt"

	"github.com/ThiraSoft/golem/imageio"
	"github.com/ThiraSoft/golem/nn"
	"github.com/ThiraSoft/golem/tensors"
)

// VisionTower is a projector bound and ready to run.
type VisionTower struct {
	Cfg *VisionConfig
	W   *VisionWeights

	file *tensors.GGUF
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
	if im.W != width || im.H != height {
		im = im.ResizeBilinear(width, height)
	}
	cols, rows := width/cfg.Patch, height/cfg.Patch

	xs := cfg.Patches(w, im)
	at := cfg.PatchPositions(cols, rows)
	s := newVisionScratch(cfg, len(at))
	for i := 0; i < cfg.Blocks; i++ {
		cfg.VisionBlockForward(w, i, xs, at, s)
	}

	// The tower's own last norm, then the merger.
	for p := 0; p < len(at); p++ {
		nn.LayerNormGGML(xs[p*cfg.Dim:(p+1)*cfg.Dim], w.PostLN.Gain, w.PostLN.Bias, cfg.Eps)
	}
	return v.merge(xs, len(at), s)
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
func (m *Model) EncodeImage(data []byte) ([][]float32, error) {
	if m.vision == nil {
		return nil, fmt.Errorf("qwen35: this model was opened without a projector")
	}
	im, err := imageio.Decode(bytes.NewReader(data))
	if err != nil {
		return nil, err
	}
	return m.vision.Encode(im), nil
}
