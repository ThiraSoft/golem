package qwen35

import (
	"fmt"
	"math"

	"github.com/ThiraSoft/golem/tensors"
)

// VisionConfig is the geometry of the projector, read from its own GGUF rather
// than assumed. A projector is a second file beside the weights, and this
// engine refuses one it was not built for by name instead of reading it into
// the wrong shapes.
type VisionConfig struct {
	Blocks  int
	Dim     int // 1152
	Heads   int // 16
	HeadDim int // 72
	FFN     int // 4304
	Patch   int // 16
	Merge   int // 2, the side of the square the merger folds
	ProjDim int // 5120, which is the text model's width

	// PosSide is the side of the learned position table, which is square. The
	// file carries the table and not its side, so this is read off its shape.
	PosSide int

	Eps float32

	// MinTokens and MaxTokens bound the output token count, and with it the
	// grid an image is resized to. The file does not carry them: they are
	// hparams.set_limit_image_tokens(8, 4096) in clip.cpp, for every Qwen-VL
	// projector.
	MinTokens int
	MaxTokens int
}

// HasVision reports whether a file declares a vision encoder at all.
func HasVision(g *tensors.GGUF) bool {
	v, err := g.Bool("clip.has_vision_encoder")
	return err == nil && v
}

// LoadVisionConfig reads an opened projector GGUF.
func LoadVisionConfig(g *tensors.GGUF) (*VisionConfig, error) {
	if !HasVision(g) {
		return nil, fmt.Errorf("qwen35: this file declares no vision encoder")
	}
	proj, err := g.String("clip.projector_type")
	if err != nil {
		return nil, err
	}
	if proj != "qwen3vl_merger" {
		return nil, fmt.Errorf("qwen35: projector type %q is not qwen3vl_merger", proj)
	}

	u := func(key string) (int, error) {
		v, err := g.Uint32(key)
		return int(v), err
	}
	cfg := &VisionConfig{MinTokens: 8, MaxTokens: 4096}
	for _, f := range []struct {
		key  string
		into *int
	}{
		{"clip.vision.block_count", &cfg.Blocks},
		{"clip.vision.embedding_length", &cfg.Dim},
		{"clip.vision.attention.head_count", &cfg.Heads},
		{"clip.vision.feed_forward_length", &cfg.FFN},
		{"clip.vision.patch_size", &cfg.Patch},
		{"clip.vision.projection_dim", &cfg.ProjDim},
	} {
		v, err := u(f.key)
		if err != nil {
			return nil, err
		}
		*f.into = v
	}
	if cfg.Heads == 0 || cfg.Dim%cfg.Heads != 0 {
		return nil, fmt.Errorf("qwen35: %d heads do not divide a width of %d", cfg.Heads, cfg.Dim)
	}
	cfg.HeadDim = cfg.Dim / cfg.Heads

	// The merge defaults to two, which is what every Qwen-VL projector has
	// used; the file names it when it disagrees.
	cfg.Merge = 2
	if v, err := u("clip.vision.spatial_merge_size"); err == nil {
		cfg.Merge = v
	}
	eps, err := g.Float32("clip.vision.attention.layer_norm_epsilon")
	if err != nil {
		return nil, err
	}
	cfg.Eps = eps

	// The position table is square and the file gives its area, not its side.
	pos, err := g.Get("v.position_embd.weight")
	if err != nil {
		return nil, err
	}
	if len(pos.Shape) != 2 || pos.Shape[0] != cfg.Dim {
		return nil, fmt.Errorf("qwen35: the position table is %v, not a grid of %d-wide rows", pos.Shape, cfg.Dim)
	}
	side := int(math.Round(math.Sqrt(float64(pos.Shape[1]))))
	if side*side != pos.Shape[1] {
		return nil, fmt.Errorf("qwen35: the position table holds %d entries, which is not a square", pos.Shape[1])
	}
	cfg.PosSide = side

	return cfg, nil
}

// Tokens is how many rows an image of this patch grid becomes: the merger
// folds a square of Merge by Merge patches into one.
func (cfg *VisionConfig) Tokens(cols, rows int) int {
	return (cols / cfg.Merge) * (rows / cfg.Merge)
}
