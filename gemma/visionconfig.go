package gemma

// What the projector file declares about the vision tower.
//
// The mmproj is a GGUF of its own, whose general.architecture is "clip" and
// whose keys are namespaced clip.vision.*. Three numbers it does not carry are
// written down here rather than read, and each says where it comes from: they
// are what llama.cpp hardcodes in the PROJECTOR_TYPE_GEMMA4V arm of clip.cpp,
// and a file that disagreed with them would run under llama.cpp with the same
// values.

import (
	"fmt"

	"github.com/ThiraSoft/golem/tensors"
)

type VisionConfig struct {
	Dim       int // clip.vision.embedding_length
	FFN       int // clip.vision.feed_forward_length
	Blocks    int // clip.vision.block_count
	Heads     int // clip.vision.attention.head_count
	HeadDim   int // Dim / Heads
	PatchSize int // clip.vision.patch_size
	ProjDim   int // clip.vision.projection_dim, the text model's width
	Eps       float32
	// Merge is the side of the average-pooling window between the tower and
	// the projector. clip.cpp: hparams.n_merge = 3 for gemma4v.
	Merge int
	// RoPEBase is 100 for gemma4v, from the same place. It is not the text
	// model's base and has no reason to be.
	RoPEBase float64
	// MinTokens and MaxTokens bound how many tokens one image becomes, which
	// is what decides the size it is resized to. clip.cpp:
	// hparams.set_limit_image_tokens(40, 280).
	MinTokens, MaxTokens int
	// Unified says the file is a gemma4uv projector rather than a gemma4v
	// one: an embedder with no tower behind it, whose blocks are the language
	// model's own. The 12B ships one of those, E2B a tower. Everything the two
	// share — the sizing of the image, the markers, the span attended to in
	// both directions — is the same; what one has and the other does not is
	// sixteen blocks and a pooling.
	Unified bool
}

// LoadVisionConfig reads an opened mmproj GGUF.
func LoadVisionConfig(g *tensors.GGUF) (*VisionConfig, error) {
	arch, err := g.String("general.architecture")
	if err != nil {
		return nil, err
	}
	if arch != "clip" {
		return nil, fmt.Errorf("gemma: the projector declares architecture %q, not \"clip\"", arch)
	}
	proj, err := g.String("clip.vision.projector_type")
	if err != nil {
		return nil, err
	}
	if proj != "gemma4v" && proj != "gemma4uv" {
		return nil, fmt.Errorf("gemma: projector %q is not implemented; gemma4v and gemma4uv are", proj)
	}
	cfg := &VisionConfig{
		Unified:   proj == "gemma4uv",
		Merge:     3,
		RoPEBase:  100,
		MinTokens: 40,
		MaxTokens: 280,
	}
	for _, f := range []struct {
		key string
		dst *int
	}{
		{"clip.vision.embedding_length", &cfg.Dim},
		{"clip.vision.feed_forward_length", &cfg.FFN},
		{"clip.vision.block_count", &cfg.Blocks},
		{"clip.vision.attention.head_count", &cfg.Heads},
		{"clip.vision.patch_size", &cfg.PatchSize},
		{"clip.vision.projection_dim", &cfg.ProjDim},
	} {
		v, err := g.Uint32(f.key)
		if err != nil {
			return nil, err
		}
		*f.dst = int(v)
	}
	if cfg.Unified {
		// The merging of three by three patches is done by the convolution
		// itself, on a patch three times as wide; nothing is pooled
		// afterwards. clip.cpp folds it the same way, in the GEMMA4UV arm of
		// the same switch the three constants above come from.
		cfg.PatchSize *= cfg.Merge
		cfg.Merge = 1
		eps, err := g.Float32("clip.vision.attention.layer_norm_epsilon")
		if err != nil {
			return nil, err
		}
		cfg.Eps = eps
		return cfg, nil
	}
	if cfg.Heads == 0 || cfg.Dim%cfg.Heads != 0 {
		return nil, fmt.Errorf("gemma: %d vision heads do not divide a width of %d", cfg.Heads, cfg.Dim)
	}
	cfg.HeadDim = cfg.Dim / cfg.Heads
	if cfg.HeadDim%4 != 0 {
		// The head is split in two and each half is rotated in pairs, so it
		// has to divide by four for the geometry to exist at all.
		return nil, fmt.Errorf("gemma: a vision head of %d cannot carry a 2D rotation", cfg.HeadDim)
	}
	eps, err := g.Float32("clip.vision.attention.layer_norm_epsilon")
	if err != nil {
		return nil, err
	}
	cfg.Eps = eps
	return cfg, nil
}
