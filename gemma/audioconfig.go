package gemma

// What the projector file declares about the audio encoder.
//
// The same mmproj that carries the vision tower carries this one, under keys
// namespaced clip.audio.*. Two projectors exist and they share almost nothing:
// gemma4a is a twelve-block conformer, and gemma4ua is a single projection
// with no encoder at all behind it. Which one a file holds is the one thing
// this type is read for.
//
// As on the vision side, the numbers llama.cpp hardcodes are written down here
// rather than read, and each says where it comes from: they are what the
// PROJECTOR_TYPE_GEMMA4A and PROJECTOR_TYPE_GEMMA4UA arms of clip.cpp set, and
// a file that disagreed would run under llama.cpp with these values anyway.

import (
	"fmt"

	"github.com/ThiraSoft/golem/tensors"
)

type AudioConfig struct {
	Dim     int // clip.audio.embedding_length
	FFN     int // clip.audio.feed_forward_length
	Blocks  int // clip.audio.block_count
	Heads   int // clip.audio.attention.head_count
	HeadDim int // Dim / Heads
	MelBins int // clip.audio.num_mel_bins
	ProjDim int // clip.audio.projection_dim, the text model's width
	Eps     float32

	// Unified says the file is a gemma4ua projector: no conformer, no mel, no
	// blocks. The waveform is cut into frames of FrameSize samples, each frame
	// normalized and projected once, and every block those rows go through
	// afterwards belongs to the language model. The 12B ships one of those,
	// E2B a tower — exactly as on the vision side.
	Unified   bool
	FrameSize int // 640 samples, forty milliseconds, when Unified

	// The blocked attention's geometry. A query sees its own chunk of Chunk
	// frames and the Past frames before that chunk begins, so a window is
	// Context wide and RPE distances are distinguished within it.
	Chunk, Past, Context, RPE int

	// Softcap bounds the attention scores through a tanh before the mask is
	// added, as the language model's own attention does.
	Softcap float32
}

// HasAudio says whether a projector declares an audio encoder. A file that
// carries none — the 26B's does not — does not even hold the key.
func HasAudio(g *tensors.GGUF) bool {
	v, err := g.Bool("clip.has_audio_encoder")
	return err == nil && v
}

// LoadAudioConfig reads an opened mmproj GGUF.
func LoadAudioConfig(g *tensors.GGUF) (*AudioConfig, error) {
	arch, err := g.String("general.architecture")
	if err != nil {
		return nil, err
	}
	if arch != "clip" {
		return nil, fmt.Errorf("gemma: the projector declares architecture %q, not \"clip\"", arch)
	}
	if !HasAudio(g) {
		return nil, fmt.Errorf("gemma: this projector declares no audio encoder; the model behind it cannot listen")
	}
	proj, err := g.String("clip.audio.projector_type")
	if err != nil {
		return nil, err
	}
	if proj != "gemma4a" && proj != "gemma4ua" {
		return nil, fmt.Errorf("gemma: audio projector %q is not implemented; gemma4a and gemma4ua are", proj)
	}

	// clip.cpp sets eps to 1e-6 for both projectors and says why: the original
	// conversion wrote the wrong value into the file, every checkpoint uses
	// 1e-6, and hardcoding it was cheaper than reconverting the weights. So
	// the file's own epsilon is deliberately not read.
	cfg := &AudioConfig{
		Unified: proj == "gemma4ua",
		Eps:     1e-6,
	}

	dim, err := g.Uint32("clip.audio.projection_dim")
	if err != nil {
		return nil, err
	}
	cfg.ProjDim = int(dim)

	if cfg.Unified {
		// 640 samples at 16 kHz, forty milliseconds a token. The file's
		// clip.audio.num_mel_bins says 128 and is a leftover from the
		// conformer's conversion; clip.cpp overwrites it with 640 for this
		// projector, so it is not read here either.
		cfg.FrameSize = 640
		return cfg, nil
	}

	for _, f := range []struct {
		key string
		dst *int
	}{
		{"clip.audio.embedding_length", &cfg.Dim},
		{"clip.audio.feed_forward_length", &cfg.FFN},
		{"clip.audio.block_count", &cfg.Blocks},
		{"clip.audio.attention.head_count", &cfg.Heads},
		{"clip.audio.num_mel_bins", &cfg.MelBins},
	} {
		v, err := g.Uint32(f.key)
		if err != nil {
			return nil, err
		}
		*f.dst = int(v)
	}
	if cfg.Heads == 0 || cfg.Dim%cfg.Heads != 0 {
		return nil, fmt.Errorf("gemma: %d audio heads do not divide a width of %d", cfg.Heads, cfg.Dim)
	}
	cfg.HeadDim = cfg.Dim / cfg.Heads
	cfg.Chunk = 12
	cfg.Past = 12
	cfg.Context = cfg.Chunk + cfg.Past
	cfg.RPE = cfg.Past + 1
	cfg.Softcap = 50
	return cfg, nil
}
