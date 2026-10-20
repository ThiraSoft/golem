package yue2

import (
	"encoding/json"
	"fmt"

	"github.com/ThiraSoft/golem/tensors"
)

// Special token IDs used by YuE2 across the score, code, and audio generation stages.
const (
	EOD         = 151643
	ABCStart    = 151847
	ABCEnd      = 151848
	MusicStart  = 151851
	MusicEnd    = 151852
	CodecOffset = 151853
	CodecSize   = 32768
)

// Config holds the hyperparameters of the YuE2 AR/NAR backbone.
type Config struct {
	Vocab, Dim, FFN, Heads, KVHeads, HeadDim, Layers, LatentDim, Context int
	RoPEBase                                                             float64
	Eps                                                                  float32
	TimestepShift                                                        float32
}

type configJSON struct {
	VocabSize             int     `json:"vocab_size"`
	HiddenSize            int     `json:"hidden_size"`
	IntermediateSize      int     `json:"intermediate_size"`
	NumAttentionHeads     int     `json:"num_attention_heads"`
	NumKeyValueHeads      int     `json:"num_key_value_heads"`
	HeadDim               int     `json:"head_dim"`
	NumHiddenLayers       int     `json:"num_hidden_layers"`
	RopeTheta             float64 `json:"rope_theta"`
	RmsNormEps            float32 `json:"rms_norm_eps"`
	LatentDim             int     `json:"latent_dim"`
	TimestepShift         float32 `json:"timestep_shift"`
	MaxPositionEmbeddings int     `json:"max_position_embeddings"`
}

// readConfig parses the YuE2 configuration from the GGUF metadata key "yue2.config_json".
func readConfig(g *tensors.GGUF) (Config, error) {
	raw, err := g.String("yue2.config_json")
	if err != nil {
		return Config{}, fmt.Errorf("yue2.config_json: %w", err)
	}
	var cj configJSON
	if err := json.Unmarshal([]byte(raw), &cj); err != nil {
		return Config{}, fmt.Errorf("decode yue2.config_json: %w", err)
	}

	cfg := Config{
		Vocab:         cj.VocabSize,
		Dim:           cj.HiddenSize,
		FFN:           cj.IntermediateSize,
		Heads:         cj.NumAttentionHeads,
		KVHeads:       cj.NumKeyValueHeads,
		HeadDim:       cj.HeadDim,
		Layers:        cj.NumHiddenLayers,
		LatentDim:     cj.LatentDim,
		Context:       cj.MaxPositionEmbeddings,
		RoPEBase:      cj.RopeTheta,
		Eps:           cj.RmsNormEps,
		TimestepShift: cj.TimestepShift,
	}

	if cfg.HeadDim == 0 && cfg.Heads > 0 {
		cfg.HeadDim = cfg.Dim / cfg.Heads
	}
	if cfg.LatentDim == 0 {
		cfg.LatentDim = 64
	}
	if cfg.TimestepShift == 0 {
		cfg.TimestepShift = 1.0
	}
	if cfg.Context == 0 {
		cfg.Context = 24576
	}

	return cfg, nil
}
