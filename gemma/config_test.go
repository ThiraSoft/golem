package gemma

import (
	"os"
	"testing"

	"github.com/ThiraSoft/golem/sample"
	"github.com/ThiraSoft/golem/tensors"
)

// modelPath is where the tests look for the weights: GOLEM_MODEL, the same
// variable cmd/chat reads. A machine without the file skips, exactly as the
// pocket-tts tests do — the weights are three gigabytes and belong to whoever
// published them, not to this repository.
func modelPath(tb testing.TB) string {
	tb.Helper()
	path := os.Getenv("GOLEM_MODEL")
	if path == "" {
		tb.Skip("set GOLEM_MODEL to a Gemma 4 GGUF to run this test")
	}
	if _, err := os.Stat(path); err != nil {
		tb.Skipf("GOLEM_MODEL names %s, which is not there", path)
	}
	return path
}

func openModel(t *testing.T) *tensors.GGUF {
	t.Helper()
	g, err := tensors.OpenGGUF(modelPath(t))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { g.Close() })
	return g
}

func TestLoadConfigE2B(t *testing.T) {
	g := openModel(t)
	cfg, err := LoadConfig(g, 4096)
	if err != nil {
		t.Fatal(err)
	}

	if cfg.Arch != "gemma4" || cfg.Dim != 1536 || cfg.Vocab != 262144 {
		t.Fatalf("arch %q, dim %d, vocab %d", cfg.Arch, cfg.Dim, cfg.Vocab)
	}
	if len(cfg.Blocks) != 35 {
		t.Fatalf("%d blocks", len(cfg.Blocks))
	}
	if cfg.PLEDim != 256 || cfg.LogitSoftcap != 30 || cfg.Eps != 1e-6 {
		t.Fatalf("ple %d, softcap %v, eps %v", cfg.PLEDim, cfg.LogitSoftcap, cfg.Eps)
	}

	// The pattern is four window blocks then a global one, from the first.
	for i, b := range cfg.Blocks {
		wantWindow := (i+1)%5 != 0
		if b.Window != wantWindow {
			t.Fatalf("block %d: window %v, want %v", i, b.Window, wantWindow)
		}
	}

	w := cfg.Blocks[0]
	if w.Heads != 8 || w.KVHeads != 1 || w.HeadDim != 256 || w.RoPEBase != 10000 ||
		w.RoPEDims != 256 || w.WindowSize != 512 || w.FFN != 6144 {
		t.Fatalf("window block: %+v", w)
	}

	gl := cfg.Blocks[4]
	if gl.Heads != 8 || gl.KVHeads != 1 || gl.HeadDim != 512 || gl.RoPEBase != 1e6 ||
		gl.RoPEDims != 512 || gl.WindowSize != 0 || gl.FFN != 6144 {
		t.Fatalf("global block: %+v", gl)
	}

	// The feed forward doubles from block 15.
	if cfg.Blocks[14].FFN != 6144 || cfg.Blocks[15].FFN != 12288 || cfg.Blocks[34].FFN != 12288 {
		t.Fatalf("feed forward widths: %d %d %d",
			cfg.Blocks[14].FFN, cfg.Blocks[15].FFN, cfg.Blocks[34].FFN)
	}

	// Blocks 0..14 own their keys and values; 15..34 borrow, window blocks from
	// 13 and global ones from 14.
	for i, b := range cfg.Blocks {
		switch {
		case i < 15:
			if !b.OwnsKV || b.KVSource != i {
				t.Fatalf("block %d should own its cache, got %+v", i, b)
			}
		default:
			want := 14
			if b.Window {
				want = 13
			}
			if b.OwnsKV || b.KVSource != want {
				t.Fatalf("block %d should read block %d, got %+v", i, want, b)
			}
		}
	}

	// The file declares what it wants sampled with.
	if cfg.Sampling.Temperature != 1 || cfg.Sampling.TopK != 64 || cfg.Sampling.TopP != 0.95 {
		t.Fatalf("sampling %+v", cfg.Sampling)
	}

	if cfg.MaxHeadDim() != 512 {
		t.Fatalf("widest head %d", cfg.MaxHeadDim())
	}
}

// The 12B declares the same architecture without either of the mechanisms that
// make E2B awkward, and with a head_count_kv that varies per block. Nothing
// here is a special case: they are ordinary values.
func TestLoadConfigTwelveB(t *testing.T) {
	const blocks = 48
	pattern := make([]any, blocks)
	kvHeads := make([]any, blocks)
	for i := range pattern {
		window := (i+1)%6 != 0 // five window blocks then a global one
		pattern[i] = window
		if window {
			kvHeads[i] = uint32(8)
		} else {
			kvHeads[i] = uint32(1)
		}
	}

	g := &tensors.GGUF{
		Meta: map[string]any{
			"general.architecture":                    "gemma4",
			"gemma4.block_count":                      uint32(blocks),
			"gemma4.embedding_length":                 uint32(3840),
			"gemma4.attention.head_count":             uint32(16),
			"gemma4.attention.head_count_kv":          kvHeads,
			"gemma4.attention.key_length":             uint32(256),
			"gemma4.attention.key_length_swa":         uint32(256),
			"gemma4.attention.sliding_window":         uint32(1024),
			"gemma4.attention.sliding_window_pattern": pattern,
			"gemma4.attention.shared_kv_layers":       uint32(0),
			"gemma4.attention.layer_norm_rms_epsilon": float32(1e-6),
			"gemma4.rope.freq_base":                   float32(1e6),
			"gemma4.rope.freq_base_swa":               float32(10000),
			"gemma4.rope.dimension_count":             uint32(256),
			"gemma4.rope.dimension_count_swa":         uint32(256),
			"gemma4.embedding_length_per_layer_input": uint32(0),
			"gemma4.feed_forward_length":              uint32(15360),
		},
		Tensors: map[string]tensors.Tensor{
			"token_embd.weight": {Shape: []int{3840, 262144}, DType: "Q4_0"},
		},
	}

	cfg, err := LoadConfig(g, 4096)
	if err != nil {
		t.Fatal(err)
	}
	if cfg.PLEDim != 0 {
		t.Fatalf("the 12B has no per-layer embeddings, got %d", cfg.PLEDim)
	}
	if cfg.LogitSoftcap != 0 {
		t.Fatalf("no softcapping is declared, got %v", cfg.LogitSoftcap)
	}
	if len(cfg.Blocks) != blocks {
		t.Fatalf("%d blocks", len(cfg.Blocks))
	}
	for i, b := range cfg.Blocks {
		if !b.OwnsKV || b.KVSource != i {
			t.Fatalf("block %d borrows a cache in a model that shares none: %+v", i, b)
		}
		if b.FFN != 15360 {
			t.Fatalf("block %d feed forward %d", i, b.FFN)
		}
		wantKV := 1
		if b.Window {
			wantKV = 8
		}
		if b.KVHeads != wantKV {
			t.Fatalf("block %d has %d key-value heads, want %d", i, b.KVHeads, wantKV)
		}
	}
	// This one declares no sampling parameters, so the package's own stand.
	if cfg.Sampling != sample.Defaults() {
		t.Fatalf("a file declaring nothing got %+v", cfg.Sampling)
	}
	if cfg.Blocks[5].Window || cfg.Blocks[4].WindowSize != 1024 {
		t.Fatalf("pattern or window wrong: %+v %+v", cfg.Blocks[4], cfg.Blocks[5])
	}
}

func TestLoadConfigRejectsAnotherArchitecture(t *testing.T) {
	g := &tensors.GGUF{Meta: map[string]any{"general.architecture": "llama"}}
	if _, err := LoadConfig(g, 4096); err == nil {
		t.Fatal("a llama file should not load as a Gemma configuration")
	}
}
