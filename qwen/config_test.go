package qwen

import (
	"os"
	"strings"
	"testing"

	"github.com/ThiraSoft/golem/tensors"
)

func modelPath(t *testing.T) string {
	t.Helper()
	path := os.Getenv("GOLEM_MODEL_QWEN")
	if path == "" {
		t.Skip("set GOLEM_MODEL_QWEN to a Qwen3 GGUF to run this test")
	}
	if _, err := os.Stat(path); err != nil {
		t.Skipf("GOLEM_MODEL_QWEN names %s, which is not there", path)
	}
	return path
}

func openFile(t *testing.T) *tensors.GGUF {
	t.Helper()
	g, err := tensors.OpenGGUF(modelPath(t))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { g.Close() })
	return g
}

func TestLoadConfigReadsTheGeometry(t *testing.T) {
	cfg, err := LoadConfig(openFile(t), 4096)
	if err != nil {
		t.Fatal(err)
	}

	if cfg.Arch != "qwen3" {
		t.Errorf("architecture %q", cfg.Arch)
	}
	if cfg.Dim != 1024 {
		t.Errorf("dim %d, want 1024", cfg.Dim)
	}
	if cfg.Vocab != 151936 {
		t.Errorf("vocab %d, want 151936", cfg.Vocab)
	}
	if cfg.Eps != 1e-6 {
		t.Errorf("eps %g, want 1e-6", cfg.Eps)
	}
	// The file declares 40960; the caller's cap is what a cache is built for.
	if cfg.MaxContext != 4096 {
		t.Errorf("max context %d, want the cap the caller asked for", cfg.MaxContext)
	}

	if len(cfg.Blocks) != 28 {
		t.Fatalf("%d blocks, want 28", len(cfg.Blocks))
	}
	// Twenty-eight blocks and no two of them different, which is a fact of
	// this file rather than of the architecture — so it is asserted, not
	// assumed.
	for i, b := range cfg.Blocks {
		if b.Index != i {
			t.Fatalf("block %d says it is %d", i, b.Index)
		}
		if b.Heads != 16 || b.KVHeads != 8 || b.HeadDim != 128 {
			t.Fatalf("block %d: %d queries, %d key-values, head of %d",
				i, b.Heads, b.KVHeads, b.HeadDim)
		}
		if b.FFN != 3072 {
			t.Fatalf("block %d: feed forward %d, want 3072", i, b.FFN)
		}
		if b.RoPEBase != 1e6 {
			t.Fatalf("block %d: rope base %g, want 1e6", i, b.RoPEBase)
		}
		// No rope.dimension_count in this file, so the whole head rotates.
		if b.RoPEDims != 128 {
			t.Fatalf("block %d: %d rotated dimensions, want 128", i, b.RoPEDims)
		}
	}
}

// The query projection is wider than the residual stream: 16 heads of 128 is
// 2048 against a model 1024 wide. Worth pinning, because a reader who assumes
// heads*headDim == dim writes code that works for neither checkpoint.
func TestQueryProjectionIsWiderThanTheModel(t *testing.T) {
	cfg, err := LoadConfig(openFile(t), 4096)
	if err != nil {
		t.Fatal(err)
	}
	b := cfg.Blocks[0]
	if b.Heads*b.HeadDim <= cfg.Dim {
		t.Errorf("%d*%d is not wider than %d, so this test no longer says anything",
			b.Heads, b.HeadDim, cfg.Dim)
	}
}

func TestLoadConfigRefusesAnotherArchitecture(t *testing.T) {
	path := os.Getenv("GOLEM_MODEL")
	if path == "" {
		t.Skip("set GOLEM_MODEL to a Gemma 4 GGUF to run this test")
	}
	g, err := tensors.OpenGGUF(path)
	if err != nil {
		t.Fatal(err)
	}
	defer g.Close()

	_, err = LoadConfig(g, 4096)
	if err == nil {
		t.Fatal("a gemma4 file was accepted as qwen3")
	}
	// The error has to name what it found, or a person meeting it learns
	// nothing they did not already know.
	if !strings.Contains(err.Error(), "gemma4") {
		t.Errorf("the error should name the architecture it found: %v", err)
	}
}
