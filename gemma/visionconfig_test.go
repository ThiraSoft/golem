package gemma

import (
	"os"
	"testing"

	"github.com/ThiraSoft/golem/tensors"
)

// openMMProj is what every vision test starts with. GOLEM_MMPROJ names the
// projector file; a machine without it skips.
func openMMProj(tb testing.TB) *tensors.GGUF {
	tb.Helper()
	path := os.Getenv("GOLEM_MMPROJ")
	if path == "" {
		tb.Skip("set GOLEM_MMPROJ to a Gemma 4 mmproj GGUF to run this test")
	}
	g, err := tensors.OpenGGUF(path)
	if err != nil {
		tb.Fatalf("GOLEM_MMPROJ names %s, which will not open: %v", path, err)
	}
	tb.Cleanup(func() { g.Close() })
	return g
}

func TestLoadVisionConfig(t *testing.T) {
	cfg, err := LoadVisionConfig(openMMProj(t))
	if err != nil {
		t.Fatal(err)
	}
	for _, c := range []struct {
		name string
		got  int
		want int
	}{
		{"dim", cfg.Dim, 768},
		{"blocks", cfg.Blocks, 16},
		{"heads", cfg.Heads, 12},
		{"head dim", cfg.HeadDim, 64},
		{"ffn", cfg.FFN, 3072},
		{"patch", cfg.PatchSize, 16},
		{"merge", cfg.Merge, 3},
		{"projection", cfg.ProjDim, 1536},
		{"min tokens", cfg.MinTokens, 40},
		{"max tokens", cfg.MaxTokens, 280},
	} {
		if c.got != c.want {
			t.Errorf("%s is %d, expected %d", c.name, c.got, c.want)
		}
	}
	if cfg.RoPEBase != 100 {
		t.Errorf("rope base is %v, expected 100", cfg.RoPEBase)
	}
}

func TestLoadVisionConfigRefusesTheTextModel(t *testing.T) {
	path := os.Getenv("GOLEM_MODEL")
	if path == "" {
		t.Skip("set GOLEM_MODEL to run this test")
	}
	g, err := tensors.OpenGGUF(path)
	if err != nil {
		t.Skipf("GOLEM_MODEL will not open: %v", err)
	}
	defer g.Close()
	if _, err := LoadVisionConfig(g); err == nil {
		t.Fatal("the text model was accepted as a projector")
	}
}
