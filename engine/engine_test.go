package engine

import (
	"os"
	"strings"
	"testing"

	"github.com/ThiraSoft/golem/chat"
)

// The dispatch is the part worth testing without weights: a file that is not a
// GGUF, and an architecture nobody implements, must both say so.
func TestOpenRefusesWhatIsNotAModel(t *testing.T) {
	f, err := os.CreateTemp(t.TempDir(), "*.gguf")
	if err != nil {
		t.Fatal(err)
	}
	f.WriteString("not a gguf at all")
	f.Close()
	if _, err := Open(f.Name(), 128); err == nil {
		t.Fatal("a file that is not a GGUF should be refused")
	}
}

func TestUnknownArchitectureNamesWhatItFoundAndWhatItHas(t *testing.T) {
	err := unknownArchitecture("mamba")
	for _, want := range []string{"mamba", "gemma4", "qwen3"} {
		if !strings.Contains(err.Error(), want) {
			t.Fatalf("the error should name %s: %v", want, err)
		}
	}
}

// The real thing, when weights are on disk.
func TestOpenReadsARealModel(t *testing.T) {
	for _, key := range []string{"GOLEM_MODEL", "GOLEM_MODEL_QWEN"} {
		t.Run(key, func(t *testing.T) {
			path := os.Getenv(key)
			if path == "" {
				t.Skipf("%s is not set", key)
			}
			m, err := Open(path, 256)
			if err != nil {
				t.Fatal(err)
			}
			defer m.Close()
			if m.Template == nil || m.Vocab == nil || m.Forward == nil {
				t.Fatalf("an incomplete model: %+v", m)
			}
			if m.Blocks == 0 || m.Vocabulary == 0 {
				t.Fatalf("blocks %d vocabulary %d", m.Blocks, m.Vocabulary)
			}
			text, err := m.Template.Render(
				[]chat.Message{{Role: "user", Content: "hi"}},
				chat.Options{AddGenerationPrompt: true})
			if err != nil {
				t.Fatal(err)
			}
			if ids := m.Vocab.Encode(text, false, true); len(ids) == 0 {
				t.Fatal("the rendered prompt encoded to nothing")
			}
		})
	}
}
