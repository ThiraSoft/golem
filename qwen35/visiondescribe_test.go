package qwen35

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/ThiraSoft/golem/chat"
	"github.com/ThiraSoft/golem/tensors"
	"github.com/ThiraSoft/golem/token/bytebpe"
)

// The whole thing: a picture in, a description out, on the card.
//
// It asserts what a user would notice and nothing finer. The tower's numbers
// are held to llama.cpp by TestVisionTowerMatchesLlamaCpp; what this says is
// that the rows reach the model at the right positions and that the model has
// something to say about them — which is the half no waypoint can check.
func TestVulkanDescribesAnImage(t *testing.T) {
	g, err := tensors.OpenGGUF(qwen38)
	if err != nil {
		t.Skipf("open: %v", err)
	}
	vocab, err := bytebpe.Load(g)
	if err != nil {
		t.Fatal(err)
	}
	m, err := New(g, 1024)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	if start, ok := vocab.ID(VisionStart); ok {
		pad, _ := vocab.ID(ImagePad)
		end, _ := vocab.ID(VisionEnd)
		m.SetVisionMarkers(start, pad, end)
	}
	if err := m.OpenProjector(qwen38mmproj); err != nil {
		t.Skipf("projector: %v", err)
	}
	if err := m.UseVulkan(); err != nil {
		t.Skipf("vulkan: %v", err)
	}

	raw, err := os.ReadFile(filepath.Join("..", "testdata", "gemma", "shapes.png"))
	if err != nil {
		t.Skipf("image: %v", err)
	}
	rows, err := m.EncodeImage(raw)
	if err != nil {
		t.Fatal(err)
	}
	grid, _ := m.GridOf(0)

	tpl := NewTemplate()
	rendered, err := tpl.Render([]chat.Message{{
		Role: "user", Content: "Describe this image in one sentence.", Images: [][]byte{raw},
	}}, chat.Options{AddGenerationPrompt: true})
	if err != nil {
		t.Fatal(err)
	}
	ids := vocab.Encode(rendered, false, true)

	p, err := m.BuildPrompt(ids, [][][]float32{rows}, [][2]int{grid})
	if err != nil {
		t.Fatal(err)
	}
	if len(p.Images) != 1 {
		t.Fatalf("%d pictures placed, wanted 1", len(p.Images))
	}
	if p.Images[0].Count != len(rows) {
		t.Fatalf("the picture took %d places and has %d rows", p.Images[0].Count, len(rows))
	}
	t.Logf("prompt: %d positions, %d of them the picture at %d",
		len(p.Tokens), p.Images[0].Count, p.Images[0].Start)

	states := m.ForwardPrompt(p, 0)
	hidden := states[len(states)-1]

	logits := make([]float32, m.Cfg.Vocab)
	var out strings.Builder
	pos := len(p.Tokens)
	for i := 0; i < 24; i++ {
		m.Logits(hidden, logits)
		best, bi := float32(-1e30), 0
		for j, v := range logits {
			if v > best {
				best, bi = v, j
			}
		}
		id := int32(bi)
		if vocab.IsEOG(id) {
			break
		}
		out.WriteString(vocab.Piece(id, false))
		hidden = m.ForwardBatch([]int32{id}, pos)[0]
		pos++
	}

	text := strings.TrimSpace(out.String())
	t.Logf("answer: %q", text)
	if text == "" {
		t.Fatal("the model said nothing about the picture")
	}
	// Not a check on what it said — that is the model's business — but on
	// whether it said anything shaped like prose. A broken splice gives
	// repeated punctuation or one token forever.
	if len([]rune(text)) < 8 {
		t.Fatalf("the answer is %q, which is not a description", text)
	}
}
