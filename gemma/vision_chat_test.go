package gemma

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func testImageBytes(t *testing.T) []byte {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join(repoRoot(t), "testdata", "gemma", "shapes.png"))
	if err != nil {
		t.Skipf("the test image is not on this machine: %v", err)
	}
	return raw
}

func TestOpenVisionAndEncode(t *testing.T) {
	path := os.Getenv("GOLEM_MMPROJ")
	if path == "" {
		t.Skip("set GOLEM_MMPROJ to run this test")
	}
	m := openTextModel(t)
	if m.HasVision() {
		t.Fatal("a model opened without a projector claims to see")
	}
	if err := m.OpenProjector(path); err != nil {
		t.Fatal(err)
	}
	if !m.HasVision() {
		t.Fatal("a model given a projector still claims not to see")
	}
	rows, err := m.EncodeImage(testImageBytes(t))
	if err != nil {
		t.Fatal(err)
	}
	if len(rows) == 0 {
		t.Fatal("the image became no tokens")
	}
	for i, r := range rows {
		if len(r) != m.Cfg.Dim {
			t.Fatalf("token %d is %d wide, the model is %d", i, len(r), m.Cfg.Dim)
		}
	}
}

func TestEncodeImageWithoutAProjector(t *testing.T) {
	m := openTextModel(t)
	if _, err := m.EncodeImage([]byte("whatever")); err == nil {
		t.Fatal("a model with no projector encoded an image")
	}
}

// The three markers have to be in the vocabulary, because everything below
// holds a place with them.
func TestImageMarkersAreInTheVocabulary(t *testing.T) {
	m := openTextModel(t)
	if m.Cfg.ImageOpen == 0 || m.Cfg.ImageClose == 0 || m.Cfg.ImageSoft == 0 {
		t.Fatalf("the markers came out %d, %d, %d", m.Cfg.ImageOpen, m.Cfg.ImageClose, m.Cfg.ImageSoft)
	}
}

// The template writes an empty pair of markers, and says so before the text.
func TestRenderChatWritesAPictureBeforeTheText(t *testing.T) {
	got, err := RenderChat([]Message{{
		Role:    "user",
		Content: "What is in this picture?",
		Images:  [][]byte{{1}, {2}},
	}}, ChatOptions{AddGenerationPrompt: true})
	if err != nil {
		t.Fatal(err)
	}
	want := "<|turn>user\n<|image><image|>\n<|image><image|>\nWhat is in this picture?<turn|>\n"
	if !strings.Contains(got, want) {
		t.Fatalf("rendered %q, which does not contain %q", got, want)
	}
}

func TestRenderChatRefusesAModelTurnWithAPicture(t *testing.T) {
	if _, err := RenderChat([]Message{
		{Role: "user", Content: "hello"},
		{Role: "model", Content: "hi", Images: [][]byte{{1}}},
	}, ChatOptions{}); err == nil {
		t.Fatal("a model turn was allowed to carry a picture")
	}
}

// BuildPrompt fills the pairs in, in order, and nothing else moves.
func TestBuildPromptSplicesTheRowsIn(t *testing.T) {
	m := openTextModel(t)
	open, closed, soft := m.Cfg.ImageOpen, m.Cfg.ImageClose, m.Cfg.ImageSoft
	tokens := []int32{2, open, closed, 476, open, closed, 477}
	rows := func(n int) [][]float32 {
		out := make([][]float32, n)
		for i := range out {
			out[i] = make([]float32, m.Cfg.Dim)
		}
		return out
	}
	p, err := m.BuildPrompt(tokens, [][][]float32{rows(2), rows(3)}, nil)
	if err != nil {
		t.Fatal(err)
	}
	want := []int32{2, open, soft, soft, closed, 476, open, soft, soft, soft, closed, 477}
	if len(p.Tokens) != len(want) {
		t.Fatalf("the prompt is %d tokens, expected %d: %v", len(p.Tokens), len(want), p.Tokens)
	}
	for i := range want {
		if p.Tokens[i] != want[i] {
			t.Fatalf("token %d is %d, expected %d (%v)", i, p.Tokens[i], want[i], p.Tokens)
		}
	}
	for i := range p.Tokens {
		soft := p.Embeds[i] != nil
		if soft != (p.Tokens[i] == m.Cfg.ImageSoft) {
			t.Fatalf("position %d holds token %d and %v row", i, p.Tokens[i], soft)
		}
		if soft && p.PLE[i] != 0 {
			t.Fatalf("position %d is a soft token and its per-layer identifier is %d", i, p.PLE[i])
		}
	}
	if len(p.Spans) != 2 || p.Spans[0] != [2]int{2, 3} || p.Spans[1] != [2]int{7, 9} {
		t.Fatalf("the spans came out %v", p.Spans)
	}

	// And the places it makes: every token of a picture looks to its end.
	at := p.Places(m.cache, 100)
	if at[2].Until != 103 || at[3].Until != 103 {
		t.Fatalf("the first picture's tokens look to %d and %d", at[2].Until, at[3].Until)
	}
	if at[5].Until != 105 {
		t.Fatalf("a text token looks to %d, expected its own position 105", at[5].Until)
	}
}

func TestBuildPromptRefusesAMiscount(t *testing.T) {
	m := openTextModel(t)
	tokens := []int32{2, m.Cfg.ImageOpen, m.Cfg.ImageClose}
	if _, err := m.BuildPrompt(tokens, nil, nil); err == nil {
		t.Fatal("a prompt that opens a picture was built with none given")
	}
	rows := [][][]float32{{make([]float32, m.Cfg.Dim)}, {make([]float32, m.Cfg.Dim)}}
	if _, err := m.BuildPrompt(tokens, rows, nil); err == nil {
		t.Fatal("two pictures went into a prompt that opens one")
	}
}

func TestPromptSliceAndBoundary(t *testing.T) {
	p := &Prompt{
		Tokens: make([]int32, 20),
		Embeds: make([][]float32, 20),
		PLE:    make([]int32, 20),
		Spans:  [][2]int{{3, 6}, {12, 17}},
	}
	// A cut inside a picture moves out past it; one outside stays put.
	for _, c := range []struct{ from, want, expect int }{
		{0, 5, 7},
		{0, 3, 3},
		{0, 7, 7},
		{7, 14, 18},
		{7, 12, 12},
		{18, 20, 20},
	} {
		if got := p.Boundary(c.from, c.want); got != c.expect {
			t.Errorf("a batch from %d wanting %d may end at %d, expected %d", c.from, c.want, got, c.expect)
		}
	}
	// A picture larger than the batch still makes progress.
	if got := p.Boundary(3, 4); got != 7 {
		t.Errorf("a cut inside the first picture came out %d", got)
	}

	s := p.Slice(7, 18)
	if len(s.Tokens) != 11 {
		t.Fatalf("the slice is %d tokens", len(s.Tokens))
	}
	if len(s.Spans) != 1 || s.Spans[0] != [2]int{5, 10} {
		t.Fatalf("the slice's spans came out %v", s.Spans)
	}
	if len(p.Slice(0, 7).Spans) != 1 {
		t.Fatalf("a slice holding the whole first picture lost it")
	}
}
