package gemma

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/ThiraSoft/golem/chat"
)

func TestAudioMarkersAreReadFromTheVocabulary(t *testing.T) {
	m := openTextModel(t)
	if m.Cfg.AudioOpen == 0 || m.Cfg.AudioClose == 0 || m.Cfg.AudioSoft == 0 {
		t.Fatal("this vocabulary carries <|audio>, <audio|> and <|audio|>, and none was found")
	}
	if m.Cfg.AudioOpen == m.Cfg.ImageOpen {
		t.Fatal("the audio and image markers came back as the same token")
	}
}

// A projector that carries both encoders is opened once and answers for both.
func TestOneProjectorOpensBothEncoders(t *testing.T) {
	path := os.Getenv("GOLEM_MMPROJ")
	if path == "" {
		t.Skip("set GOLEM_MMPROJ to run this test")
	}
	m := openTextModel(t)
	if m.HasAudio() {
		t.Fatal("a model opened without a projector claims to hear")
	}
	if err := m.OpenProjector(path); err != nil {
		t.Fatal(err)
	}
	if !m.HasAudio() || !m.HasVision() {
		t.Fatalf("E2B's projector carries both encoders; sees=%v hears=%v", m.HasVision(), m.HasAudio())
	}
}

// A piece of audio occupies one soft position per encoded row, between the two
// markers, and every one of those positions attends to the whole span.
func TestAudioSpansAreAttendedBothWays(t *testing.T) {
	path := os.Getenv("GOLEM_MMPROJ")
	if path == "" {
		t.Skip("set GOLEM_MMPROJ to run this test")
	}
	m := openTextModel(t)
	if err := m.OpenProjector(path); err != nil {
		t.Fatal(err)
	}
	rows, err := m.EncodeAudio(speechBytes(t))
	if err != nil {
		t.Fatal(err)
	}
	if len(rows) == 0 {
		t.Fatal("the recording became no tokens")
	}

	// The tokens a rendered turn carrying one recording becomes, written out
	// rather than encoded: what this test is about is the splicing, and the
	// rendering has a test of its own.
	tokens := []int32{2, m.Cfg.AudioOpen, m.Cfg.AudioClose, 476}
	p, err := m.BuildPrompt(tokens, nil, [][][]float32{rows})
	if err != nil {
		t.Fatal(err)
	}
	if len(p.Spans) != 1 {
		t.Fatalf("%d spans for one recording", len(p.Spans))
	}
	span := p.Spans[0]
	if span[1]-span[0]+1 != len(rows) {
		t.Fatalf("the span holds %d positions for %d rows", span[1]-span[0]+1, len(rows))
	}
	for i := span[0]; i <= span[1]; i++ {
		if p.Tokens[i] != m.Cfg.AudioSoft {
			t.Fatalf("position %d of the span is not the soft marker", i)
		}
		if p.Embeds[i] == nil {
			t.Fatalf("position %d has no row", i)
		}
	}
	if p.Tokens[span[0]-1] != m.Cfg.AudioOpen || p.Tokens[span[1]+1] != m.Cfg.AudioClose {
		t.Fatal("the span is not enclosed by the two markers")
	}
	// Every position of the span looks at the whole of it, in both directions.
	until := p.Until(0)
	for i := span[0]; i <= span[1]; i++ {
		if until[i] != span[1] {
			t.Fatalf("position %d looks only as far as %d, not to the end of the span at %d", i, until[i], span[1])
		}
	}
}

// A model opened without a projector says so rather than guessing.
func TestEncodingAudioWithoutAProjectorIsAnError(t *testing.T) {
	m := openTextModel(t)
	if _, err := m.EncodeAudio([]byte("RIFF")); err == nil {
		t.Fatal("a model with no projector encoded a piece of audio")
	}
}

// And a prompt whose markers and recordings do not match is an error, not a
// prompt with a hole in it.
func TestAMissingRecordingIsAnError(t *testing.T) {
	m := openTextModel(t)
	tokens := []int32{1, m.Cfg.AudioOpen, m.Cfg.AudioClose, 2}
	if _, err := m.BuildPrompt(tokens, nil, nil); err == nil {
		t.Fatal("a prompt opening a recording that was not given was accepted")
	}
	row := make([]float32, m.Cfg.Dim)
	if _, err := m.BuildPrompt([]int32{1, 2}, nil, [][][]float32{{row}}); err == nil {
		t.Fatal("a recording with no markers to go in was accepted")
	}
}

func speechBytes(t *testing.T) []byte {
	t.Helper()
	path := filepath.Join(repoRoot(t), "testdata", "audio", "speech.wav")
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Skipf("%s is not here; ref/README.md says how to make it", path)
	}
	return raw
}

// The template writes an empty pair of markers where a recording goes, after
// the pictures of the same turn and before its text.
func TestRenderChatWritesTheAudioMarkers(t *testing.T) {
	got, err := RenderChat([]chat.Message{{
		Role: "user", Content: "what is said?", Audio: [][]byte{{1}, {2}},
	}}, ChatOptions{AddGenerationPrompt: true})
	if err != nil {
		t.Fatal(err)
	}
	if want := "<|audio><audio|>\n<|audio><audio|>\nwhat is said?"; !strings.Contains(got, want) {
		t.Fatalf("rendered %q, which does not contain %q", got, want)
	}
}

func TestRenderChatRefusesAModelTurnWithARecording(t *testing.T) {
	if _, err := RenderChat([]chat.Message{
		{Role: "user", Content: "hello"},
		{Role: "model", Content: "hi", Audio: [][]byte{{1}}},
	}, ChatOptions{}); err == nil {
		t.Fatal("a model turn was allowed to carry a recording")
	}
}
