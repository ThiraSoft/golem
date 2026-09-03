package stt

import (
	"context"
	"os"
	"strings"
	"testing"
)

// TestTranscribeClip is the only test here that judges the transcript, and it
// judges it as a word error rate: greedy decoding is deterministic, but pinning
// an exact string would pin the wrong thing — a comma moving is not a
// regression, and half the words changing is.
func TestTranscribeClip(t *testing.T) {
	dir := os.Getenv("GOLEM_STT")
	if dir == "" {
		t.Skip("GOLEM_STT not set")
	}
	clip, err := os.ReadFile("../testdata/stt/clip.wav")
	if err != nil {
		t.Skipf("no clip: %v", err)
	}
	want, err := os.ReadFile("../testdata/stt/transcript.txt")
	if err != nil {
		t.Fatal(err)
	}
	o, err := Locate(dir)
	if err != nil {
		t.Fatal(err)
	}
	m, err := Open(o)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	got, err := m.Transcribe(clip)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("transcribed: %q", got)
	if rate := wordErrorRate(strings.Fields(string(want)), strings.Fields(got)); rate > 0.15 {
		t.Fatalf("word error rate %.2f\n got: %s\nwant: %s", rate, got, want)
	}
}

// wordErrorRate is Levenshtein over words, divided by the reference length.
func wordErrorRate(want, got []string) float64 {
	prev := make([]int, len(got)+1)
	cur := make([]int, len(got)+1)
	for j := range prev {
		prev[j] = j
	}
	for i := 1; i <= len(want); i++ {
		cur[0] = i
		for j := 1; j <= len(got); j++ {
			cost := 1
			if strings.EqualFold(want[i-1], got[j-1]) {
				cost = 0
			}
			cur[j] = min(prev[j]+1, min(cur[j-1]+1, prev[j-1]+cost))
		}
		prev, cur = cur, prev
	}
	if len(want) == 0 {
		return 0
	}
	return float64(prev[len(got)]) / float64(len(want))
}

// TestTranscribeMatchesStream checks that the two doors are one path: the same
// clip written in one call and in 80 ms pieces must give the same text.
func TestTranscribeMatchesStream(t *testing.T) {
	dir := os.Getenv("GOLEM_STT")
	if dir == "" {
		t.Skip("GOLEM_STT not set")
	}
	o, err := Locate(dir)
	if err != nil {
		t.Fatal(err)
	}
	m, err := Open(o)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()

	clip, err := os.ReadFile("../testdata/audio/speech.wav")
	if err != nil {
		t.Skipf("no speech.wav: %v", err)
	}
	whole, err := m.Transcribe(clip)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("speech.wav transcription: %q", whole)

	var streamed strings.Builder
	err = m.TranscribeStream(context.Background(), clip, func(seg Segment) {
		streamed.WriteString(seg.Text)
	})
	if err != nil {
		t.Fatal(err)
	}
	if whole != strings.TrimSpace(streamed.String()) {
		t.Fatalf("streamed %q != whole %q", streamed.String(), whole)
	}
}
