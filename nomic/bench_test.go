package nomic

// What an embedding request costs: a batch of short texts, which is what a
// search index sends, and one text as long as the context allows.
//
// GOLEM_NOMIC_CORPUS names a file with one text a line; without it the
// benchmark builds its own. The rate reported is positions a second, which is
// what llama.cpp's own timings print, so the two can be set side by side.

import (
	"bufio"
	"os"
	"strings"
	"testing"
)

func benchTexts(b *testing.B, m *Model, long bool) [][]int32 {
	var lines []string
	if path := os.Getenv("GOLEM_NOMIC_CORPUS"); path != "" && !long {
		f, err := os.Open(path)
		if err != nil {
			b.Fatal(err)
		}
		defer f.Close()
		s := bufio.NewScanner(f)
		s.Buffer(make([]byte, 1<<20), 1<<20)
		for s.Scan() {
			if line := strings.TrimSpace(s.Text()); line != "" {
				lines = append(lines, line)
			}
		}
	} else if long {
		lines = []string{strings.Repeat("the model reads a text once and every position sees every other ", 60)}
	} else {
		for i := 0; i < 64; i++ {
			lines = append(lines, strings.Repeat("search_document: a short passage about embeddings ", 5))
		}
	}
	texts := make([][]int32, len(lines))
	for i, l := range lines {
		texts[i] = m.Tokenize(l)
	}
	return texts
}

func benchEmbed(b *testing.B, long bool) {
	path := os.Getenv("GOLEM_MODEL_NOMIC")
	if path == "" {
		b.Skip("GOLEM_MODEL_NOMIC is not set")
	}
	m, err := Open(path)
	if err != nil {
		b.Fatal(err)
	}
	defer m.Close()
	texts := benchTexts(b, m, long)
	positions := 0
	for _, t := range texts {
		positions += len(t)
	}
	if _, err := m.Embed(texts); err != nil { // warm the mapping
		b.Fatal(err)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := m.Embed(texts); err != nil {
			b.Fatal(err)
		}
	}
	b.ReportMetric(float64(positions*b.N)/b.Elapsed().Seconds(), "pos/s")
	b.ReportMetric(float64(len(texts)*b.N)/b.Elapsed().Seconds(), "texts/s")
}

func BenchmarkEmbedBatch(b *testing.B) { benchEmbed(b, false) }
func BenchmarkEmbedLong(b *testing.B)  { benchEmbed(b, true) }
