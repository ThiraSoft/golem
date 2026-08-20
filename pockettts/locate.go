package pockettts

import (
	"os"
	"path/filepath"
	"sort"
	"strings"
)

// The two Hugging Face repositories Kyutai publishes. The weights come from the
// voice-cloning one, the tokenizer and the predefined voices from the other;
// that is the split the upstream configs use. Locate looks in both rather than
// insisting on which, because a machine may have fetched either.
var repositories = []string{
	".cache/huggingface/hub/models--kyutai--pocket-tts/snapshots/*/",
	".cache/huggingface/hub/models--kyutai--pocket-tts-without-voice-cloning/snapshots/*/",
}

// Locate returns the absolute path of relative inside the Hugging Face cache,
// or the empty string when it is not there. relative is what WeightsPath,
// TokenizerPath or EmbeddingPath returned.
//
// A missing file is not an error here: the caller knows what it was looking for
// and can say so better than this function can.
func Locate(relative string) string {
	for _, repo := range repositories {
		found, _ := filepath.Glob(filepath.Join(os.Getenv("HOME"), repo, relative))
		for _, p := range found {
			if _, err := os.Stat(p); err == nil {
				return p
			}
		}
	}
	return ""
}

// LocateVoices lists the predefined voices available for a language, by name
// and in order. An empty result means the model was never downloaded, or was
// downloaded without its voices.
func LocateVoices(l Language) []string {
	seen := map[string]bool{}
	for _, repo := range repositories {
		pattern := filepath.Join(os.Getenv("HOME"), repo, l.EmbeddingPath("*"))
		found, _ := filepath.Glob(pattern)
		for _, p := range found {
			seen[strings.TrimSuffix(filepath.Base(p), ".safetensors")] = true
		}
	}
	names := make([]string, 0, len(seen))
	for n := range seen {
		names = append(names, n)
	}
	sort.Strings(names)
	return names
}
