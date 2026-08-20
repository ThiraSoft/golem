package reference

// Where to find the files the parity tests need. The voice states ship with the
// repository, under testdata/voices, one directory per language. The weights and
// the tokenizers do not — 672 MB has no place in a git history — so the tests
// that depend on them skip cleanly when they are absent.

import (
	"os"
	"path/filepath"
	"testing"
)

// The two Hugging Face repositories the Python daemon downloads into. Only the
// languages actually fetched on this machine are present, so a test for a
// language nobody downloaded skips rather than fails.
const cloningRepo = ".cache/huggingface/hub/models--kyutai--pocket-tts/snapshots/*/"
const plainRepo = ".cache/huggingface/hub/models--kyutai--pocket-tts-without-voice-cloning/snapshots/*/"

// DefaultLanguage is the language the fixtures were historically written
// against, and the one the tests fall back to.
const DefaultLanguage = "french_24l"

// ModelPath returns the weights of a language, or skips the test if they cannot
// be found. The weights are looked up in both repositories: the engine only
// needs the decoding half, which the voice-cloning-free one also carries.
func ModelPath(t testing.TB, language string) string {
	t.Helper()
	if p := os.Getenv("POCKET_TTS_WEIGHTS"); p != "" && language == DefaultLanguage {
		return p
	}
	rel := "languages/" + language + "/model.safetensors"
	for _, repo := range []string{cloningRepo, plainRepo} {
		if p := glob(repo, rel); p != "" {
			return p
		}
	}
	t.Skipf("weights for %s not downloaded", language)
	return ""
}

// TokenizerPath returns the SentencePiece model of a language. english_2026-01
// keeps its tokenizer at the root of the repository rather than under its
// language directory, so both places are tried.
func TokenizerPath(t testing.TB, language string) string {
	t.Helper()
	if p := os.Getenv("POCKET_TTS_TOKENIZER"); p != "" && language == DefaultLanguage {
		return p
	}
	for _, rel := range []string{"languages/" + language + "/tokenizer.model", "tokenizer.model"} {
		if p := glob(plainRepo, rel); p != "" {
			return p
		}
	}
	t.Skipf("tokenizer for %s not downloaded", language)
	return ""
}

// VoicePath returns a voice state for a language. None ship with the repository
// — they are Kyutai's to publish — so three places are tried: what
// POCKET_TTS_VOICE names, whatever has been put under testdata/voices/, and the
// voices Kyutai ships inside the model repository itself, which is where a
// machine that has downloaded the weights already has some.
func VoicePath(t testing.TB, language string) string {
	t.Helper()
	if p := os.Getenv("POCKET_TTS_VOICE"); p != "" && language == DefaultLanguage {
		return p
	}
	if found, _ := filepath.Glob(filepath.Join(root(t), "testdata", "voices", language, "*.safetensors")); len(found) > 0 {
		return found[0]
	}
	for _, repo := range []string{plainRepo, cloningRepo} {
		if p := glob(repo, "languages/"+language+"/embeddings/*.safetensors"); p != "" {
			return p
		}
	}
	t.Skipf("no voice state for %s, in testdata/voices or in the model repository", language)
	return ""
}

// glob resolves one file inside a Hugging Face snapshot, or returns "".
func glob(repo, relative string) string {
	found, _ := filepath.Glob(filepath.Join(os.Getenv("HOME"), repo, relative))
	if len(found) > 0 {
		return found[0]
	}
	return ""
}
