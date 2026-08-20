package pockettts

// The languages the engine knows, and what distinguishes them.
//
// Kyutai ships one model per language, all sharing the same architecture. Four
// things actually vary, and only four — the rest of the YAML config is identical
// from one language to the next:
//
//   - the depth of the flow_lm transformer, 24 layers or 6;
//   - whether semicolons are folded into commas before synthesis;
//   - whether very short inputs are padded with spaces;
//   - how many frames to keep generating after the end is detected.
//
// Two config keys look like they should matter and do not.
// `flow_lm.insert_bos_before_voice` is applied when the voice state is built,
// on the Python side, so it is already baked into the file this engine reads.
// `mimi.inner_dim` only sizes the encoder's downsampling and the speaker
// projection, neither of which is on the decoding path.

import (
	"fmt"
	"sort"
)

// Language holds what the engine needs to know about one language.
type Language struct {
	Name string

	// Layers is the depth of the flow_lm transformer.
	Layers int

	// RemoveSemicolons folds ";" into "," before synthesis. The models trained
	// with it hesitate on a semicolon they never saw.
	RemoveSemicolons bool

	// PadShortInputs prefixes eight spaces to inputs of fewer than five words.
	// The model generates poorly on very few tokens, and the padding buys it
	// some.
	PadShortInputs bool

	// FramesAfterEnd is how many frames to keep generating once the end is
	// detected, so the sentence can settle.
	FramesAfterEnd int
}

// DefaultLanguage is the one the engine uses when none is named.
const DefaultLanguage = "french_24l"

// defaultFramesAfterEnd is what the reference daemon falls back to when a
// language does not state its own.
const defaultFramesAfterEnd = 8

var languages = map[string]Language{
	"english":         {Name: "english", Layers: 6, FramesAfterEnd: defaultFramesAfterEnd},
	"english_2026-01": {Name: "english_2026-01", Layers: 6, PadShortInputs: true, FramesAfterEnd: defaultFramesAfterEnd},
	"english_2026-04": {Name: "english_2026-04", Layers: 6, FramesAfterEnd: defaultFramesAfterEnd},
	"french_24l":      {Name: "french_24l", Layers: 24, RemoveSemicolons: true, FramesAfterEnd: 8},
	"german":          {Name: "german", Layers: 6, RemoveSemicolons: true, FramesAfterEnd: defaultFramesAfterEnd},
	"german_24l":      {Name: "german_24l", Layers: 24, RemoveSemicolons: true, FramesAfterEnd: defaultFramesAfterEnd},
	"italian":         {Name: "italian", Layers: 6, FramesAfterEnd: defaultFramesAfterEnd},
	"italian_24l":     {Name: "italian_24l", Layers: 24, FramesAfterEnd: defaultFramesAfterEnd},
	"portuguese":      {Name: "portuguese", Layers: 6, FramesAfterEnd: defaultFramesAfterEnd},
	"portuguese_24l":  {Name: "portuguese_24l", Layers: 24, FramesAfterEnd: defaultFramesAfterEnd},
	"spanish":         {Name: "spanish", Layers: 6, FramesAfterEnd: defaultFramesAfterEnd},
	"spanish_24l":     {Name: "spanish_24l", Layers: 24, FramesAfterEnd: defaultFramesAfterEnd},
}

// LookupLanguage returns a language by name.
func LookupLanguage(name string) (Language, error) {
	if name == "" {
		name = DefaultLanguage
	}
	l, ok := languages[name]
	if !ok {
		return Language{}, fmt.Errorf("unknown language %q; known: %v", name, Languages())
	}
	return l, nil
}

// Languages lists the known language names, in order.
func Languages() []string {
	names := make([]string, 0, len(languages))
	for n := range languages {
		names = append(names, n)
	}
	sort.Strings(names)
	return names
}

// WeightsPath is where the language's weights sit inside a Pocket TTS
// snapshot, and TokenizerPath where its tokenizer does. The tokenizer of
// english_2026-01 lives at the root of the repository rather than under its
// language directory — the one irregularity in the layout.
func (l Language) WeightsPath() string {
	return "languages/" + l.Name + "/model.safetensors"
}

func (l Language) TokenizerPath() string {
	if l.Name == "english_2026-01" {
		return "tokenizer.model"
	}
	return "languages/" + l.Name + "/tokenizer.model"
}

// EmbeddingPath is where a predefined voice sits inside a Pocket TTS snapshot.
// The file is the voice state the model starts from, already computed by
// Kyutai; nothing here encodes a voice from sound.
func (l Language) EmbeddingPath(voice string) string {
	return "languages/" + l.Name + "/embeddings/" + voice + ".safetensors"
}
