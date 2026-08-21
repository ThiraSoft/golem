package gemma

// Giving a model eyes, and putting a picture into a prompt.
//
// The tower is a second file and a second set of weights, bound to a model
// that was already loaded: a checkpoint is a checkpoint whether or not anyone
// opens a projector for it, and everything below returns an error rather than
// a guess when none was opened.

import (
	"bytes"
	"fmt"

	"github.com/ThiraSoft/golem/imageio"
	"github.com/ThiraSoft/golem/tensors"
)

// OpenVision maps a projector file and binds it to this model.
func (m *Model) OpenVision(path string) error {
	g, err := tensors.OpenGGUF(path)
	if err != nil {
		return err
	}
	cfg, err := LoadVisionConfig(g)
	if err != nil {
		g.Close()
		return err
	}
	if cfg.ProjDim != m.Cfg.Dim {
		g.Close()
		return fmt.Errorf("gemma: the projector produces %d-wide tokens and this model reads %d; they are not a pair",
			cfg.ProjDim, m.Cfg.Dim)
	}
	w, err := LoadVisionWeights(g, cfg)
	if err != nil {
		g.Close()
		return err
	}
	m.visionFile = g
	m.vision = NewVisionTower(cfg, w)
	return nil
}

// HasVision reports whether a projector was opened.
func (m *Model) HasVision() bool { return m.vision != nil }

// EncodeImage decodes one encoded image and runs the tower over it. What comes
// back is one row per soft token, each as wide as the model's embedding.
func (m *Model) EncodeImage(data []byte) ([][]float32, error) {
	if m.vision == nil {
		return nil, fmt.Errorf("gemma: this model was opened without a projector, so it cannot look at a picture")
	}
	im, err := imageio.Decode(bytes.NewReader(data))
	if err != nil {
		return nil, err
	}
	return m.vision.Encode(im), nil
}

// Prompt is a tokenized conversation with the pictures already in it: three
// slices of the same length, and where each picture's soft tokens sit.
//
// Tokens is what the cache holds and what the per-layer lookup would read.
// Embeds is nil at every position but a soft one, where it is the tower's own
// row. PLE is Tokens with the padding token at the soft positions, because
// that is the per-layer input llama.cpp gives an embedding batch. Spans are
// the inclusive index ranges the model attends to in both directions.
type Prompt struct {
	Tokens []int32
	Embeds [][]float32
	PLE    []int32
	Spans  [][2]int
}

// BuildPrompt splices encoded images into an encoded prompt.
//
// tokens is the rendered conversation as the vocabulary encoded it, markers
// and all: RenderChat wrote an empty <|image><image|> pair for each picture,
// and this fills the pairs in, in order. images[i] is what EncodeImage
// returned for the i-th of them.
func (m *Model) BuildPrompt(tokens []int32, images [][][]float32) (*Prompt, error) {
	open, closed, soft := m.Cfg.ImageOpen, m.Cfg.ImageClose, m.Cfg.ImageSoft
	if len(images) > 0 && (open == 0 || closed == 0 || soft == 0) {
		return nil, fmt.Errorf("gemma: this vocabulary has no image markers, so nothing can carry a picture")
	}
	p := &Prompt{}
	seen := 0
	for i := 0; i < len(tokens); i++ {
		p.Tokens = append(p.Tokens, tokens[i])
		p.PLE = append(p.PLE, tokens[i])
		p.Embeds = append(p.Embeds, nil)
		if tokens[i] != open {
			continue
		}
		if seen >= len(images) {
			return nil, fmt.Errorf("gemma: the prompt opens %d pictures and %d were given", seen+1, len(images))
		}
		rows := images[seen]
		seen++
		first := len(p.Tokens)
		for _, row := range rows {
			if len(row) != m.Cfg.Dim {
				return nil, fmt.Errorf("gemma: a soft token is %d wide and this model reads %d", len(row), m.Cfg.Dim)
			}
			p.Tokens = append(p.Tokens, soft)
			p.PLE = append(p.PLE, 0)
			p.Embeds = append(p.Embeds, row)
		}
		if len(rows) > 0 {
			p.Spans = append(p.Spans, [2]int{first, first + len(rows) - 1})
		}
		// The closing marker follows in the rendered string, and is copied by
		// the next turn of the loop.
	}
	if seen != len(images) {
		return nil, fmt.Errorf("gemma: %d pictures were given and the prompt opens %d", len(images), seen)
	}
	return p, nil
}

// Places says where each token of the prompt goes, starting at startPos, with
// the tokens of one picture given the whole picture to look at.
func (p *Prompt) Places(cache *Cache, startPos int) []Place {
	at := Run(cache, startPos, len(p.Tokens))
	for _, span := range p.Spans {
		for i := span[0]; i <= span[1]; i++ {
			at[i].Until = startPos + span[1]
		}
	}
	return at
}

// ForwardPrompt reads the whole of it in one pass and returns one hidden state
// per position.
func (m *Model) ForwardPrompt(p *Prompt, startPos int) [][]float32 {
	return m.ForwardEmbedded(p.Tokens, p.Embeds, p.PLE, p.Places(m.cache, startPos))
}
