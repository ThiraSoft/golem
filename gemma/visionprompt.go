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

// OpenProjector maps a projector file and binds whichever encoders it
// declares: a vision tower, an audio encoder, or both. One file carries the
// two, and which of them it holds is the file's business rather than the
// caller's — the commands pass -mmproj and ask afterwards what the model can
// do.
//
// It is an error only when the file declares neither, or when an encoder it
// does declare produces rows of the wrong width for this model.
func (m *Model) OpenProjector(path string) error {
	g, err := tensors.OpenGGUF(path)
	if err != nil {
		return err
	}
	hasVision, hasAudio := HasVision(g), HasAudio(g)
	if !hasVision && !hasAudio {
		g.Close()
		return fmt.Errorf("gemma: %s declares neither a vision encoder nor an audio one", path)
	}
	var vision *VisionTower
	var audio *AudioTower
	if hasVision {
		cfg, err := LoadVisionConfig(g)
		if err != nil {
			g.Close()
			return err
		}
		if cfg.ProjDim != m.Cfg.Dim {
			g.Close()
			return fmt.Errorf("gemma: the projector produces %d-wide picture tokens and this model reads %d; they are not a pair",
				cfg.ProjDim, m.Cfg.Dim)
		}
		w, err := LoadVisionWeights(g, cfg)
		if err != nil {
			g.Close()
			return err
		}
		vision = NewVisionTower(cfg, w)
	}
	if hasAudio {
		cfg, err := LoadAudioConfig(g)
		if err != nil {
			g.Close()
			return err
		}
		if cfg.ProjDim != m.Cfg.Dim {
			g.Close()
			return fmt.Errorf("gemma: the projector produces %d-wide sound tokens and this model reads %d; they are not a pair",
				cfg.ProjDim, m.Cfg.Dim)
		}
		w, err := LoadAudioWeights(g, cfg)
		if err != nil {
			g.Close()
			return err
		}
		audio = NewAudioTower(cfg, w)
	}
	m.projFile = g
	m.vision = vision
	m.audio = audio
	return nil
}

// HasVision reports whether an opened projector carries a vision tower.
func (m *Model) HasVision() bool { return m.vision != nil }

// HasAudio reports whether an opened projector carries an audio encoder.
func (m *Model) HasAudio() bool { return m.audio != nil }

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

// BuildPrompt splices encoded media into an encoded prompt.
//
// tokens is the rendered conversation as the vocabulary encoded it, markers
// and all: RenderChat wrote an empty <|image><image|> pair for each picture
// and an empty <|audio><audio|> pair for each recording, and this fills the
// pairs in, in order. images[i] is what EncodeImage returned for the i-th
// picture and audio[i] what EncodeAudio returned for the i-th recording; the
// two are counted separately because the markers are.
func (m *Model) BuildPrompt(tokens []int32, images, audio [][][]float32) (*Prompt, error) {
	if len(images) > 0 && (m.Cfg.ImageOpen == 0 || m.Cfg.ImageClose == 0 || m.Cfg.ImageSoft == 0) {
		return nil, fmt.Errorf("gemma: this vocabulary has no image markers, so nothing can carry a picture")
	}
	if len(audio) > 0 && (m.Cfg.AudioOpen == 0 || m.Cfg.AudioClose == 0 || m.Cfg.AudioSoft == 0) {
		return nil, fmt.Errorf("gemma: this vocabulary has no audio markers, so nothing can carry a recording")
	}
	p := &Prompt{}
	seenImages, seenAudio := 0, 0
	for i := 0; i < len(tokens); i++ {
		p.Tokens = append(p.Tokens, tokens[i])
		p.PLE = append(p.PLE, tokens[i])
		p.Embeds = append(p.Embeds, nil)

		var rows [][]float32
		switch {
		case m.Cfg.ImageOpen != 0 && tokens[i] == m.Cfg.ImageOpen:
			if seenImages >= len(images) {
				return nil, fmt.Errorf("gemma: the prompt opens %d pictures and %d were given", seenImages+1, len(images))
			}
			rows = images[seenImages]
			seenImages++
		case m.Cfg.AudioOpen != 0 && tokens[i] == m.Cfg.AudioOpen:
			if seenAudio >= len(audio) {
				return nil, fmt.Errorf("gemma: the prompt opens %d recordings and %d were given", seenAudio+1, len(audio))
			}
			rows = audio[seenAudio]
			seenAudio++
		default:
			continue
		}

		soft := m.Cfg.ImageSoft
		if tokens[i] == m.Cfg.AudioOpen {
			soft = m.Cfg.AudioSoft
		}
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
	if seenImages != len(images) {
		return nil, fmt.Errorf("gemma: %d pictures were given and the prompt opens %d", len(images), seenImages)
	}
	if seenAudio != len(audio) {
		return nil, fmt.Errorf("gemma: %d recordings were given and the prompt opens %d", len(audio), seenAudio)
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

// Slice is the run of positions [from, to) as a prompt of its own, with the
// spans that fall inside it moved to match.
//
// It exists because a long prompt is fed in batches, and a picture must not be
// cut across two of them: every key of a span has to be in the cache before
// any of its queries is scored, which is true within one batch and false
// across two. Whoever cuts asks Boundary where it may.
func (p *Prompt) Slice(from, to int) *Prompt {
	out := &Prompt{
		Tokens: p.Tokens[from:to],
		Embeds: p.Embeds[from:to],
		PLE:    p.PLE[from:to],
	}
	for _, span := range p.Spans {
		if span[0] >= from && span[1] < to {
			out.Spans = append(out.Spans, [2]int{span[0] - from, span[1] - from})
		}
	}
	return out
}

// Boundary is the largest cut no further than want that falls outside every
// picture: a batch may end there. It never returns from itself, so a caller
// looping on it makes progress even when one picture is larger than a batch.
func (p *Prompt) Boundary(from, want int) int {
	if want > len(p.Tokens) {
		want = len(p.Tokens)
	}
	for _, span := range p.Spans {
		// A cut inside a span, or one that would leave its start behind,
		// moves out to the end of it.
		if want > span[0] && want <= span[1] {
			want = span[1] + 1
		}
	}
	if want <= from {
		want = from + 1
	}
	return want
}

// ForwardEmbeddedSlots is ForwardPrompt for a caller that names the caches by
// index rather than by pointer, and that may carry several conversations in
// one pass — which is what a server does. slots and positions say where each
// token goes, until how far forward it may look.
func (m *Model) ForwardEmbeddedSlots(tokens []int32, embeds [][]float32, ple []int32,
	slots, positions, until []int) [][]float32 {
	at := make([]Place, len(tokens))
	for i := range tokens {
		at[i] = Place{Cache: m.Slot(slots[i]), Pos: positions[i], Until: until[i]}
	}
	return m.ForwardEmbedded(tokens, embeds, ple, at)
}

// Until is the upper bound of each position of the prompt, given the position
// its first token is fed at: its own for text, the end of the picture for a
// token inside one.
func (p *Prompt) Until(startPos int) []int {
	out := make([]int, len(p.Tokens))
	for i := range out {
		out[i] = startPos + i
	}
	for _, span := range p.Spans {
		for i := span[0]; i <= span[1]; i++ {
			out[i] = startPos + span[1]
		}
	}
	return out
}
