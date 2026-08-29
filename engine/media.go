package engine

// Pictures and sound, for the engines that have them.
//
// It is an optional interface rather than a part of Forward, because one of
// the engines here has neither eyes nor ears and there is no honest thing for
// it to return. A command asks, and is told no in a sentence naming the
// architecture rather than by a method that fails at run time.

import (
	"fmt"

	"github.com/ThiraSoft/golem/gemma"
	"github.com/ThiraSoft/golem/qwen35"
)

// Media is what a command drives to put a picture or a recording in a prompt.
//
// EncodeImage and EncodeAudio run the matching encoder over one file, still in
// the bytes whoever wrote it wrote, and return one row per soft token. Prompt
// takes the tokens of a rendered conversation — markers and all — with the
// rows of each picture and each recording in order, and ForwardPrompt reads
// the result in one pass.
//
// A model whose projector carries only one of the two answers the other with
// an error rather than with silence; CanSee and CanHear say which in advance.
type Media interface {
	CanSee() bool
	CanHear() bool
	EncodeImage(data []byte) ([][]float32, error)
	EncodeAudio(data []byte) ([][]float32, error)
	Prompt(tokens []int32, images, audio [][][]float32) (Prompt, error)
	ForwardPrompt(p Prompt, startPos int) [][]float32
	// ForwardChunk is the same pass for a server, which names its caches by
	// index and may carry several conversations at once.
	ForwardChunk(c Chunk) [][]float32
}

// A Chunk is one pass of a batch, as an engine that carries media reads it.
//
// Tokens and Positions are what every pass has. Embeds is the row given at a
// position instead of the token table's, which is how a picture reaches the
// model. The last three are what one engine needs and the other does not, and
// they travel together because a batch is one pass whatever its conversations
// hold: Gemma fills PLE and Until, Qwen3.8 fills Axes, and each leaves the
// other's nil.
type Chunk struct {
	Tokens    []int32
	Embeds    [][]float32
	Slots     []int
	Positions []int

	PLE   []int32
	Until []int
	Axes  [][3]int
}

// gemmaMedia is Gemma's adapter, and one of the two places that names an
// engine's own types.
type gemmaMedia struct{ m *gemma.Model }

func (v gemmaMedia) CanSee() bool                                 { return v.m.HasVision() }
func (v gemmaMedia) CanHear() bool                                { return v.m.HasAudio() }
func (v gemmaMedia) EncodeImage(data []byte) ([][]float32, error) { return v.m.EncodeImage(data) }
func (v gemmaMedia) EncodeAudio(data []byte) ([][]float32, error) { return v.m.EncodeAudio(data) }

func (v gemmaMedia) Prompt(tokens []int32, images, audio [][][]float32) (Prompt, error) {
	p, err := v.m.BuildPrompt(tokens, images, audio)
	if err != nil {
		return nil, err
	}
	return gemmaPrompt{p}, nil
}

func (v gemmaMedia) ForwardPrompt(p Prompt, startPos int) [][]float32 {
	switch q := p.(type) {
	case gemmaPrompt:
		return v.m.ForwardPrompt(q.p, startPos)
	default:
		// A text-only prompt, which needs none of the splicing.
		return v.m.ForwardBatch(p.Tokens(), startPos)
	}
}

func (v gemmaMedia) ForwardChunk(c Chunk) [][]float32 {
	return v.m.ForwardEmbeddedSlots(c.Tokens, c.Embeds, c.PLE, c.Slots, c.Positions, c.Until)
}

// gemmaPrompt carries Gemma's own prompt behind the interface. Slice returns
// the interface rather than the concrete type, which is the one thing Go will
// not do for us.
type gemmaPrompt struct{ p *gemma.Prompt }

func (g gemmaPrompt) Len() int                    { return len(g.p.Tokens) }
func (g gemmaPrompt) Tokens() []int32             { return g.p.Tokens }
func (g gemmaPrompt) Boundary(from, want int) int { return g.p.Boundary(from, want) }
func (g gemmaPrompt) Slice(a, b int) Prompt       { return gemmaPrompt{g.p.Slice(a, b)} }
func (g gemmaPrompt) Embeds() [][]float32         { return g.p.Embeds }

func (g gemmaPrompt) Extras(startPos int) ([]int32, []int, [][3]int) {
	return g.p.PLE, g.p.Until(startPos), nil
}

// qwenMedia is Qwen3.8's. It sees and does not hear: its projector declares a
// vision encoder and no audio one.
type qwenMedia struct{ m *qwen35.Model }

func (v qwenMedia) CanSee() bool  { return v.m.HasVision() }
func (v qwenMedia) CanHear() bool { return false }

func (v qwenMedia) EncodeImage(data []byte) ([][]float32, error) { return v.m.EncodeImage(data) }

func (v qwenMedia) EncodeAudio(data []byte) ([][]float32, error) {
	return nil, fmt.Errorf("engine: qwen35's projector carries no audio encoder")
}

func (v qwenMedia) Prompt(tokens []int32, images, audio [][][]float32) (Prompt, error) {
	if len(audio) > 0 {
		return nil, fmt.Errorf("engine: qwen35's projector carries no audio encoder")
	}
	grids := make([][2]int, len(images))
	for i := range images {
		g, ok := v.m.GridOf(i)
		if !ok {
			return nil, fmt.Errorf("engine: picture %d was not encoded by this model", i)
		}
		grids[i] = g
	}
	p, err := v.m.BuildPrompt(tokens, images, grids)
	if err != nil {
		return nil, err
	}
	return qwenPrompt{p}, nil
}

func (v qwenMedia) ForwardPrompt(p Prompt, startPos int) [][]float32 {
	switch q := p.(type) {
	case qwenPrompt:
		return v.m.ForwardPrompt(q.p, startPos)
	default:
		return v.m.ForwardBatch(p.Tokens(), startPos)
	}
}

func (v qwenMedia) ForwardChunk(c Chunk) [][]float32 {
	at := make([]qwen35.Place, len(c.Tokens))
	for i := range at {
		q := c.Positions[i]
		axes := [3]int{q, q, q}
		if c.Axes != nil {
			axes = c.Axes[i]
		}
		at[i] = qwen35.Place{Slot: c.Slots[i], Pos: q, T: axes[0], H: axes[1], W: axes[2]}
	}
	return v.m.ForwardEmbeddedPlaces(c.Tokens, c.Embeds, at)
}

// qwenPrompt carries Qwen3.8's.
type qwenPrompt struct{ p *qwen35.Prompt }

func (q qwenPrompt) Len() int        { return len(q.p.Tokens) }
func (q qwenPrompt) Tokens() []int32 { return q.p.Tokens }
func (q qwenPrompt) Slice(a, b int) Prompt {
	return qwenPrompt{q.p.Slice(a, b)}
}

func (q qwenPrompt) Embeds() [][]float32 { return q.p.Embeds }

// Extras are the three axes, which is all this engine's pass needs beyond the
// cache index. A picture holds one time and spreads over the grid; text
// advances on every axis at once.
func (q qwenPrompt) Extras(startPos int) ([]int32, []int, [][3]int) {
	at := q.p.Places(0, startPos)
	axes := make([][3]int, len(at))
	for i, p := range at {
		axes[i] = [3]int{p.T, p.H, p.W}
	}
	return nil, nil, axes
}

// Boundary is want: this model attends causally over a picture as over text,
// so a pass may end anywhere. Gemma's may not — its picture is attended to in
// both directions and every key of it has to be in the cache before any query
// of it is scored.
func (q qwenPrompt) Boundary(from, want int) int {
	if want > len(q.p.Tokens) {
		return len(q.p.Tokens)
	}
	return want
}

// OpenProjector gives this model a projector, whichever encoders it carries.
// An engine that cannot take one says so by name.
func (m *Model) OpenProjector(path string) error {
	switch inner := m.Forward.(type) {
	case *gemma.Model:
		return inner.OpenProjector(path)
	case *qwen35.Model:
		return inner.OpenProjector(path)
	}
	return fmt.Errorf("engine: %s cannot be given a projector; gemma4 and qwen35 can", m.Name)
}

// Media answers whether this model can look or listen, and how.
func (m *Model) Media() (Media, bool) {
	switch inner := m.Forward.(type) {
	case *gemma.Model:
		if !inner.HasVision() && !inner.HasAudio() {
			return nil, false
		}
		return gemmaMedia{inner}, true
	case *qwen35.Model:
		if !inner.HasVision() {
			return nil, false
		}
		return qwenMedia{inner}, true
	}
	return nil, false
}
