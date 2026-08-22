package engine

// Pictures and sound, for the engines that have them.
//
// It is an optional interface rather than a part of Forward, because one of
// the two engines here has neither eyes nor ears and there is no honest thing
// for it to return. A command asks, and is told no in a sentence naming the
// architecture rather than by a method that fails at run time.

import (
	"fmt"

	"github.com/ThiraSoft/golem/gemma"
)

// Media is what a command drives to put a picture or a recording in a prompt.
//
// EncodeImage and EncodeAudio run the matching encoder over one file, still in
// the bytes whoever wrote it wrote, and return one row per soft token. Prompt
// takes the tokens of a rendered conversation — markers and all — with the
// rows of each picture and each recording in order, and Forward reads the
// result in one pass.
//
// A model whose projector carries only one of the two answers the other with
// an error rather than with silence; CanSee and CanHear say which in advance.
type Media interface {
	CanSee() bool
	CanHear() bool
	EncodeImage(data []byte) ([][]float32, error)
	EncodeAudio(data []byte) ([][]float32, error)
	Prompt(tokens []int32, images, audio [][][]float32) (*gemma.Prompt, error)
	ForwardPrompt(p *gemma.Prompt, startPos int) [][]float32
	// ForwardEmbeddedSlots is the same pass for a server, which names its
	// caches by index and may carry several conversations at once.
	ForwardEmbeddedSlots(tokens []int32, embeds [][]float32, ple []int32,
		slots, positions, until []int) [][]float32
}

// mediaOf is the adapter, and the one place that names an engine's own types.
type mediaOf struct{ m *gemma.Model }

func (v mediaOf) CanSee() bool  { return v.m.HasVision() }
func (v mediaOf) CanHear() bool { return v.m.HasAudio() }

func (v mediaOf) EncodeImage(data []byte) ([][]float32, error) { return v.m.EncodeImage(data) }
func (v mediaOf) EncodeAudio(data []byte) ([][]float32, error) { return v.m.EncodeAudio(data) }

func (v mediaOf) Prompt(tokens []int32, images, audio [][][]float32) (*gemma.Prompt, error) {
	return v.m.BuildPrompt(tokens, images, audio)
}

func (v mediaOf) ForwardPrompt(p *gemma.Prompt, startPos int) [][]float32 {
	return v.m.ForwardPrompt(p, startPos)
}

func (v mediaOf) ForwardEmbeddedSlots(tokens []int32, embeds [][]float32, ple []int32,
	slots, positions, until []int) [][]float32 {
	return v.m.ForwardEmbeddedSlots(tokens, embeds, ple, slots, positions, until)
}

// OpenProjector gives this model a projector, whichever encoders it carries.
// An engine that cannot take one says so by name.
func (m *Model) OpenProjector(path string) error {
	inner, ok := m.Forward.(*gemma.Model)
	if !ok {
		return fmt.Errorf("engine: %s cannot be given a projector; gemma4 can", m.Name)
	}
	return inner.OpenProjector(path)
}

// Media answers whether this model can look or listen, and how.
func (m *Model) Media() (Media, bool) {
	inner, ok := m.Forward.(*gemma.Model)
	if !ok || (!inner.HasVision() && !inner.HasAudio()) {
		return nil, false
	}
	return mediaOf{inner}, true
}
