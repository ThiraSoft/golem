package engine

// Vision, for the engines that have it.
//
// It is an optional interface rather than a part of Forward, because one of
// the two engines here cannot see and there is no honest thing for it to
// return. A command asks, and is told no in a sentence naming the
// architecture rather than by a method that fails at run time.

import (
	"fmt"

	"github.com/ThiraSoft/golem/gemma"
)

// Vision is what a command drives to put a picture in a prompt.
//
// EncodeImage runs the encoder over one image, still in the bytes some encoder
// wrote, and returns one row per soft token. Prompt takes the tokens of a
// rendered conversation — markers and all — and the rows of each picture in
// order, and Forward reads the result in one pass.
type Vision interface {
	EncodeImage(data []byte) ([][]float32, error)
	Prompt(tokens []int32, images [][][]float32) (*gemma.Prompt, error)
	ForwardPrompt(p *gemma.Prompt, startPos int) [][]float32
}

// visionOf is the adapter, and the one place that names an engine's own types.
type visionOf struct{ m *gemma.Model }

func (v visionOf) EncodeImage(data []byte) ([][]float32, error) { return v.m.EncodeImage(data) }

func (v visionOf) Prompt(tokens []int32, images [][][]float32) (*gemma.Prompt, error) {
	return v.m.BuildPrompt(tokens, images)
}

func (v visionOf) ForwardPrompt(p *gemma.Prompt, startPos int) [][]float32 {
	return v.m.ForwardPrompt(p, startPos)
}

// OpenVision gives this model a projector. An engine that cannot take one says
// so by name.
func (m *Model) OpenVision(path string) error {
	inner, ok := m.Forward.(*gemma.Model)
	if !ok {
		return fmt.Errorf("engine: %s cannot be given a projector; gemma4 can", m.Name)
	}
	return inner.OpenVision(path)
}

// Vision answers whether this model can look at a picture, and how.
func (m *Model) Vision() (Vision, bool) {
	inner, ok := m.Forward.(*gemma.Model)
	if !ok || !inner.HasVision() {
		return nil, false
	}
	return visionOf{inner}, true
}
