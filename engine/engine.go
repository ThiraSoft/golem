// Package engine picks which engine reads a file, so that a command does not
// have to.
//
// A GGUF says what it is under general.architecture. Everything a command needs
// past that — the forward pass, the vocabulary, the chat template, the numbers
// for the line printed at startup — has the same shape whichever engine
// answered, and that shape is Model.
package engine

import (
	"fmt"

	"github.com/ThiraSoft/golem/chat"
	"github.com/ThiraSoft/golem/gemma"
	"github.com/ThiraSoft/golem/qwen"
	"github.com/ThiraSoft/golem/sample"
	"github.com/ThiraSoft/golem/tensors"
	"github.com/ThiraSoft/golem/token/bpe"
)

// Forward is the part of an engine a conversation drives.
type Forward interface {
	ForwardBatch(tokens []int32, startPos int) [][]float32
	Logits(hidden, out []float32)
	Reset()
}

// Model is one opened checkpoint, as a command sees it.
type Model struct {
	Forward  Forward
	Vocab    *bpe.Vocab
	Template chat.Template
	// Name is the architecture the file declares.
	Name string
	// Window is the largest sliding window any block uses, and 0 when every
	// block is global. A rewind of the cache has to respect it.
	Window int
	// Vocabulary is how many logits a pass produces.
	Vocabulary int
	// Blocks is how many there are, for the line printed at startup.
	Blocks int
	// Sampling is what the file asks to be sampled with.
	Sampling sample.Params

	closer interface{ Close() error }
}

// Close releases the model and the file behind it.
func (m *Model) Close() error { return m.closer.Close() }

// Open reads the architecture and hands the file to the engine that implements
// it. maxContext caps the cache; the files declare far more than a machine here
// would survive.
func Open(path string, maxContext int) (*Model, error) {
	g, err := tensors.OpenGGUF(path)
	if err != nil {
		return nil, err
	}
	arch, err := g.String("general.architecture")
	if err != nil {
		g.Close()
		return nil, err
	}
	var m *Model
	switch arch {
	case "gemma4":
		m, err = openGemma(g, maxContext)
	case "qwen3":
		m, err = openQwen(g, maxContext)
	default:
		err = unknownArchitecture(arch)
	}
	if err != nil {
		g.Close()
		return nil, err
	}
	m.Name = arch
	// The vocabulary is read from the same file, and the model owns it now, so
	// a failure here closes through the model rather than through the file.
	m.Vocab, err = bpe.Load(g)
	if err != nil {
		m.Close()
		return nil, err
	}
	return m, nil
}

func openGemma(g *tensors.GGUF, maxContext int) (*Model, error) {
	inner, err := gemma.New(g, maxContext)
	if err != nil {
		return nil, err
	}
	// The largest sliding window is what a rewind of the cache has to respect.
	window := 0
	for _, b := range inner.Cfg.Blocks {
		if b.Window && b.WindowSize > window {
			window = b.WindowSize
		}
	}
	return &Model{
		Forward: inner, Template: gemma.NewTemplate(inner.Cfg),
		Window: window, Vocabulary: inner.Cfg.Vocab,
		Blocks: len(inner.Cfg.Blocks), Sampling: inner.Cfg.Sampling,
		closer: inner,
	}, nil
}

func openQwen(g *tensors.GGUF, maxContext int) (*Model, error) {
	inner, err := qwen.New(g, maxContext)
	if err != nil {
		return nil, err
	}
	// Every Qwen3 block attends to the whole context: there is no window to
	// respect, and a rewind costs nothing.
	return &Model{
		Forward: inner, Template: qwen.NewTemplate(inner.Cfg),
		Window: 0, Vocabulary: inner.Cfg.Vocab,
		Blocks: len(inner.Cfg.Blocks), Sampling: inner.Cfg.Sampling,
		closer: inner,
	}, nil
}

func unknownArchitecture(arch string) error {
	return fmt.Errorf("engine: architecture %q is not implemented; gemma4 and qwen3 are", arch)
}
