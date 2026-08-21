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
	"github.com/ThiraSoft/golem/token/bytebpe"
)

// Vocabulary is the part of a tokenizer a conversation drives. The two
// implementations are not interchangeable and are not chosen by hand: Gemma's
// vocabulary is SentencePiece, Qwen's is byte-level BPE with the qwen2
// pre-tokenizer, and each engine loads its own.
type Vocabulary interface {
	Encode(text string, addBOS, parseSpecial bool) []int32
	Piece(id int32, special bool) string
	IsEOG(id int32) bool
}

// Forward is the part of an engine a conversation drives.
//
// Slots are how several conversations share one set of weights: a slot is a
// cache, UseSlot says which one the next pass writes to, and Reset forgets the
// one in use. A model that was never asked for more than one answers 1 and
// takes UseSlot(0), so a caller that does not care never has to know.
type Forward interface {
	ForwardBatch(tokens []int32, startPos int) [][]float32
	Logits(hidden, out []float32)
	Reset()
	SetSlots(n int) error
	Slots() int
	SlotContext() int
	UseSlot(i int)
}

// Model is one opened checkpoint, as a command sees it.
type Model struct {
	Forward  Forward
	Vocab    Vocabulary
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

// Slots is how many conversations the model holds at once, and SlotContext
// how many positions each of them has. They are read off the engine rather
// than copied here, so that nothing can disagree with it.
func (m *Model) Slots() int { return m.Forward.Slots() }

// SlotContext is how many positions one conversation has.
func (m *Model) SlotContext() int { return m.Forward.SlotContext() }

// Close releases the model and the file behind it.
func (m *Model) Close() error { return m.closer.Close() }

// Open reads the architecture and hands the file to the engine that implements
// it. maxContext caps the cache; the files declare far more than a machine here
// would survive.
//
// slots, when given, cuts that context into that many independent
// conversations, the way llama.cpp's -parallel does: the memory is what the
// caller allowed, and each conversation gets its share of the positions.
func Open(path string, maxContext int, slots ...int) (*Model, error) {
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
	if len(slots) > 0 {
		if err := m.Forward.SetSlots(slots[0]); err != nil {
			m.Close()
			return nil, err
		}
	}
	return m, nil
}

func openGemma(g *tensors.GGUF, maxContext int) (*Model, error) {
	inner, err := gemma.New(g, maxContext)
	if err != nil {
		return nil, err
	}
	vocab, err := bpe.Load(g)
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
		Forward: inner, Vocab: vocab, Template: gemma.NewTemplate(inner.Cfg),
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
	vocab, err := bytebpe.Load(g)
	if err != nil {
		return nil, err
	}
	// Every Qwen3 block attends to the whole context: there is no window to
	// respect, and a rewind costs nothing.
	return &Model{
		Forward: inner, Vocab: vocab, Template: qwen.NewTemplate(inner.Cfg),
		Window: 0, Vocabulary: inner.Cfg.Vocab,
		Blocks: len(inner.Cfg.Blocks), Sampling: inner.Cfg.Sampling,
		closer: inner,
	}, nil
}

func unknownArchitecture(arch string) error {
	return fmt.Errorf("engine: architecture %q is not implemented; gemma4 and qwen3 are", arch)
}
