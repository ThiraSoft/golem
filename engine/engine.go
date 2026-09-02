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
	"github.com/ThiraSoft/golem/qwen35"
	"github.com/ThiraSoft/golem/sample"
	"github.com/ThiraSoft/golem/tensors"
	"github.com/ThiraSoft/golem/token/bpe"
	"github.com/ThiraSoft/golem/token/bytebpe"
)

// Vocabulary is the part of a tokenizer a conversation drives. The two
// implementations are not interchangeable and are not chosen by hand: Gemma's
// is token/bpe — BPE over raw UTF-8 with SentencePiece's whitespace escaping —
// and Qwen's is token/bytebpe, byte-level BPE with the qwen2 pre-tokenizer.
// Each engine loads its own; qwen35 reads Qwen's.
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
	// ForwardSlots carries tokens of several conversations through one pass:
	// slots and positions say, for each token, which cache it writes to and
	// where in it. What comes back is what each token would have been given
	// alone.
	ForwardSlots(tokens []int32, slots, positions []int) [][]float32
	Logits(hidden, out []float32)
	// LogitsBatch scores several states in one read of the head, which is what
	// several conversations drawing at once want: the head is the largest
	// matrix in the model.
	LogitsBatch(hidden [][]float32, out [][]float32)
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

// vulkanHead is implemented by the engines whose logit head can move to a
// device. It is not part of Forward: an engine that cannot do it is not
// broken, and a caller that never asks should not have to know the method
// exists.
type vulkanHead interface {
	UseVulkanHead() error
	VulkanHead() bool
	UseVulkanStack() error
	VulkanStack() bool
}

// vulkanVision is implemented by the engines whose image tower can move to a
// device as well. It is apart from vulkanHead because the two are opened at
// different moments — a projector is a second file, and the commands do not
// agree on whether it is read before the device is chosen or after.
type vulkanVision interface {
	UseVisionVulkan() error
	VisionVulkan() (on, resident bool)
}

// UseVulkan moves to a Vulkan device everything of this engine that can go:
// the logit head, and the expert stacks of a mixture. Both are read in full or
// nearly so for every token drawn, and both are bandwidth on a CPU.
//
// It is all or nothing per part, and it fails rather than falling back: a
// model half on a card the caller believed it was wholly on is a model whose
// speed nobody can explain.
// noDrafter is an engine whose checkpoint carries a prediction block it can be
// told to leave behind. Only Qwen3.8 has one.
type noDrafter interface{ SkipDraftBlock() }

// SkipDraftBlock tells an engine that will not speculate not to spend the card
// on what speculation needs. It has to be called before UseVulkan, because
// what it decides is what gets uploaded; after it, it is a no-op that lies.
func (m *Model) SkipDraftBlock() {
	if d, ok := m.Forward.(noDrafter); ok {
		d.SkipDraftBlock()
	}
}

func (m *Model) UseVulkan() error {
	h, ok := m.Forward.(vulkanHead)
	if !ok {
		return fmt.Errorf("engine: %s has nothing that can move to a device", m.Name)
	}
	if err := h.UseVulkanStack(); err != nil {
		return err
	}
	if err := h.UseVulkanHead(); err != nil {
		return err
	}
	return m.useVisionVulkan()
}

// useVisionVulkan puts an already-opened image tower on the device. It is a
// no-op for an engine that has no such tower and for one whose projector has
// not been opened yet — the second case is why OpenProjector calls it too.
func (m *Model) useVisionVulkan() error {
	v, ok := m.Forward.(vulkanVision)
	if !ok {
		return nil
	}
	return v.UseVisionVulkan()
}

// VisionVulkan says whether the image tower is on a device and whether the
// whole of it is resident there, for the line printed at startup. A tower that
// is not resident still runs on the card; its weights cross the bus once an
// image instead of once.
func (m *Model) VisionVulkan() (on, resident bool) {
	v, ok := m.Forward.(vulkanVision)
	if !ok {
		return false, false
	}
	return v.VisionVulkan()
}

// Vulkan says what is on a device, for the line printed at startup.
func (m *Model) Vulkan() (head, blocks bool) {
	h, ok := m.Forward.(vulkanHead)
	if !ok {
		return false, false
	}
	return h.VulkanHead(), h.VulkanStack()
}

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
	case "qwen35", "qwen3.8":
		m, err = openQwen35(g, maxContext)
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

func openQwen35(g *tensors.GGUF, maxContext int) (*Model, error) {
	inner, err := qwen35.New(g, maxContext)
	if err != nil {
		return nil, err
	}
	vocab, err := bytebpe.Load(g)
	if err != nil {
		return nil, err
	}
	// The three identifiers a picture is written with. The vocabulary is opened
	// here and not inside the engine, so this is where the engine is told.
	if start, ok := vocab.ID(qwen35.VisionStart); ok {
		if pad, ok := vocab.ID(qwen35.ImagePad); ok {
			if end, ok := vocab.ID(qwen35.VisionEnd); ok {
				inner.SetVisionMarkers(start, pad, end)
			}
		}
	}
	return &Model{
		Forward: inner, Vocab: vocab, Template: qwen35.NewTemplate(),
		Window: 0, Vocabulary: inner.Cfg.Vocab,
		Blocks: len(inner.Cfg.Blocks), Sampling: inner.Cfg.Sampling,
		closer: inner,
	}, nil
}

func unknownArchitecture(arch string) error {
	return fmt.Errorf("engine: architecture %q is not implemented; gemma4, qwen3, and qwen35 are", arch)
}
