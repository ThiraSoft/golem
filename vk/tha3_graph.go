package vk

// Talking Head(?) Anime 3 on the card: the graph.
//
// The poser is five networks of about four hundred small operations, and on
// the card each operation is a dispatch that reads and writes tensors. Rather
// than a buffer per tensor, every activation lives in one device-local arena
// at an offset, and every kernel binds the same three buffers. A tensor, a
// channel slice of it, a crop's source: all of them are an offset in a push
// constant, and each pipeline needs one descriptor set for its whole life.
//
// The graph is described first, as a list of operations over handles, in the
// order they run. Then plan gives each tensor its place: a tensor is placed
// when an operation creates it and its floats return to the pool after the
// last operation that uses it, so the arena holds what is alive at once and
// not everything a pass ever makes.
//
// There are two phases. The image phase runs once per picture (the eyebrow
// decomposer), the pose phase once per pose. A tensor the image phase makes
// and the pose phase reads is pinned, since the pose phase runs many times
// over it; so is an input, which the host writes whenever it likes.
//
// The pose phase may also be entered part way, at an Entry, when what comes
// before it would compute the same as last time. A tensor made before an
// entry and read after it is pinned for the same reason: a pass that starts
// at the entry reads what an earlier pass left there.

import (
	"fmt"
	"sort"
)

// The two passes a graph records.
const (
	THA3ImagePhase = 0
	THA3PosePhase  = 1
)

// THA3Act is the non-linearity a kernel applies to what it writes.
type THA3Act uint32

const (
	THA3None THA3Act = iota
	THA3ReLU
	THA3Leaky // slope 0.1, the rotator's and the editor's
	THA3Sigmoid
	THA3Tanh
)

// THA3Tensor is a handle on an activation, channel by channel like the CPU's
// tha3.Tensor. A handle made by Channels shares its tensor's floats.
type THA3Tensor struct {
	id, ch0 int
	C, H, W int
}

// Channels is channels [from, to) of t, sharing its floats.
func (t THA3Tensor) Channels(from, to int) THA3Tensor {
	if from < 0 || to > t.C || from >= to {
		panic(fmt.Sprintf("vk: channels [%d, %d) of a tensor of %d", from, to, t.C))
	}
	t.ch0 += from
	t.C = to - from
	return t
}

// THA3Weights is a range of the weights buffer, in floats.
type THA3Weights struct{ off, n int }

type tha3Tensor struct {
	floats int // of the whole tensor, whatever views of it exist
	input  bool
	phase  int
	off    int
}

type tha3Op struct {
	phase   int
	uses    []int
	creates []int
	record  func(r *Recorder, x *THA3Runner)
	stamp   string
}

// THA3Graph is a poser described as operations over tensors.
type THA3Graph struct {
	tensors []tha3Tensor
	ops     []tha3Op
	weights []float32
	phase   int
	traces  []string
	traceT  []THA3Tensor
	outputs []THA3Tensor
	stamps  int
	entries []int // ops where a pose pass may start, besides the first
}

// NewTHA3Graph starts a graph in the image phase.
func NewTHA3Graph() *THA3Graph { return &THA3Graph{} }

// SetPhase moves on to the next phase. Phases only go forward: the planner
// relies on the image phase's operations all coming first.
func (g *THA3Graph) SetPhase(phase int) {
	if phase < g.phase {
		panic("vk: THA3 phases only go forward")
	}
	g.phase = phase
}

// Weights appends a tensor of weights and returns where it will be.
func (g *THA3Graph) Weights(data []float32) THA3Weights {
	w := THA3Weights{off: len(g.weights), n: len(data)}
	g.weights = append(g.weights, data...)
	return w
}

// Input is a tensor the host writes with THA3Runner.Write. It is pinned.
func (g *THA3Graph) Input(c, h, w int) THA3Tensor {
	t := g.tensor(c, h, w)
	g.tensors[t.id].input = true
	return t
}

// Stamp marks the end of a span the runner's timeline reports under label.
// Only stamps in the pose phase are recorded.
func (g *THA3Graph) Stamp(label string) {
	g.ops = append(g.ops, tha3Op{phase: g.phase, stamp: label})
	g.stamps++
}

// Entry marks where a pose pass may start, for THA3Runner.RunPoseFrom, and
// returns its number. Entry 0 is the start of the pose phase.
func (g *THA3Graph) Entry() int {
	if g.phase != THA3PosePhase {
		panic("vk: THA3 entry outside the pose phase")
	}
	g.entries = append(g.entries, len(g.ops))
	return len(g.entries)
}

func (g *THA3Graph) tensor(c, h, w int) THA3Tensor {
	g.tensors = append(g.tensors, tha3Tensor{floats: c * h * w, phase: g.phase})
	return THA3Tensor{id: len(g.tensors) - 1, C: c, H: h, W: w}
}

// op appends one operation: the tensors it reads, the tensors it creates, and
// what it records once every tensor has its place.
func (g *THA3Graph) op(reads, creates []THA3Tensor, record func(*Recorder, *THA3Runner)) {
	o := tha3Op{phase: g.phase, record: record}
	for _, t := range reads {
		o.uses = append(o.uses, t.id)
	}
	for _, t := range creates {
		o.uses = append(o.uses, t.id)
		o.creates = append(o.creates, t.id)
	}
	g.ops = append(g.ops, o)
}

// offset is where t's first float is in the arena, once plan has run.
func (g *THA3Graph) offset(t THA3Tensor) int {
	return g.tensors[t.id].off + t.ch0*t.H*t.W
}

// tha3Align keeps every tensor on a 256-byte boundary, which is what a
// workgroup reading a row wants to start on.
const tha3Align = 64

func alignFloats(n int) int { return (max(n, 1) + tha3Align - 1) / tha3Align * tha3Align }

// liveness returns, for every tensor, the first and last operation that uses
// it and whether it is pinned. An input's first is -1, a tensor nothing uses
// has -2 for both, and a pinned tensor's last is one past the last operation.
func (g *THA3Graph) liveness() (first, last []int, pinned []bool) {
	n := len(g.tensors)
	first, last, pinned = make([]int, n), make([]int, n), make([]bool, n)
	for i := range first {
		first[i], last[i] = -2, -2
	}
	for i, o := range g.ops {
		for _, id := range o.uses {
			if first[id] == -2 {
				first[id] = i
			}
			last[id] = i
		}
	}
	for id, t := range g.tensors {
		switch {
		case t.input:
			pinned[id] = true
			first[id] = -1
		case first[id] >= 0 && t.phase == THA3ImagePhase && g.ops[last[id]].phase == THA3PosePhase:
			pinned[id] = true
		case first[id] >= 0:
			for _, e := range g.entries {
				if first[id] < e && last[id] >= e {
					pinned[id] = true
				}
			}
		}
		if pinned[id] {
			last[id] = len(g.ops)
		}
	}
	return first, last, pinned
}

type tha3Span struct{ off, n int }

// plan places every tensor and returns the arena's size in floats. Placement
// is first fit in a free list kept sorted and merged; a tensor's floats are
// freed after the operation that uses it last, and never for a pinned one.
func (g *THA3Graph) plan() int {
	_, last, pinned := g.liveness()
	var free []tha3Span
	top := 0
	alloc := func(n int) int {
		n = alignFloats(n)
		for i, s := range free {
			if s.n < n {
				continue
			}
			if s.n == n {
				free = append(free[:i], free[i+1:]...)
			} else {
				free[i] = tha3Span{s.off + n, s.n - n}
			}
			return s.off
		}
		off := top
		top += n
		return off
	}
	release := func(off, n int) {
		free = append(free, tha3Span{off, alignFloats(n)})
		sort.Slice(free, func(a, b int) bool { return free[a].off < free[b].off })
		merged := free[:1]
		for _, s := range free[1:] {
			if l := &merged[len(merged)-1]; l.off+l.n == s.off {
				l.n += s.n
			} else {
				merged = append(merged, s)
			}
		}
		free = merged
	}
	for id, t := range g.tensors {
		if t.input {
			g.tensors[id].off = alloc(t.floats)
		}
	}
	for i, o := range g.ops {
		for _, id := range o.creates {
			g.tensors[id].off = alloc(g.tensors[id].floats)
		}
		released := map[int]bool{}
		for _, id := range o.uses {
			if released[id] || pinned[id] || last[id] != i {
				continue
			}
			released[id] = true
			release(g.tensors[id].off, g.tensors[id].floats)
		}
	}
	return top
}
