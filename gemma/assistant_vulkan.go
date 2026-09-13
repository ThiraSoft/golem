package gemma

// The assistant on the card, beside its model's blocks.
//
// With the model's blocks on a card its caches are there too, and the
// assistant has to follow them: it is nothing but four blocks attending over
// those caches. So it gets a stack of its own, as wide as it is, whose
// attention borrows the model's two cache buffers instead of owning any
// (vk.BlockShape.Cache), and the head goes over with it — 262144 rows of
// Q6_K read in full for every draft, which is the largest part of what one
// costs.
//
// The pre-projection stays on this side. It is 3.9 megabytes of Q4_0, and its
// input is the model's state, which is already here: the model hands every
// pass's last state back.

import (
	"fmt"

	"github.com/ThiraSoft/golem/nn"
	"github.com/ThiraSoft/golem/vk"
)

// useVulkan builds the assistant's stack and head on the model's device. It is
// called when both the assistant and the model's stack exist, whichever came
// second.
func (a *Assistant) useVulkan() error {
	m := a.target
	if a.stack != nil {
		return nil
	}
	if m.stack == nil {
		return fmt.Errorf("gemma: the assistant follows the model's blocks, which are not on a card")
	}
	d, err := m.device()
	if err != nil {
		return err
	}
	cfg := a.Cfg

	var maxHeads, maxKV, maxQueryHeads int
	for _, bc := range cfg.Blocks {
		maxHeads = max(maxHeads, bc.Heads*bc.HeadDim)
		maxQueryHeads = max(maxQueryHeads, bc.Heads)
		maxKV = max(maxKV, bc.KVHeads*bc.HeadDim)
		if bc.FFN != cfg.Blocks[0].FFN {
			return fmt.Errorf("gemma: the feed-forward width is one buffer on the card, and assistant block %d is %d wide against block 0's %d",
				bc.Index, bc.FFN, cfg.Blocks[0].FFN)
		}
	}
	// Two geometries, the model's own two: window blocks rotate by base 10⁴
	// with no factors, global blocks by 10⁶ with the assistant's.
	rotation := map[bool]int{true: 0, false: 1}
	attn, err := vk.NewAttention(d, cfg.Dim, maxHeads, maxKV, maxQueryHeads, m.SlotContext(), m.Slots(), len(rotation))
	if err != nil {
		return err
	}
	mix, err := vk.NewMixture(d, cfg.Dim, 0, cfg.Blocks[0].FFN, 0, 0, vk.GELU)
	if err != nil {
		attn.Close()
		return err
	}
	stack, err := vk.NewStack(d, cfg.Dim, cfg.Eps, attn, mix)
	if err != nil {
		attn.Close()
		mix.Close()
		return err
	}
	fail := func(err error) error {
		stack.Close()
		return err
	}
	set := map[bool]bool{}
	for _, bc := range cfg.Blocks {
		if set[bc.Window] {
			continue
		}
		set[bc.Window] = true
		freqs := a.W.RoPEFreqs
		if bc.Window {
			freqs = nil
		}
		if err := stack.SetGeometry(rotation[bc.Window], bc.RoPEDims, bc.RoPEBase, freqs); err != nil {
			return fail(err)
		}
	}
	if err := stack.Experts(0); err != nil {
		return fail(err)
	}

	for i, bc := range cfg.Blocks {
		bw := &a.W.Blocks[i]
		for _, q := range []nn.Quant{bw.Q.Quant, bw.O.Quant, bw.Gate.Quant, bw.Up.Quant, bw.Down.Quant} {
			if !vk.QuantReadable(q) {
				return fail(fmt.Errorf("gemma: assistant block %d holds a %s matrix, which no kernel here reads", i, q))
			}
		}
		k, v, capacity, err := m.stack.Cache(bc.KVSource)
		if err != nil {
			return fail(err)
		}
		shape := vk.BlockShape{
			Heads: bc.Heads, KVHeads: bc.KVHeads, HeadDim: bc.HeadDim,
			RoPEDims: bc.RoPEDims, Capacity: capacity, Rotation: rotation[bc.Window],
			Cache: [2]*vk.Buffer{k, v}, NormValue: true,
			Eps: cfg.Eps, Scale: 1,
			Formats: vk.BlockFormats{Q: bw.Q.Quant, O: bw.O.Quant},
		}
		// No key norm, since no key is computed; the binding still wants a
		// buffer, and the kernel reads it only for a key head.
		if err := attn.AddBlock(shape, bw.Q.Data, nil, nil, bw.O.Data, bw.QNorm, make([]float32, bc.HeadDim)); err != nil {
			return fail(err)
		}
		if err := mix.AddBlock(vk.MixtureFormats{GateUp: bw.Gate.Quant, Down: bw.Down.Quant},
			nil, nil, bw.Gate.Data, bw.Up.Data, bw.Down.Data); err != nil {
			return fail(err)
		}
		if err := stack.AddBlock(vk.BlockNorms{
			Attn: bw.AttnNorm, PostAttn: bw.PostAttnNorm, FFN: bw.FFNNorm,
			PostFFW: bw.PostFFWNorm, OutScale: bw.OutScale, Layout: vk.LayoutDense,
		}); err != nil {
			return fail(err)
		}
	}
	if err := stack.SetOutputNorm(a.W.OutputNorm); err != nil {
		return fail(err)
	}
	if err := stack.Ready(); err != nil {
		return fail(err)
	}

	e := a.W.TokenEmbd
	var head vk.Head
	if e.Quant == nn.Q6_K {
		head, err = vk.NewQ6KHead(d, e.Data, e.Rows, e.Cols)
	} else {
		head, err = vk.NewQuantHead(d, e.Quant, e.Data, e.Rows, e.Cols)
	}
	if err != nil {
		return fail(err)
	}
	a.stack, a.vhead = stack, head
	return nil
}

// draftOnCard carries the projection in a.xs through the blocks on the card and
// scores it there.
func (a *Assistant) draftOnCard(pos int, out []float32) {
	m := a.target
	dim := a.Cfg.Dim
	stream := a.stack.Stream().Floats()
	copy(stream[:dim], a.xs[0])

	at := vk.Position{Slot: m.slotOf(m.cache), Pos: pos}
	for _, bc := range a.Cfg.Blocks {
		first, last := m.cache.Visible(bc, pos, pos)
		if bc.Behind {
			last--
		}
		at.First = append(at.First, first)
		at.Last = append(at.Last, last)
	}
	if err := a.stack.Run([]vk.Position{at}, 0, 0); err != nil {
		panic(fmt.Sprintf("gemma: the device failed the assistant: %v", err))
	}
	copy(a.hidden, stream[:dim])
	copy(a.head.F[0], a.hidden)
	a.vhead.Prepare(a.head)
	if err := a.vhead.MatVec(a.head, 0, out); err != nil {
		panic(fmt.Sprintf("gemma: the assistant's head failed: %v", err))
	}
}

func (a *Assistant) closeVulkan() {
	if a.vhead != nil {
		a.vhead.Close()
		a.vhead = nil
	}
	if a.stack != nil {
		a.stack.Close()
		a.stack = nil
	}
}
