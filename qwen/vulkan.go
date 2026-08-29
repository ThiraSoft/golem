package qwen

// Qwen3 on a Vulkan device, over the same stack Gemma 4 runs on.
//
// vk/stack.go was written for Gemma 4's mixture and then for its dense block,
// and what those two have in common with this one is nearly everything: the
// four attention projections, a query norm and a key norm per head, a rotation,
// a cache in fp16, the scores and the mix, and a gated feed forward whose
// three matrices are read the same way. What differs is small enough to name.
//
//   - The block is an ordinary pre-norm one. The attention's projection rejoins
//     the stream with no norm between them, and so does the feed forward. Gemma
//     norms on the way out of both halves and scales the whole block by a
//     scalar. vk.LayoutPreNorm is that difference.
//   - The gate is a SiLU rather than ggml's tabulated GELU. vk.SiLU.
//   - The scores are scaled by one over the square root of the head, which
//     Gemma leaves at one because its query norm holds them in range.
//   - The head is Q4_0, not Q6_K, so it is vk.Q40Head and not vk.Q6KHead.
//
// Everything else is the file being read rather than a second path.

import (
	"fmt"

	"github.com/ThiraSoft/golem/nn"
	"github.com/ThiraSoft/golem/vk"
)

// UseVulkanHead uploads the logit head to a Vulkan device and makes Logits
// read it there. It fails, and changes nothing, when there is no device, when
// the head is not Q4_0, or when the tensor does not fit in device memory.
func (m *Model) UseVulkanHead() error {
	if m.head != nil || m.d4gHead != nil {
		return nil
	}
	if bits := m.W.TokenEmbd.Quant.D4Width(); bits > 0 {
		if m.W.PreHead == nil {
			return fmt.Errorf("qwen: a %s head without output.pre — the checkpoint was written with the table left plain", m.W.TokenEmbd.Quant)
		}
		d, err := m.device()
		if err != nil {
			return err
		}
		k, err := vk.NewD4GKernels(d, bits)
		if err != nil {
			return err
		}
		h, err := vk.NewD4GHead(k, m.W.TokenEmbd.Data, m.W.TokenEmbd.Rows, m.W.TokenEmbd.Cols, m.W.PreHead)
		if err != nil {
			k.Close()
			return err
		}
		m.d4gHead = h
		if m.stack != nil {
			if err := m.useVulkanEmbedding(); err != nil {
				return err
			}
		}
		return nil
	}
	if m.W.TokenEmbd.Quant != nn.Q4_0 {
		return fmt.Errorf("qwen: the Vulkan head wants a Q4_0 embedding, this one is %s", m.W.TokenEmbd.Quant)
	}
	d, err := m.device()
	if err != nil {
		return err
	}
	h, err := vk.NewQ40Head(d, m.W.TokenEmbd.Data, m.W.TokenEmbd.Rows, m.W.TokenEmbd.Cols)
	if err != nil {
		return err
	}
	m.head = h
	if m.stack != nil {
		if err := m.useVulkanEmbedding(); err != nil {
			return err
		}
	}
	return nil
}

// useVulkanEmbedding points the stack at whichever head table is on the card,
// so that a token crosses the bus as an identifier rather than as a row.
func (m *Model) useVulkanEmbedding() error {
	if m.stack == nil || m.stack.Embedding() {
		return nil
	}
	if m.d4gHead != nil {
		table, lattice, cols := m.d4gHead.Table()
		return m.stack.SetEmbeddingD4G(table, lattice, cols, m.W.PreHead)
	}
	if m.head == nil {
		return nil
	}
	table, cols := m.head.Table()
	return m.stack.SetEmbeddingQ40(table, cols, 1.0)
}

// VulkanEmbedding says whether the card looks the token embedding up itself,
// which it does when both the stack and the head are on it.
func (m *Model) VulkanEmbedding() bool { return m.stack != nil && m.stack.Embedding() }

// UseVulkanStack puts every block of the model on a Vulkan device: the
// attention with its cache, the feed forward, and the two norms between them.
// A token then crosses the bus twice — the embedding in and the last hidden
// state out — instead of once a block.
//
// It fails rather than falling back: a model half on a card the caller
// believed it was wholly on is a model whose speed nobody can explain.
func (m *Model) UseVulkanStack() error {
	if m.stack != nil {
		return nil
	}
	cfg := m.Cfg
	d, err := m.device()
	if err != nil {
		return err
	}

	// One rotation geometry per base, the way gemma/vulkan.go does it. This
	// checkpoint has one; qwen35 declares a full_attention_interval and will
	// not, so the table is built from the file rather than assumed.
	m.rotations = nil
	index := map[float64]int{}
	for _, bc := range cfg.Blocks {
		if _, ok := index[bc.RoPEBase]; ok {
			continue
		}
		index[bc.RoPEBase] = len(m.rotations)
		m.rotations = append(m.rotations, rotation{base: bc.RoPEBase, dims: bc.RoPEDims})
	}

	var maxHeads, maxKV, maxQueryHeads int
	ffn := cfg.Blocks[0].FFN
	for i, bc := range cfg.Blocks {
		maxHeads = max(maxHeads, bc.Heads*bc.HeadDim)
		maxQueryHeads = max(maxQueryHeads, bc.Heads)
		maxKV = max(maxKV, bc.KVHeads*bc.HeadDim)
		if bc.FFN != ffn {
			return fmt.Errorf("qwen: the feed-forward width is one buffer on the card, and block %d is %d wide against block 0's %d", i, bc.FFN, ffn)
		}
	}
	// One ring a conversation, each holding SlotContext positions, laid end to
	// end — the same cut the processor's caches take. gemma/vulkan.go says the
	// rest.
	attn, err := vk.NewAttention(d, cfg.Dim, maxHeads, maxKV, maxQueryHeads, m.SlotContext(), m.Slots(), len(m.rotations))
	if err != nil {
		return err
	}
	// No experts, and the gate is a SiLU: this is a swiglu, not Gemma's
	// tabulated GELU.
	mix, err := vk.NewMixture(d, cfg.Dim, 0, ffn, 0, 0, vk.SiLU)
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

	// The inverse frequencies, once. The angles are made by the card at the
	// head of every pass; see vk/shaders/rope_table.comp.
	for i, r := range m.rotations {
		if err := stack.SetGeometry(i, r.dims, r.base, nil); err != nil {
			stack.Close()
			return err
		}
	}

	// A .golem checkpoint reads through a different set of kernels and a
	// different activation: floats through a scale and a rotation, rather than
	// Q8_0. The lattice table belongs to the device and is uploaded once for
	// the model, whatever a block does with it.
	var d4g *vk.D4GKernels
	if bits := m.W.Blocks[0].Q.Quant.D4Width(); bits > 0 {
		if d4g, err = vk.NewD4GKernels(d, bits); err != nil {
			stack.Close()
			return err
		}
	}

	for i := range cfg.Blocks {
		bc, bw := cfg.Blocks[i], &m.W.Blocks[i]
		want := nn.Q4_0
		if d4g != nil {
			want = bw.Q.Quant
		}
		for _, q := range []nn.Quant{bw.Q.Quant, bw.K.Quant, bw.V.Quant, bw.O.Quant, bw.Gate.Quant, bw.Up.Quant, bw.Down.Quant} {
			if q != want {
				stack.Close()
				return fmt.Errorf("qwen: the kernels read %s, block %d has a %s", want, i, q)
			}
		}
		shape := vk.BlockShape{
			Heads: bc.Heads, KVHeads: bc.KVHeads, HeadDim: bc.HeadDim,
			RoPEDims: bc.RoPEDims, Capacity: m.SlotContext(), Rotation: index[bc.RoPEBase],
			OwnsKV: true, Eps: cfg.Eps,
			// llama.cpp passes this into the softmax rather than scaling the
			// query; qwen/attention.go is the other copy.
			Scale: float32(1 / sqrtOf(bc.HeadDim)),
		}
		if d4g != nil {
			if err := attn.AddBlockD4G(d4g, shape, bw.Q.Data, bw.K.Data, bw.V.Data, bw.O.Data,
				bw.QNorm, bw.KNorm, bw.PreQKV, bw.PreO); err != nil {
				stack.Close()
				return err
			}
			if err := mix.AddBlockD4G(d4g, bw.Gate.Data, bw.Up.Data, bw.Down.Data,
				bw.PreGateUp, bw.PreDown); err != nil {
				stack.Close()
				return err
			}
		} else {
			if err := attn.AddBlock(shape, bw.Q.Data, bw.K.Data, bw.V.Data, bw.O.Data, bw.QNorm, bw.KNorm); err != nil {
				stack.Close()
				return err
			}
			if err := mix.AddBlock(nil, nil, bw.Gate.Data, bw.Up.Data, bw.Down.Data); err != nil {
				stack.Close()
				return err
			}
		}
		if err := stack.AddBlock(vk.BlockNorms{
			Attn: bw.AttnNorm, FFN: bw.FFNNorm, OutScale: 1, Layout: vk.LayoutPreNorm,
		}); err != nil {
			stack.Close()
			return err
		}
	}
	// The final norm too, so that what comes back off the card is the hidden
	// state a caller can compare against llama.cpp's at the same point rather
	// than one norm short of it.
	if err := stack.SetOutputNorm(m.W.OutputNorm); err != nil {
		stack.Close()
		return err
	}
	if err := stack.Ready(); err != nil {
		stack.Close()
		return err
	}
	m.stack = stack
	if err := m.useVulkanEmbedding(); err != nil {
		return err
	}
	return nil
}

// sqrtOf is math.Sqrt on an int, by Newton's method, so that this file's
// arithmetic is the same shape as vk/stack.go's.
func sqrtOf(n int) float64 {
	x := float64(n)
	if x <= 0 {
		return 1
	}
	guess := x
	for i := 0; i < 40; i++ {
		guess = 0.5 * (guess + x/guess)
	}
	return guess
}

// A rotation is one geometry of the model's rotation, tabulated once a token
// rather than once a block.
type rotation struct {
	base float64
	dims int
}

// VulkanColumns is how many positions one pass of the device carries, or zero
// when there is no device.
func (m *Model) VulkanColumns() int {
	if m.stack == nil {
		return 0
	}
	return m.stack.Columns()
}

// ProfileStack points the device path at a timeline, which stamps the card's
// clock between the stages of every block. It is nil-safe and off by default.
func (m *Model) ProfileStack(t *vk.Timeline) {
	if m.stack != nil {
		m.stack.Profile(t)
	}
}

// NewStackTimeline is a timeline sized for one pass of this model's stack.
func (m *Model) NewStackTimeline() (*vk.Timeline, error) {
	if m.stack == nil {
		return nil, fmt.Errorf("qwen: there is no device stack to profile")
	}
	return m.stack.NewTimeline()
}

// VulkanStack says whether the blocks are on a device.
func (m *Model) VulkanStack() bool { return m.stack != nil }

// VulkanHead says whether the head is on a device.
func (m *Model) VulkanHead() bool { return m.head != nil || m.d4gHead != nil }

// device opens the Vulkan device the model shares between its parts, or
// returns the one it already has. The head and the blocks sit on the same card
// and must: they are two halves of one token.
func (m *Model) device() (*vk.Device, error) {
	if m.dev != nil {
		return m.dev, nil
	}
	d, err := vk.Open()
	if err != nil {
		return nil, err
	}
	m.dev = d
	return d, nil
}

// closeVulkan releases the device. Close calls it.
func (m *Model) closeVulkan() {
	if m.stack != nil {
		m.stack.Close()
		m.stack = nil
	}
	if m.head != nil {
		m.head.Close()
		m.head = nil
	}
	if m.dev != nil {
		m.dev.Close()
		m.dev = nil
	}
}

// runStack carries a stretch of positions through every block on the device,
// in one submission. What crosses is those vectors in and the same vectors
// back (or nil when the card looks up embeddings directly).
func (m *Model) runStack(xs [][]float32, at []Place) {
	cfg := m.Cfg
	if xs != nil {
		stream := m.stack.Stream().Floats()
		for t := range xs {
			copy(stream[t*cfg.Dim:], xs[t])
		}
	}

	positions := make([]vk.Position, len(at))
	for c, one := range at {
		positions[c] = vk.Position{Slot: m.slotOf(one.Cache), Pos: one.Pos}
		for _, bc := range cfg.Blocks {
			first, last := one.Cache.Visible(bc, one.Pos)
			positions[c].First = append(positions[c].First, first)
			positions[c].Last = append(positions[c].Last, last)
		}
	}
	if err := m.stack.Run(positions, 0, 0); err != nil {
		panic(fmt.Sprintf("qwen: the device failed: %v", err))
	}
	if xs != nil {
		stream := m.stack.Stream().Floats()
		for t := range xs {
			copy(xs[t], stream[t*cfg.Dim:])
		}
	}
}

// TraceBlocks asks the device path to keep every block's output, which it does
// not do by default.
func (m *Model) TraceBlocks() {
	if m.stack != nil {
		m.stack.Trace(true)
	}
}
