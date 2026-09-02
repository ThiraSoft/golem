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
	if m.head != nil || m.golemHead != nil || m.q6kHead != nil {
		return nil
	}
	if gq := m.W.TokenEmbd.Quant; gq.Golem() {
		if m.W.PreHead == nil {
			return fmt.Errorf("qwen: a %s head without output.pre — the checkpoint was written with the table left plain", m.W.TokenEmbd.Quant)
		}
		d, err := m.device()
		if err != nil {
			return err
		}
		k, err := vk.NewGolemKernels(d, gq)
		if err != nil {
			return err
		}
		h, err := vk.NewGolemHead(k, m.W.TokenEmbd.Data, m.W.TokenEmbd.Rows, m.W.TokenEmbd.Cols, m.W.PreHead)
		if err != nil {
			k.Close()
			return err
		}
		m.golemHead = h
		if m.stack != nil {
			if err := m.useVulkanEmbedding(); err != nil {
				return err
			}
		}
		return nil
	}
	if q := m.W.TokenEmbd.Quant; q != nn.Q4_0 && q != nn.Q6_K {
		return fmt.Errorf("qwen: the Vulkan head reads a Q4_0 or Q6_K embedding, this one is %s", m.W.TokenEmbd.Quant)
	}
	d, err := m.device()
	if err != nil {
		return err
	}
	// Six bits is what a K-quant mix leaves the embedding at, and it is the
	// same kernel Gemma's head has taken since it existed — vk/q6k.go, against
	// a Q8_K activation, which qwen/model.go's Logits already builds for the
	// processor's own K-quant path.
	if m.W.TokenEmbd.Quant == nn.Q6_K {
		h, err := vk.NewQ6KHead(d, m.W.TokenEmbd.Data, m.W.TokenEmbd.Rows, m.W.TokenEmbd.Cols)
		if err != nil {
			return err
		}
		m.q6kHead = h
		if m.stack != nil {
			if err := m.useVulkanEmbedding(); err != nil {
				return err
			}
		}
		return nil
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
	if m.golemHead != nil {
		table, steps, cols := m.golemHead.Table()
		return m.stack.SetEmbeddingGolem(table, steps, cols, m.W.PreHead, m.golemHead.Quant())
	}
	if m.q6kHead != nil {
		table, cols := m.q6kHead.Table()
		return m.stack.SetEmbedding(table, cols, 1.0)
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

// UseVulkan puts the whole model on a Vulkan device: the blocks and the logit
// head. It is the name qwen35 gives the same thing, and it exists here for the
// same reason vqdiff gives for measuring on a card at all — a quality number
// nobody runs is a quality number nobody has. A perplexity over four thousand
// positions of this model is minutes of eight cores and seconds of a card, and
// the two agree: the card and the processor differ by 9e-4 of the hidden state
// and 3e-7 of a logit, which is float32 addition order and nothing else.
//
// The head goes on first, because the stack asks it for the embedding table.
func (m *Model) UseVulkan() error {
	if err := m.UseVulkanHead(); err != nil {
		return err
	}
	return m.UseVulkanStack()
}

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
	var golem *vk.GolemKernels
	if gq := m.W.Blocks[0].Q.Quant; gq.Golem() {
		if golem, err = vk.NewGolemKernels(d, gq); err != nil {
			stack.Close()
			return err
		}
	}

	for i := range cfg.Blocks {
		bc, bw := cfg.Blocks[i], &m.W.Blocks[i]
		// A .golem checkpoint is one format for the whole model by construction;
		// anything else is asked per matrix. A K-quant mix gives the same role
		// different formats in different blocks — a Q4_K_M puts Q6_K on half its
		// ffn_down and a quarter of its attention — so a model-wide answer here
		// is wrong before it starts.
		if golem != nil {
			for _, q := range []nn.Quant{bw.Q.Quant, bw.K.Quant, bw.V.Quant, bw.O.Quant, bw.Gate.Quant, bw.Up.Quant, bw.Down.Quant} {
				if q != bw.Q.Quant {
					stack.Close()
					return fmt.Errorf("qwen: a %s checkpoint is one format throughout, and block %d has a %s", bw.Q.Quant, i, q)
				}
			}
		} else {
			for _, spec := range []struct {
				what string
				q    nn.Quant
			}{
				{"attn_q", bw.Q.Quant}, {"attn_k", bw.K.Quant}, {"attn_v", bw.V.Quant},
				{"attn_output", bw.O.Quant}, {"ffn_gate", bw.Gate.Quant},
				{"ffn_up", bw.Up.Quant}, {"ffn_down", bw.Down.Quant},
			} {
				if !vk.QuantReadable(spec.q) {
					stack.Close()
					return fmt.Errorf("qwen: block %d's %s is %s, which no kernel here reads", i, spec.what, spec.q)
				}
			}
			if bw.Gate.Quant != bw.Up.Quant {
				stack.Close()
				return fmt.Errorf("qwen: block %d has ffn_gate in %s and ffn_up in %s, and one kernel reads them joined",
					i, bw.Gate.Quant, bw.Up.Quant)
			}
		}
		shape := vk.BlockShape{
			Heads: bc.Heads, KVHeads: bc.KVHeads, HeadDim: bc.HeadDim,
			RoPEDims: bc.RoPEDims, Capacity: m.SlotContext(), Rotation: index[bc.RoPEBase],
			OwnsKV: true, Eps: cfg.Eps,
			Formats: vk.BlockFormats{Q: bw.Q.Quant, K: bw.K.Quant, V: bw.V.Quant, O: bw.O.Quant},
			// llama.cpp passes this into the softmax rather than scaling the
			// query; qwen/attention.go is the other copy.
			Scale: float32(1 / sqrtOf(bc.HeadDim)),
		}
		if golem != nil {
			if err := attn.AddBlockGolem(golem, shape, bw.Q.Data, bw.K.Data, bw.V.Data, bw.O.Data,
				bw.QNorm, bw.KNorm, bw.PreQKV, bw.PreO); err != nil {
				stack.Close()
				return err
			}
			if err := mix.AddBlockGolem(golem, bw.Gate.Data, bw.Up.Data, bw.Down.Data,
				bw.PreGateUp, bw.PreDown); err != nil {
				stack.Close()
				return err
			}
		} else {
			if err := attn.AddBlock(shape, bw.Q.Data, bw.K.Data, bw.V.Data, bw.O.Data, bw.QNorm, bw.KNorm); err != nil {
				stack.Close()
				return err
			}
			formats := vk.MixtureFormats{GateUp: bw.Gate.Quant, Down: bw.Down.Quant}
			if err := mix.AddBlock(formats, nil, nil, bw.Gate.Data, bw.Up.Data, bw.Down.Data); err != nil {
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
func (m *Model) VulkanHead() bool { return m.head != nil || m.golemHead != nil || m.q6kHead != nil }

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
	if m.q6kHead != nil {
		m.q6kHead.Close()
		m.q6kHead = nil
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
