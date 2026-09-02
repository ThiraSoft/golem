package gemma

// The logit head on a Vulkan device, when the caller asks for it.
//
// The head is the input embedding read the other way round, so it is the
// largest tensor in the file and it is read in full for every token drawn:
// 577 mebibytes of Q6_K on the 26B, a quarter of what a token costs, and a
// quarter that no amount of CPU work shortens because the bytes are the cost.
// A card reads those bytes about five times faster on the round trip a token
// makes, and fifteen times faster once it is awake, which vk/q6k_test.go
// measures both ways.
//
// It is opt-in and it is one tensor. Nothing else moves: the thirty-five
// blocks stay where they are, and the activation that crosses is eleven
// kilobytes. That is the whole reason the split is affordable — see vk/q6k.go.

import (
	"fmt"
	"math"
	"unsafe"

	"github.com/ThiraSoft/golem/nn"
	"github.com/ThiraSoft/golem/vk"
)

// UseVulkanHead uploads the logit head to a Vulkan device and makes Logits
// read it there. It fails, and changes nothing, when there is no device, when
// the head is not Q6_K, or when the tensor does not fit in device memory.
//
// LogitsBatch keeps the CPU path whatever else is on the card: the shader
// scores one activation at a time, and a batch reads the head once for all of
// its columns already.
func (m *Model) UseVulkanHead() error {
	if m.head != nil {
		return nil
	}
	if m.W.TokenEmbd.Quant != nn.Q6_K {
		return fmt.Errorf("gemma: the Vulkan head wants a Q6_K embedding, this one is %s", m.W.TokenEmbd.Quant)
	}
	d, err := m.device()
	if err != nil {
		return err
	}
	h, err := vk.NewQ6KHead(d, m.W.TokenEmbd.Data, m.W.TokenEmbd.Rows, m.W.TokenEmbd.Cols)
	if err != nil {
		return err
	}
	// The cap goes with the product rather than after it: the shader writes a
	// logit that is already capped, and Logits does not walk the vocabulary
	// again to do it. gemma/model.go's Logits asks the head whether it did.
	h.Softcap(m.Cfg.LogitSoftcap)
	m.head = h
	// The head is the embedding read the other way round, so a stack built
	// before it can now read its rows too rather than being handed them.
	if m.stack != nil {
		if err := m.useVulkanEmbedding(); err != nil {
			return err
		}
	}
	return nil
}

// useVulkanEmbedding points the stack at the head's table. It is one tensor
// serving both directions: nothing more is uploaded, and what it saves is not
// the microseconds but the last step of a pass that was still this side of the
// bus. vk/shaders/embed_q6k.comp says the rest.
func (m *Model) useVulkanEmbedding() error {
	if m.head == nil || m.stack == nil || m.stack.Embedding() {
		return nil
	}
	if m.Cfg.PLEDim > 0 {
		// The per-layer inputs are built from the embedding on this side, so
		// it has to exist here.
		return nil
	}
	table, cols := m.head.Table()
	return m.stack.SetEmbedding(table, cols, float32(math.Sqrt(float64(m.Cfg.Dim))))
}

// UseVulkanStack puts every block of the model on a Vulkan device: the
// attention with its cache, both branches of the feed forward, the norms
// between them and the routing. A token then crosses the bus twice — the
// embedding in and the last hidden state out — instead of sixty times.
//
// It is the last of four moves and the one the other three were for. A
// submission costs sixty-three microseconds whatever is in it, and a card
// handed a hundred microseconds of work and then left alone runs at half its
// clocks. vk/stack.go has the measurements.
//
// It fails rather than falling back: a model half on a card the caller
// believed it was wholly on is a model whose speed nobody can explain.
func (m *Model) UseVulkanStack() error {
	if m.stack != nil {
		return nil
	}
	cfg := m.Cfg
	if cfg.PLEDim > 0 {
		return fmt.Errorf("gemma: the Vulkan stack has no per-layer embedding branch, and this checkpoint carries one")
	}
	d, err := m.device()
	if err != nil {
		return err
	}

	// One rotation geometry per base. A block is asked which it uses rather
	// than branched on, the way nn/rope.go keys its tables.
	m.rotations = nil
	index := map[float64]int{}
	for i, bc := range cfg.Blocks {
		if _, ok := index[bc.RoPEBase]; ok {
			continue
		}
		freqs := m.W.RoPEFreqs
		if bc.Window {
			freqs = nil // the frequency factors belong to the global blocks
		}
		index[bc.RoPEBase] = len(m.rotations)
		m.rotations = append(m.rotations, rotation{base: bc.RoPEBase, dims: cfg.Blocks[i].RoPEDims, freqs: freqs})
	}

	var maxHeads, maxKV, maxQueryHeads int
	for _, bc := range cfg.Blocks {
		maxHeads = max(maxHeads, bc.Heads*bc.HeadDim)
		maxQueryHeads = max(maxQueryHeads, bc.Heads)
		maxKV = max(maxKV, bc.KVHeads*bc.HeadDim)
	}
	// The caches are cut into slots on the card exactly as they are cut on the
	// processor: one ring a conversation, each holding SlotContext positions,
	// the whole of them coming to the context the caller allowed. Which of
	// them a column belongs to travels in the position buffer, so a pass may
	// carry tokens of several conversations at once — vk/attention.go's span
	// says what that costs the scores.
	attn, err := vk.NewAttention(d, cfg.Dim, maxHeads, maxKV, maxQueryHeads, m.SlotContext(), m.Slots(), len(m.rotations))
	if err != nil {
		return err
	}
	// A dense checkpoint is a mixture with no experts: the shared branch of a
	// mixture block and an ordinary feed forward are the same three matrices
	// under the same norm, which is what gemma/block.go says in prose.
	dense := cfg.Blocks[0].FFN
	for i, bc := range cfg.Blocks {
		if bc.FFN != dense {
			return fmt.Errorf("gemma: the feed-forward width is one buffer on the card, and block %d is %d wide against block 0's %d", i, bc.FFN, dense)
		}
	}
	mix, err := vk.NewMixture(d, cfg.Dim, cfg.ExpertFFN, dense, cfg.Experts, cfg.ExpertsUsed, vk.GELU)
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

	// The inverse frequencies, once. They are all the CPU knows about the
	// rotation now: the angles themselves are made at the head of every pass,
	// out of the position buffer. vk/shaders/rope_table.comp says what that
	// replaced — six percent of a wide prompt, spent in math.Pow.
	for i, r := range m.rotations {
		if err := stack.SetGeometry(i, r.dims, r.base, r.freqs); err != nil {
			stack.Close()
			return err
		}
	}

	// The router's logits are bound into every block's sets, so their buffer
	// has to exist before the first block is added.
	if err := stack.Experts(cfg.Experts); err != nil {
		stack.Close()
		return err
	}

	for i := range cfg.Blocks {
		bc, bw := cfg.Blocks[i], &m.W.Blocks[i]
		// Asked per matrix rather than per model. A Q4_K_M gives the same role
		// different formats in different blocks — half of Gemma 4 12B's
		// ffn_down and its attn_v are Q6_K where the rest is Q4_K — so a check
		// that wanted one format for the whole file refused every K-quant mix
		// there is.
		for _, spec := range []struct {
			what string
			q    nn.Quant
		}{
			{"attn_q", bw.Q.Quant}, {"attn_output", bw.O.Quant},
			{"ffn_gate", bw.Gate.Quant}, {"ffn_up", bw.Up.Quant}, {"ffn_down", bw.Down.Quant},
		} {
			if !vk.QuantReadable(spec.q) {
				stack.Close()
				return fmt.Errorf("gemma: block %d's %s is %s, which no kernel here reads", i, spec.what, spec.q)
			}
		}
		if bw.Gate.Quant != bw.Up.Quant {
			stack.Close()
			return fmt.Errorf("gemma: block %d has ffn_gate in %s and ffn_up in %s, and one kernel reads them joined",
				i, bw.Gate.Quant, bw.Up.Quant)
		}
		if bc.OwnsKV {
			for _, spec := range []struct {
				what string
				q    nn.Quant
			}{{"attn_k", bw.K.Quant}, {"attn_v", bw.V.Quant}} {
				if bc.ValueIsKey && spec.what == "attn_v" {
					continue
				}
				if !vk.QuantReadable(spec.q) {
					stack.Close()
					return fmt.Errorf("gemma: block %d's %s is %s, which no kernel here reads", i, spec.what, spec.q)
				}
			}
		}
		if bc.MoE && bw.Router.Quant != nn.F32 {
			stack.Close()
			return fmt.Errorf("gemma: the router kernel reads float32, block %d has a %s", i, bw.Router.Quant)
		}

		var k, v []byte
		if bc.OwnsKV {
			k = bw.K.Data
			if !bc.ValueIsKey {
				v = bw.V.Data
			}
		}
		capacity := m.SlotContext()
		if bc.Window && bc.WindowSize < capacity {
			capacity = bc.WindowSize
		}
		shape := vk.BlockShape{
			Heads: bc.Heads, KVHeads: bc.KVHeads, HeadDim: bc.HeadDim,
			RoPEDims: bc.RoPEDims, Capacity: capacity, Rotation: index[bc.RoPEBase],
			ValueIsKey: bc.ValueIsKey, OwnsKV: bc.OwnsKV, KVSource: bc.KVSource, NormValue: true,
			Eps: cfg.Eps, Scale: 1, // Gemma 4's query norm holds the scores in range
			Formats: vk.BlockFormats{Q: bw.Q.Quant, K: bw.K.Quant, V: bw.V.Quant, O: bw.O.Quant},
		}
		if err := attn.AddBlock(shape, bw.Q.Data, k, v, bw.O.Data, bw.QNorm, bw.KNorm); err != nil {
			stack.Close()
			return err
		}
		var gateUpExps, downExps []byte
		norms := vk.BlockNorms{
			Attn: bw.AttnNorm, PostAttn: bw.PostAttnNorm, FFN: bw.FFNNorm,
			PostFFW: bw.PostFFWNorm, OutScale: bw.OutScale, Layout: vk.LayoutDense,
		}
		if bc.MoE {
			norms.Layout = vk.LayoutMixture
			gateUpExps, downExps = bw.GateUpExps.Data, bw.DownExps.Data
			norms.PreFFW2, norms.PostFFW1, norms.PostFFW2 = bw.PreFFWNorm2, bw.PostFFWNorm1, bw.PostFFWNorm2
			norms.RouterScale, norms.DownScale, norms.Router = bw.RouterScale, bw.DownScale, routerRows(bw.Router)
		}
		formats := vk.MixtureFormats{GateUp: bw.Gate.Quant, Down: bw.Down.Quant}
		if bc.MoE {
			formats.GateUpExps, formats.DownExps = bw.GateUpExps.Quant, bw.DownExps.Quant
		}
		if err := mix.AddBlock(formats, gateUpExps, downExps, bw.Gate.Data, bw.Up.Data, bw.Down.Data); err != nil {
			stack.Close()
			return err
		}
		if err := stack.AddBlock(norms); err != nil {
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

// routerRows reads a float32 matrix out of the mapping, which is where the
// router alone among this model's matrices is kept.
func routerRows(m nn.Matrix) []float32 {
	return unsafe.Slice((*float32)(unsafe.Pointer(&m.Data[0])), m.Rows*m.Cols)
}

// VulkanStack says whether the blocks are on a device.
func (m *Model) VulkanStack() bool { return m.stack != nil }

// A rotation is one geometry of the model's rotation, tabulated once a token
// rather than once a block.
type rotation struct {
	base  float64
	dims  int
	freqs []float32
}

// device opens the Vulkan device the model shares between its parts, or
// returns the one it already has. The head and the experts sit on the same
// card and must: they are two halves of one token.
func (m *Model) device() (*vk.Device, error) {
	if m.headDev != nil {
		return m.headDev, nil
	}
	d, err := vk.Open()
	if err != nil {
		return nil, err
	}
	m.headDev = d
	return d, nil
}

// VulkanEmbedding says whether the card looks the token embedding up itself,
// which it does when both the stack and the head are on it.
func (m *Model) VulkanEmbedding() bool { return m.stack != nil && m.stack.Embedding() }

// VulkanHead says whether the head is on a device.
func (m *Model) VulkanHead() bool { return m.head != nil }

// closeVulkanHead releases the device. Close calls it; a caller that wants the
// memory back sooner has no reason to.
func (m *Model) closeVulkanHead() {
	if m.stack != nil {
		m.stack.Close()
		m.stack = nil
	}
	if m.head != nil {
		m.head.Close()
		m.head = nil
	}
	if m.headDev != nil {
		m.headDev.Close()
		m.headDev = nil
	}
}
