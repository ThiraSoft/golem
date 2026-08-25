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
	"unsafe"

	"github.com/ThiraSoft/golem/nn"
	"github.com/ThiraSoft/golem/vk"
)

// UseVulkanHead uploads the logit head to a Vulkan device and makes Logits
// read it there. It fails, and changes nothing, when there is no device, when
// the head is not Q6_K, or when the tensor does not fit in device memory.
//
// A model that has it keeps the CPU path for everything else, including
// LogitsBatch: the shader scores one activation at a time, and a batch reads
// the head once for all of its columns already.
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
	m.head = h
	return nil
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
	if len(m.caches) > 1 {
		return fmt.Errorf("gemma: the device holds one cache, and this model was opened with %d slots", len(m.caches))
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

	var maxHeads, maxKV int
	for _, bc := range cfg.Blocks {
		maxHeads = max(maxHeads, bc.Heads*bc.HeadDim)
		maxKV = max(maxKV, bc.KVHeads*bc.HeadDim)
	}
	attn, err := vk.NewAttention(d, cfg.Dim, maxHeads, maxKV, cfg.MaxContext, len(m.rotations))
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

	// The router's logits are bound into every block's sets, so their buffer
	// has to exist before the first block is added.
	if err := stack.Experts(cfg.Experts); err != nil {
		stack.Close()
		return err
	}

	for i := range cfg.Blocks {
		bc, bw := cfg.Blocks[i], &m.W.Blocks[i]
		quants := []nn.Quant{bw.Q.Quant, bw.O.Quant, bw.Gate.Quant, bw.Up.Quant, bw.Down.Quant}
		if bc.MoE {
			quants = append(quants, bw.GateUpExps.Quant, bw.DownExps.Quant)
		}
		for _, q := range quants {
			if q != nn.Q4_0 {
				stack.Close()
				return fmt.Errorf("gemma: the kernels read Q4_0, block %d has a %s", i, q)
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
		capacity := cfg.MaxContext
		if bc.Window && bc.WindowSize < capacity {
			capacity = bc.WindowSize
		}
		shape := vk.BlockShape{
			Heads: bc.Heads, KVHeads: bc.KVHeads, HeadDim: bc.HeadDim,
			RoPEDims: bc.RoPEDims, Capacity: capacity, Rotation: index[bc.RoPEBase],
			ValueIsKey: bc.ValueIsKey, OwnsKV: bc.OwnsKV, KVSource: bc.KVSource, NormValue: true,
			Eps: cfg.Eps, Scale: 1, // Gemma 4's query norm holds the scores in range
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
		if err := mix.AddBlock(gateUpExps, downExps, bw.Gate.Data, bw.Up.Data, bw.Down.Data); err != nil {
			stack.Close()
			return err
		}
		if err := stack.AddBlock(norms); err != nil {
			stack.Close()
			return err
		}
	}
	if err := stack.Ready(); err != nil {
		stack.Close()
		return err
	}
	m.stack = stack
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
