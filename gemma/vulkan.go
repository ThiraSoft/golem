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

// UseVulkanExperts uploads every mixture block's expert stacks to a Vulkan
// device and makes the expert branch run there.
//
// This is the large one. The stacks are 11.96 gibibytes on the 26B A4B, all of
// them resident, which is why a card with sixteen is the smallest that can
// take them — and why it fails rather than falls back when they do not fit.
// What it buys is the other end of the same fact: the experts are 0.8 of the
// 1.7 gigabytes a token reads, and on the CPU that is all bandwidth.
//
// A model with no mixture blocks is not an error; it simply has nothing to
// upload, and VulkanExperts then answers false.
func (m *Model) UseVulkanExperts() error {
	if m.experts != nil {
		return nil
	}
	cfg := m.Cfg
	if cfg.Experts == 0 {
		return nil
	}
	d, err := m.device()
	if err != nil {
		return err
	}
	e, err := vk.NewMixture(d, cfg.Dim, cfg.ExpertFFN, cfg.Blocks[0].FFN, cfg.Experts, cfg.ExpertsUsed)
	if err != nil {
		return err
	}
	for i := range cfg.Blocks {
		if !cfg.Blocks[i].MoE {
			continue
		}
		bw := &m.W.Blocks[i]
		for _, q := range []nn.Quant{bw.GateUpExps.Quant, bw.DownExps.Quant, bw.Gate.Quant, bw.Up.Quant, bw.Down.Quant} {
			if q != nn.Q4_0 {
				e.Close()
				return fmt.Errorf("gemma: the feed-forward kernels read Q4_0, block %d has a %s", i, q)
			}
		}
		if cfg.Blocks[i].FFN != cfg.Blocks[0].FFN {
			e.Close()
			return fmt.Errorf("gemma: block %d has a shared branch of %d where block 0 has %d", i, cfg.Blocks[i].FFN, cfg.Blocks[0].FFN)
		}
		if err := e.AddBlock(bw.GateUpExps.Data, bw.DownExps.Data, bw.Gate.Data, bw.Up.Data, bw.Down.Data); err != nil {
			e.Close()
			return err
		}
		bw.Mixture, bw.MixtureIndex = e, e.Blocks()-1
	}
	m.experts = e
	return nil
}

// UseVulkanAttention uploads every block's four attention matrices and makes
// the four products run there.
//
// Only the products move. The norms, the rotation, the cache and the scores
// stay here, which is where every particular of this model's attention lives
// and where almost none of its bytes are: a block's matrices are nineteen
// megabytes and its scores are a few kilobytes.
func (m *Model) UseVulkanAttention() error {
	if m.attn != nil {
		return nil
	}
	cfg := m.Cfg
	// Two geometries alternate through this model and their heads are not the
	// same size, so the shared buffers are cut for the widest.
	var maxHeads, maxKV int
	for _, bc := range cfg.Blocks {
		maxHeads = max(maxHeads, bc.Heads*bc.HeadDim)
		maxKV = max(maxKV, bc.KVHeads*bc.HeadDim)
	}
	d, err := m.device()
	if err != nil {
		return err
	}
	a, err := vk.NewAttention(d, cfg.Dim, maxHeads, maxKV, cfg.MaxContext)
	if err != nil {
		return err
	}
	for i := range cfg.Blocks {
		bc, bw := cfg.Blocks[i], &m.W.Blocks[i]
		// A block answers for itself which matrices it has: fifteen at the end
		// of this model have neither keys nor values, and some take the value
		// from the key rather than from a matrix.
		var k, v []byte
		if bc.OwnsKV {
			k = bw.K.Data
			if !bc.ValueIsKey {
				v = bw.V.Data
			}
		}
		for _, q := range []nn.Quant{bw.Q.Quant, bw.O.Quant} {
			if q != nn.Q4_0 {
				a.Close()
				return fmt.Errorf("gemma: the attention kernel reads Q4_0, block %d has a %s", i, q)
			}
		}
		capacity := cfg.MaxContext
		if bc.Window && bc.WindowSize < capacity {
			capacity = bc.WindowSize
		}
		shape := vk.BlockShape{
			Heads: bc.Heads, KVHeads: bc.KVHeads, HeadDim: bc.HeadDim,
			RoPEDims: bc.RoPEDims, Capacity: capacity,
			ValueIsKey: bc.ValueIsKey, OwnsKV: bc.OwnsKV, KVSource: bc.KVSource,
			Eps: cfg.Eps,
		}
		if err := a.AddBlock(shape, bw.Q.Data, k, v, bw.O.Data, bw.QNorm, bw.KNorm); err != nil {
			a.Close()
			return err
		}
		bw.Attn, bw.AttnIndex = a, a.Blocks()-1
	}
	m.attn = a
	return nil
}

// VulkanCacheBytes is what the keys and values take on the card, which is the
// part of this that grows with the context rather than with the model.
func (m *Model) VulkanCacheBytes() int {
	if m.attn == nil {
		return 0
	}
	return m.attn.Bytes()
}

// VulkanAttention says whether the attention matrices are on a device.
func (m *Model) VulkanAttention() bool { return m.attn != nil }

// VulkanExperts says whether the expert stacks are on a device.
func (m *Model) VulkanExperts() bool { return m.experts != nil }

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
	if m.attn != nil {
		m.attn.Close()
		m.attn = nil
		for i := range m.W.Blocks {
			m.W.Blocks[i].Attn = nil
		}
	}
	if m.experts != nil {
		m.experts.Close()
		m.experts = nil
		for i := range m.W.Blocks {
			m.W.Blocks[i].Mixture = nil
		}
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
