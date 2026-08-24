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
	e, err := vk.NewExperts(d, cfg.Dim, cfg.ExpertFFN, cfg.Experts, cfg.ExpertsUsed)
	if err != nil {
		return err
	}
	for i := range cfg.Blocks {
		if !cfg.Blocks[i].MoE {
			continue
		}
		bw := &m.W.Blocks[i]
		if bw.GateUpExps.Quant != nn.Q4_0 || bw.DownExps.Quant != nn.Q4_0 {
			e.Close()
			return fmt.Errorf("gemma: the expert kernels read Q4_0, block %d is %s", i, bw.GateUpExps.Quant)
		}
		if err := e.AddBlock(bw.GateUpExps.Data, bw.DownExps.Data); err != nil {
			e.Close()
			return err
		}
		bw.Experts, bw.ExpertIndex = e, e.Blocks()-1
	}
	m.experts = e
	return nil
}

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
	if m.experts != nil {
		m.experts.Close()
		m.experts = nil
		for i := range m.W.Blocks {
			m.W.Blocks[i].Experts = nil
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
