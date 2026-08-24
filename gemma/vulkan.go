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
	d, err := vk.Open()
	if err != nil {
		return err
	}
	h, err := vk.NewQ6KHead(d, m.W.TokenEmbd.Data, m.W.TokenEmbd.Rows, m.W.TokenEmbd.Cols)
	if err != nil {
		d.Close()
		return err
	}
	m.headDev, m.head = d, h
	return nil
}

// VulkanHead says whether the head is on a device.
func (m *Model) VulkanHead() bool { return m.head != nil }

// closeVulkanHead releases the device. Close calls it; a caller that wants the
// memory back sooner has no reason to.
func (m *Model) closeVulkanHead() {
	if m.head != nil {
		m.head.Close()
		m.head = nil
	}
	if m.headDev != nil {
		m.headDev.Close()
		m.headDev = nil
	}
}
