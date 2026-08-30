package qwen35

import (
	"fmt"

	"github.com/ThiraSoft/golem/nn"
	"github.com/ThiraSoft/golem/vk"
)

// The tower on the card.
//
// What crosses is everything after the patch embedding: twenty-seven blocks
// and the merger. The convolutions, the reordering and the antialiased resize
// of the learned position table stay on the processor — they run once against
// the twenty-seven times a block does, and they are the half of the tower that
// is geometry rather than arithmetic.

// UseVulkan uploads the tower to a device.
//
// Every matrix of the projector is fp16 in the file and the kernels read it
// that way, so a projector stored otherwise is refused here rather than
// answered wrongly.
func (v *VisionTower) UseVulkan(d *vk.Device) error {
	if v.gpu != nil {
		return nil
	}
	cfg, w := v.Cfg, v.W

	// Every matrix of the tower, so that a projector stored otherwise is named
	// at the point it is opened rather than answered wrongly twenty-seven
	// blocks later.
	var wrong error
	fp16 := func(name string, m nn.Matrix) {
		if wrong == nil && m.Quant != nn.F16 {
			wrong = fmt.Errorf("qwen35: %s is %s, and the card's tower reads F16", name, m.Quant)
		}
	}
	fp16("mm.0", w.MM0.W)
	fp16("mm.2", w.MM2.W)
	for i := range w.Blocks {
		b, at := &w.Blocks[i], fmt.Sprintf("v.blk.%d.", i)
		fp16(at+"attn_qkv", b.QKV.W)
		fp16(at+"attn_out", b.O.W)
		fp16(at+"ffn_up", b.Up.W)
		fp16(at+"ffn_down", b.Dn.W)
	}
	if wrong != nil {
		return wrong
	}

	pipe, err := vk.NewVisionPipeline(d, vk.VisionShape{
		Blocks:  cfg.Blocks,
		Dim:     cfg.Dim,
		Heads:   cfg.Heads,
		HeadDim: cfg.HeadDim,
		FFN:     cfg.FFN,
		Merge:   cfg.Merge,
		ProjDim: cfg.ProjDim,
		Eps:     cfg.Eps,
		// The base every Qwen-VL projector turns by. clip.cpp writes it into
		// the graph rather than into the file, so it is a constant here too —
		// named, so that a projector that disagrees is a line to change and
		// not a number to find.
		RoPEBase: 10000,
	})
	if err != nil {
		return err
	}
	for i := range w.Blocks {
		b := &w.Blocks[i]
		if err := pipe.AddBlock(vk.VisionBlockData{
			LN1Gain: b.LN1.Gain, LN1Bias: b.LN1.Bias,
			QKV: linearData(b.QKV), O: linearData(b.O),
			LN2Gain: b.LN2.Gain, LN2Bias: b.LN2.Bias,
			Up: linearData(b.Up), Dn: linearData(b.Dn),
		}); err != nil {
			pipe.Close()
			return fmt.Errorf("qwen35: the tower's block %d: %w", i, err)
		}
	}
	if err := pipe.SetTail(vk.VisionTailData{
		PostGain: w.PostLN.Gain, PostBias: w.PostLN.Bias,
		MM0: linearData(w.MM0), MM2: linearData(w.MM2),
	}); err != nil {
		pipe.Close()
		return fmt.Errorf("qwen35: the tower's merger: %w", err)
	}
	if err := pipe.Prepare(); err != nil {
		pipe.Close()
		return fmt.Errorf("qwen35: the tower cannot go to the card: %w", err)
	}
	v.gpu = pipe
	return nil
}

func linearData(l VisionLinear) vk.VisionLinearData {
	return vk.VisionLinearData{W: l.W.Data, Bias: l.Bias}
}

// Vulkan reports whether the tower runs on a device.
func (v *VisionTower) Vulkan() bool { return v.gpu != nil }

// VulkanResident reports whether the whole tower is resident on the card, as
// against one group of weights at a time across the bus. It is what a card
// already holding a 27B at Q4_0 answers false to, and the tower runs either
// way.
func (v *VisionTower) VulkanResident() bool { return v.gpu != nil && v.gpu.Resident() }

// GPU is the device pipeline, for a test that wants its waypoints.
func (v *VisionTower) GPU() *vk.VisionPipeline { return v.gpu }

// encodeVulkan is Encode's device path: the same patch grid and the same
// positions, and everything from the first block on run on the card.
func (v *VisionTower) encodeVulkan(xs []float32, at [][2]int) ([][]float32, error) {
	cfg := v.Cfg
	pos := make([]uint32, 2*len(at))
	for i, a := range at {
		pos[2*i], pos[2*i+1] = uint32(a[0]), uint32(a[1])
	}
	tokens := len(at) / (cfg.Merge * cfg.Merge)
	flat := make([]float32, tokens*cfg.ProjDim)
	if err := v.gpu.Encode(xs, pos, flat); err != nil {
		return nil, err
	}
	out := make([][]float32, tokens)
	for t := range out {
		out[t] = flat[t*cfg.ProjDim : (t+1)*cfg.ProjDim]
	}
	return out, nil
}

// UseVisionVulkan puts the projector's tower on the same device the model's
// blocks are on. It is a no-op when no projector has been opened, because a
// model without one has no tower to move.
func (m *Model) UseVisionVulkan() error {
	if m.vision == nil {
		return nil
	}
	d, err := m.device()
	if err != nil {
		return err
	}
	return m.vision.UseVulkan(d)
}

// VisionVulkan says whether the tower is on a device and whether the whole of
// it is resident there.
func (m *Model) VisionVulkan() (on, resident bool) {
	if m.vision == nil {
		return false, false
	}
	return m.vision.Vulkan(), m.vision.VulkanResident()
}
