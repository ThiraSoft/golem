package qwen35

import (
	"fmt"
	"os"
	"strconv"

	"github.com/ThiraSoft/golem/nn"
	"github.com/ThiraSoft/golem/vk"
)

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

// UseVulkanHead uploads the logit head to the Vulkan device.
func (m *Model) UseVulkanHead() error {
	if m.headQ6K != nil || m.head != nil {
		return nil
	}
	d, err := m.device()
	if err != nil {
		return err
	}
	if m.W.OutputHead.Quant == nn.Q6_K {
		h, err := vk.NewQ6KHead(d, m.W.OutputHead.Data, m.W.OutputHead.Rows, m.W.OutputHead.Cols)
		if err != nil {
			return fmt.Errorf("qwen35: cannot upload Q6_K head to Vulkan: %w", err)
		}
		m.headQ6K = h
	} else if m.W.OutputHead.Quant == nn.Q4_0 {
		h, err := vk.NewQ40Head(d, m.W.OutputHead.Data, m.W.OutputHead.Rows, m.W.OutputHead.Cols)
		if err != nil {
			return fmt.Errorf("qwen35: cannot upload Q4_0 head to Vulkan: %w", err)
		}
		m.head = h
	}
	return nil
}

func (m *Model) VulkanHead() bool {
	return m.headQ6K != nil || m.head != nil
}

func (m *Model) VulkanStack() bool {
	return m.gpuPipe != nil
}

func (m *Model) UseVulkanStack() error {
	if m.gpuPipe != nil {
		return nil
	}
	d, err := m.device()
	if err != nil {
		return err
	}

	cfg := m.Cfg
	numBlocks := m.trunk()
	// GOLEM_QWEN35_GPU_BLOCKS caps how many blocks go to the card. A partial
	// upload answers nothing useful, but it is what lets a divergence be
	// reproduced without fourteen gigabytes of it.
	if v := os.Getenv("GOLEM_QWEN35_GPU_BLOCKS"); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 && n < numBlocks {
			numBlocks = n
		}
	}
	if numBlocks == 0 {
		return fmt.Errorf("qwen35: the model has no blocks to upload")
	}

	// Every block shares one geometry; only which mixer a block has differs.
	// The first full-attention block names the attention side of it, and the
	// first delta net the other.
	shape := vk.QwenShape{
		Dim:        cfg.Dim,
		FFN:        cfg.Blocks[0].FFN,
		MaxContext: cfg.MaxContext,
		Eps:        cfg.Eps,
		// vk takes the widths as a plain array: it has no reason to import nn
		// for a type, and this is the one place the two spellings meet.
		RoPESections: [4]int(cfg.RoPESections),
	}
	for _, bc := range cfg.Blocks[:numBlocks] {
		if bc.Type == BlockFullAttn && shape.Heads == 0 {
			shape.Heads, shape.KVHeads = bc.Heads, bc.KVHeads
			shape.HeadDim, shape.RoPEDims = bc.HeadDim, bc.RoPEDims
			shape.RoPEBase = float32(bc.RoPEBase)
		}
		if bc.Type == BlockSSM && shape.Rank == 0 {
			shape.ConvDim = bc.SSMGroupCount*bc.SSMStateSize*2 + bc.SSMInnerSize
			shape.Inner, shape.Rank = bc.SSMInnerSize, bc.SSMTimeStepRank
			shape.StateSize, shape.Groups = bc.SSMStateSize, bc.SSMGroupCount
		}
	}

	pipe, err := vk.NewQwenPipeline(d, shape)
	if err != nil {
		return fmt.Errorf("qwen35: cannot create GPU pipeline: %w", err)
	}

	var attnNorms, ffnNorms [][]float32
	var isSSM []bool

	for i := 0; i < numBlocks; i++ {
		bc := cfg.Blocks[i]
		bw := &m.W.Blocks[i]

		attnNorms = append(attnNorms, bw.AttnNorm)
		ffnNorms = append(ffnNorms, bw.FFNNorm)
		isSSM = append(isSSM, bc.Type != BlockFullAttn)

		if err := pipe.AddFFNBlock(vk.QwenFFNData{
			Gate:     bw.Gate.Data,
			Up:       bw.Up.Data,
			Down:     bw.Down.Data,
			DownQ4_1: bw.Down.Quant == nn.Q4_1,
		}); err != nil {
			pipe.Close()
			return fmt.Errorf("qwen35: block %d feed forward: %w", i, err)
		}

		if bc.Type == BlockFullAttn {
			err = pipe.AddAttnBlock(i, vk.QwenAttnData{
				WQ: bw.Q.Data, WK: bw.K.Data, WV: bw.V.Data, WO: bw.O.Data,
				QNorm: bw.QNorm, KNorm: bw.KNorm,
			})
		} else {
			err = pipe.AddSSMBlock(i, vk.QwenSSMData{
				WQKV:       bw.QKV.Data,
				WGate:      bw.AttnGate.Data,
				WAlpha:     bw.SSMAlpha.Data,
				WBeta:      bw.SSMBeta.Data,
				WOut:       bw.SSMOut.Data,
				OutIsQ5K:   bw.SSMOut.Quant == nn.Q5_K,
				OutIsF32:   bw.SSMOut.Quant == nn.F32,
				ConvWeight: bw.Conv1D,
				SSMA:       bw.SSMA,
				SSMDtBias:  bw.SSMDtBias,
				SSMNorm:    bw.SSMNorm,
			})
		}
		if err != nil {
			pipe.Close()
			return fmt.Errorf("qwen35: block %d mixer: %w", i, err)
		}
	}

	if err := pipe.SetNorms(attnNorms, ffnNorms, m.W.OutputNorm, isSSM); err != nil {
		pipe.Close()
		return fmt.Errorf("qwen35: norms: %w", err)
	}

	if m.HasMTP() && numBlocks == m.trunk() {
		il := m.trunk()
		bw := &m.W.Blocks[il]
		headNorm := m.W.MTP.SharedHeadNorm
		if len(headNorm) == 0 {
			headNorm = m.W.OutputNorm
		}
		if err := pipe.AddMTPBlock(vk.QwenMTPData{
			EHProj: m.W.MTP.EHProj.Data,
			Attn: vk.QwenAttnData{
				WQ: bw.Q.Data, WK: bw.K.Data, WV: bw.V.Data, WO: bw.O.Data,
				QNorm: bw.QNorm, KNorm: bw.KNorm,
			},
			FFN: vk.QwenFFNData{
				Gate: bw.Gate.Data, Up: bw.Up.Data, Down: bw.Down.Data,
				DownQ4_1: bw.Down.Quant == nn.Q4_1,
			},
			AttnNorm: bw.AttnNorm,
			FFNNorm:  bw.FFNNorm,
			HeadNorm: headNorm,
		}); err != nil {
			pipe.Close()
			return fmt.Errorf("qwen35: prediction block: %w", err)
		}
		if m.W.MTP.EHProj.Quant != nn.Q8_0 {
			pipe.Close()
			return fmt.Errorf("qwen35: the prediction block's projection is %s; the card reads Q8_0", m.W.MTP.EHProj.Quant)
		}
	}

	m.gpuPipe = pipe
	return nil
}

func (m *Model) UseVulkan() error {
	if err := m.UseVulkanStack(); err != nil {
		return err
	}
	return m.UseVulkanHead()
}

func (m *Model) closeVulkan() {
	if m.headQ6K != nil {
		m.headQ6K.Close()
		m.headQ6K = nil
	}
	if m.head != nil {
		m.head.Close()
		m.head = nil
	}
	if m.gpuPipe != nil {
		m.gpuPipe.Close()
		m.gpuPipe = nil
	}
	if m.dev != nil {
		m.dev.Close()
		m.dev = nil
	}
}
