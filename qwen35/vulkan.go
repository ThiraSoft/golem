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
	if m.headQ6K != nil || m.head != nil || m.golemHead != nil {
		return nil
	}
	d, err := m.device()
	if err != nil {
		return err
	}
	if gq := m.W.OutputHead.Quant; gq.Golem() {
		// A .golem head is a site like any other: the hidden state meets the
		// reciprocal of its scale and the same rotation before the product,
		// and vk/golemhead.go does both.
		k, err := vk.NewGolemKernels(d, gq)
		if err != nil {
			return fmt.Errorf("qwen35: cannot build the %s kernels: %w", gq, err)
		}
		h, err := vk.NewGolemHead(k, m.W.OutputHead.Data, m.W.OutputHead.Rows, m.W.OutputHead.Cols, m.W.OutputHead.Pre)
		if err != nil {
			k.Close()
			return fmt.Errorf("qwen35: cannot upload the Golem head to Vulkan: %w", err)
		}
		m.golemKernels, m.golemHead = k, h
		return nil
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
	return m.headQ6K != nil || m.head != nil || m.golemHead != nil
}

// StartVulkanCalibration turns on the per-site accumulators of the block
// stack, so that a conversion can measure what every matrix is fed without
// reading a single activation back across the bus. vk/qwen_calib.go says what
// they are and why they are there rather than on the processor.
func (m *Model) StartVulkanCalibration() error {
	if m.gpuPipe == nil {
		return fmt.Errorf("qwen35: there is no device stack to calibrate on")
	}
	return m.gpuPipe.StartCalibration()
}

// CountVulkanCalibration tells the accumulators how many rows they have seen.
func (m *Model) CountVulkanCalibration(rows int) {
	if m.gpuPipe != nil {
		m.gpuPipe.CountCalibration(rows)
	}
}

// VulkanCalibrationSums is the per-column power of every site, and the rows it
// was taken over, filed under the block and the site — "7/qkv".
func (m *Model) VulkanCalibrationSums() (map[string][]float32, int, error) {
	if m.gpuPipe == nil {
		return nil, 0, fmt.Errorf("qwen35: there is no device stack to read")
	}
	return m.gpuPipe.CalibrationSums()
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

	// A .golem checkpoint takes the other form of every projection. The format
	// is the model's and not a block's, so it is read once here and carried in
	// the shape; anything of llama.cpp's own is not one of these.
	gq := m.W.Blocks[0].Down.Quant
	if !gq.Golem() {
		gq = 0
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
		Golem:        gq,
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

		// One format for the whole model. A checkpoint half converted would
		// load, because every matrix carries its own type, and would answer
		// nonsense from whichever half the pipeline read the other way.
		mixer := []nn.Matrix{bw.Q, bw.K, bw.V, bw.O}
		if bc.Type != BlockFullAttn {
			mixer = []nn.Matrix{bw.QKV, bw.AttnGate, bw.SSMAlpha, bw.SSMBeta, bw.SSMOut}
		}
		if gq.Golem() {
			for _, w := range append(mixer, bw.Gate, bw.Up, bw.Down) {
				if w.Quant != gq {
					pipe.Close()
					return fmt.Errorf("qwen35: block %d has a %s among %s", i, w.Quant, gq)
				}
			}
		}

		attnNorms = append(attnNorms, bw.AttnNorm)
		ffnNorms = append(ffnNorms, bw.FFNNorm)
		isSSM = append(isSSM, bc.Type != BlockFullAttn)

		if err := pipe.AddFFNBlock(vk.QwenFFNData{
			Gate:      bw.Gate.Data,
			Up:        bw.Up.Data,
			Down:      bw.Down.Data,
			GateUp:    bw.Gate.Quant,
			DownQ:     bw.Down.Quant,
			PreGateUp: bw.Gate.Pre,
			PreDown:   bw.Down.Pre,
		}); err != nil {
			pipe.Close()
			return fmt.Errorf("qwen35: block %d feed forward: %w", i, err)
		}

		if bc.Type == BlockFullAttn {
			err = pipe.AddAttnBlock(i, vk.QwenAttnData{
				WQ: bw.Q.Data, WK: bw.K.Data, WV: bw.V.Data, WO: bw.O.Data,
				QNorm: bw.QNorm, KNorm: bw.KNorm,
				PreQKV: bw.Q.Pre, PreO: bw.O.Pre,
			})
		} else {
			err = pipe.AddSSMBlock(i, vk.QwenSSMData{
				PreQKV:     bw.QKV.Pre,
				PreO:       bw.SSMOut.Pre,
				WQKV:       bw.QKV.Data,
				WGate:      bw.AttnGate.Data,
				WAlpha:     bw.SSMAlpha.Data,
				WBeta:      bw.SSMBeta.Data,
				WOut:       bw.SSMOut.Data,
				Out:        bw.SSMOut.Quant,
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
		// The prediction block is the trunk's last block read a second time,
		// with a projection of its own in front of it. That projection is the
		// one matrix in the model no calibration site names, so in a .golem it
		// carries its own vector rather than a site's — see QwenMTPData.
		if q := m.W.MTP.EHProj.Quant; q != gq && !(!gq.Golem() && q == nn.Q8_0) {
			pipe.Close()
			return fmt.Errorf("qwen35: the prediction block's projection is %s where the model is %s",
				q, m.W.Blocks[0].Down.Quant)
		}
		if err := pipe.AddMTPBlock(vk.QwenMTPData{
			EHProj: m.W.MTP.EHProj.Data,
			PreEH:  m.W.MTP.EHProj.Pre,
			Attn: vk.QwenAttnData{
				WQ: bw.Q.Data, WK: bw.K.Data, WV: bw.V.Data, WO: bw.O.Data,
				QNorm: bw.QNorm, KNorm: bw.KNorm,
				PreQKV: bw.Q.Pre, PreO: bw.O.Pre,
			},
			FFN: vk.QwenFFNData{
				Gate: bw.Gate.Data, Up: bw.Up.Data, Down: bw.Down.Data,
				GateUp:    bw.Gate.Quant,
				DownQ:     bw.Down.Quant,
				PreGateUp: bw.Gate.Pre,
				PreDown:   bw.Down.Pre,
			},
			AttnNorm: bw.AttnNorm,
			FFNNorm:  bw.FFNNorm,
			HeadNorm: headNorm,
		}); err != nil {
			pipe.Close()
			return fmt.Errorf("qwen35: prediction block: %w", err)
		}
	}

	m.gpuPipe = pipe
	return nil
}

func (m *Model) UseVulkan() error {
	if err := m.UseVulkanStack(); err != nil {
		return err
	}
	if err := m.UseVulkanHead(); err != nil {
		return err
	}
	// The image tower too, when a projector has been opened. It is last
	// because it is the part that may not fit: the blocks take the card's
	// memory first, and what the tower does with what is left is its own
	// business — see VisionPipeline.Prepare.
	return m.UseVisionVulkan()
}

func (m *Model) closeVulkan() {
	if m.golemHead != nil {
		m.golemHead.Close()
		m.golemHead = nil
	}
	if m.golemKernels != nil {
		m.golemKernels.Close()
		m.golemKernels = nil
	}
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
