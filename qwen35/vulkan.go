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
	// A prediction block is only worth its memory to a caller that will draft
	// with it, and on the 27B that memory is what decides whether the logit
	// head stays on the card: the block's own weights and a shadow of every
	// delta net's recurrence, which together are most of a gigabyte.
	drafts := m.HasMTP() && !m.noDraft
	shape := vk.QwenShape{
		Snapshots:  drafts,
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

	// What this card can afford: whether the prediction block goes over, and
	// how wide a prompt pass the scratch may be. Both asked of the device.
	drafts, width := m.deviceBudget(d, shape, numBlocks, drafts)
	shape.Snapshots, shape.PassWidth = drafts, width

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
				Formats: vk.BlockFormats{Q: bw.Q.Quant, K: bw.K.Quant, V: bw.V.Quant, O: bw.O.Quant},
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
				QKV:        bw.QKV.Quant,
				Gate:       bw.AttnGate.Quant,
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

	if drafts && numBlocks == m.trunk() {
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
		// A .golem carries every matrix in the model's one form, and the check
		// is that this one is not the exception. Anything of llama.cpp's own
		// is asked no such question: a K-quant mix files this projection where
		// it likes — Q8_0 in the builds seen first, Q4_K in a Q4_K_M — and the
		// pipeline refuses by name a form it has no kernel for.
		if q := m.W.MTP.EHProj.Quant; gq.Golem() && q != gq {
			pipe.Close()
			return fmt.Errorf("qwen35: the prediction block's projection is %s where the model is %s",
				q, m.W.Blocks[0].Down.Quant)
		}
		if err := pipe.AddMTPBlock(vk.QwenMTPData{
			EHProj: m.W.MTP.EHProj.Data,
			EHQ:    m.W.MTP.EHProj.Quant,
			PreEH:  m.W.MTP.EHProj.Pre,
			Attn: vk.QwenAttnData{
				WQ: bw.Q.Data, WK: bw.K.Data, WV: bw.V.Data, WO: bw.O.Data,
				QNorm: bw.QNorm, KNorm: bw.KNorm,
				PreQKV: bw.Q.Pre, PreO: bw.O.Pre,
				Formats: vk.BlockFormats{Q: bw.Q.Quant, K: bw.K.Quant, V: bw.V.Quant, O: bw.O.Quant},
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
	// The head first: it is the largest tensor here and the blocks take the
	// card greedily, so a head uploaded after them is the allocation the
	// driver quietly puts in host memory — where it crosses the bus once per
	// token. engine/engine.go's UseVulkan carries the measurement.
	if err := m.UseVulkanHead(); err != nil {
		return err
	}
	if err := m.UseVulkanStack(); err != nil {
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

// SkipDraftBlock says this model will not draft, so UseVulkanStack should leave
// the prediction block and the delta nets' shadow states off the card.
//
// It is a saving worth naming. On Qwen3.8-27B-Q4_K_M the block's own weights
// are about two hundred and thirty megabytes and the shadows a hundred and
// fifty, and the model is 15.65 GiB on a card that holds 15.92: with them the
// driver puts the logit head in system memory and the model draws at three and
// a half tokens a second, and without them it stays resident. Speculation is
// worth about thirty per cent when it fits; a gigabyte across the bus every
// token is worth minus eighty-eight.
//
// It must be called before UseVulkan, and it is ignored afterwards — the
// weights are already there.
func (m *Model) SkipDraftBlock() { m.noDraft = true }

// deviceWeightBytes is what this model will put on the card, exactly: every
// matrix the upload below walks, plus the logit head as the packing will hold
// it. It is summed rather than taken from the file's size, because the token
// embedding is never uploaded — the host looks a row up and hands the pipeline
// floats — and on the 27B that is seven hundred megabytes of difference.
func (m *Model) deviceWeightBytes(blocks int, drafts bool) uint64 {
	var n uint64
	add := func(w nn.Matrix) { n += uint64(len(w.Data)) }
	for i := 0; i < blocks; i++ {
		bw := &m.W.Blocks[i]
		for _, w := range []nn.Matrix{bw.Q, bw.K, bw.V, bw.O, bw.QKV, bw.AttnGate,
			bw.SSMAlpha, bw.SSMBeta, bw.SSMOut, bw.Gate, bw.Up, bw.Down} {
			add(w)
		}
	}
	if drafts && len(m.W.Blocks) > m.trunk() {
		bw := &m.W.Blocks[m.trunk()]
		for _, w := range []nn.Matrix{bw.Q, bw.K, bw.V, bw.O, bw.Gate, bw.Up, bw.Down} {
			add(w)
		}
		add(m.W.MTP.EHProj)
	}
	// The head, in the layout the card holds it in. A Q6_K superblock is two
	// hundred and ten bytes in the file and two hundred and twelve here, which
	// on a quarter of a million rows is ten megabytes and is not noise at this
	// margin.
	head := m.W.OutputHead
	if head.Quant == nn.Q6_K {
		n += uint64(head.Rows) * uint64(head.Cols) / 256 * 212
	} else {
		add(head)
	}
	return n
}

// deviceBudget decides two things together, because they trade against each
// other on a card that is nearly full: whether the prediction block goes over
// at all, and how wide a prompt pass the scratch may be.
//
// What is at stake is not the scratch. The logit head is the last thing
// uploaded and the largest single buffer in the model — a gigabyte on the 27B —
// and when the heap runs out it is the one the driver leaves in system memory,
// where it is read across the bus for every token drawn. Measured on
// Qwen3.8-27B-Q4_K_M: 3.5 tokens a second with the head exiled, 26.1 with it
// resident. Nothing else this file can decide is worth an eighth of that.
//
// So both answers are taken from the card's own heap rather than from a
// constant. Drafting costs the prediction block's weights and a shadow of every
// delta net's recurrence, together about four hundred mebibytes, and it buys
// perhaps thirty per cent; it is dropped rather than allowed to exile the head.
// The width costs scratch and buys prefill, and it is stepped down until it
// fits.
//
// The margin is what the rest of the machine has of the card. A desktop with a
// browser on it measured five hundred mebibytes here and the driver keeps some
// of the heap for itself, so this leaves three quarters of a gigabyte. A margin
// too large costs prefill; one too small costs seven eighths of the generation.
func (m *Model) deviceBudget(d *vk.Device, shape vk.QwenShape, blocks int, wantDraft bool) (bool, int) {
	const margin = 768 << 20
	heap := d.DeviceLocalBytes()
	attn, ssm := 0, 0
	for i := 0; i < blocks; i++ {
		if m.Cfg.Blocks[i].Type == BlockFullAttn {
			attn++
		} else {
			ssm++
		}
	}

	// fits says whether that configuration leaves the card room, at its
	// narrowest useful pass.
	fits := func(drafts bool, floor int) bool {
		s := shape
		s.Snapshots = drafts
		extra := 0
		if drafts {
			extra = 1 // the prediction block's own attention keeps a cache
		}
		return m.deviceWeightBytes(blocks, drafts)+
			vk.QwenBufferBytes(s, floor, attn+extra, ssm)+margin <= heap
	}

	drafts := wantDraft
	if drafts && !fits(true, 2) {
		// A speculative pass carries two columns, so two is the floor; below
		// it the block cannot be used at all.
		fmt.Fprintf(os.Stderr,
			"qwen35: the prediction block does not fit beside the model on this card; drafting is off\n")
		drafts = false
	}
	shape.Snapshots = drafts
	floor := 1
	if drafts {
		floor = 2
	}
	extra := 0
	if drafts {
		extra = 1
	}
	for _, w := range []int{512, 256, 128, 64, 32, 16, 8, 4, 2, 1} {
		if w < floor {
			break
		}
		if m.deviceWeightBytes(blocks, drafts)+
			vk.QwenBufferBytes(shape, w, attn+extra, ssm)+margin <= heap {
			return drafts, w
		}
	}
	// Nothing fits with room to spare. Take the narrowest and let the driver
	// place what it can: a model that answers slowly is better than one that
	// refuses.
	return drafts, floor
}
