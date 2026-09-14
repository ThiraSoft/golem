package nomic

// The card. vk/nomic.go runs the blocks; this hands it the weights, does on
// the host what is cheaper there — the embedding lookup, a table of a quarter
// of a million rows read a few thousand at a time, and the router's choice —
// and pools what comes back.

import (
	"encoding/binary"
	"fmt"

	"github.com/ThiraSoft/golem/nn"
	"github.com/ThiraSoft/golem/vk"
)

// UseVulkan puts the blocks on a Vulkan device. It fails rather than falling
// back: an embedder the caller believed was on the card and is not is a
// capacity plan built on a flag that did nothing.
//
// The card reads fp16. An fp16 file hands its bytes over as they are; any other
// format is widened a row at a time and rounded to fp16 on the way up, which
// is not what ggml computes for that format — it quantizes the activation
// instead — and is the same model to within that format's own noise.
func (m *Model) UseVulkan() error {
	m.turn <- struct{}{}
	defer m.release()
	if m.gpu != nil {
		return nil
	}
	cfg := m.Cfg
	d, err := vk.Open()
	if err != nil {
		return err
	}
	p, err := vk.NewNomicPipeline(d, vk.NomicShape{
		Dim: cfg.Dim, Heads: cfg.Heads, HeadDim: cfg.HeadDim, FF: cfg.FF,
		Experts: cfg.Experts, Used: cfg.ExpertsUsed,
		Eps: cfg.Eps, RoPEBase: float32(cfg.RoPEBase),
	})
	if err != nil {
		d.Close()
		return err
	}
	fail := func(err error) error {
		p.Close()
		d.Close()
		return err
	}
	if err := p.SetEmbedNorm(m.W.EmbNorm.Gain, m.W.EmbNorm.Bias); err != nil {
		return fail(err)
	}
	for i := range m.W.Blocks {
		b := &m.W.Blocks[i]
		data := vk.NomicBlockData{
			QKV: linear(b.QKV), O: linear(b.O),
			AttnGain: b.AttnNorm.Gain, AttnBias: b.AttnNorm.Bias,
			OutGain: b.OutNorm.Gain, OutBias: b.OutNorm.Bias,
		}
		if cfg.Mixture(i) {
			data.Router = widen(b.Router)
			for e := range b.UpExps {
				data.UpExps = append(data.UpExps, linear(b.UpExps[e]))
				data.DownExps = append(data.DownExps, linear(b.DownExps[e]))
			}
		} else {
			data.Up, data.Down = linear(b.Up), linear(b.Down)
		}
		if err := p.AddBlock(data); err != nil {
			return fail(fmt.Errorf("nomic: block %d: %w", i, err))
		}
	}
	if err := p.Prepare(); err != nil {
		return fail(err)
	}
	m.dev, m.gpu = d, p
	return nil
}

// Vulkan says whether the blocks are on a device.
func (m *Model) Vulkan() bool { return m.gpu != nil }

func (m *Model) closeVulkan() {
	if m.gpu != nil {
		m.gpu.Close()
		m.gpu = nil
	}
	if m.dev != nil {
		m.dev.Close()
		m.dev = nil
	}
}

// linear is a matrix as the card reads it: fp16, one row per output.
func linear(mat nn.Matrix) vk.NomicLinear {
	if mat.Quant == nn.F16 {
		return vk.NomicLinear{W: mat.Data, Bias: mat.Bias}
	}
	w := make([]byte, mat.Rows*mat.Cols*2)
	row := make([]float32, mat.Cols)
	for r := 0; r < mat.Rows; r++ {
		mat.Row(r, row)
		for c, v := range row {
			binary.LittleEndian.PutUint16(w[2*(r*mat.Cols+c):], nn.FloatToHalf(v))
		}
	}
	return vk.NomicLinear{W: w, Bias: mat.Bias}
}

// widen is a matrix as float32 rows, which is what the router is in the file.
func widen(mat nn.Matrix) []float32 {
	out := make([]float32, mat.Rows*mat.Cols)
	for r := 0; r < mat.Rows; r++ {
		mat.Row(r, out[r*mat.Cols:(r+1)*mat.Cols])
	}
	return out
}

// hiddenVulkan is hidden's pass on the card. x holds each position's
// embedding with the sentence-A row added; it gets back what the last block
// wrote.
func (m *Model) hiddenVulkan(x [][]float32, pos []int, spans []span) error {
	cfg := m.Cfg
	n := len(x)
	flat := flatOf(x, n)
	at := make([]uint32, n)
	seg := make([]uint32, 2*n)
	for t, p := range pos {
		at[t] = uint32(p)
	}
	for _, sp := range spans {
		for j := 0; j < sp.length; j++ {
			t := sp.start + j
			seg[2*t], seg[2*t+1] = uint32(sp.start), uint32(sp.length)
		}
	}

	route := func(block int, logits []float32) ([]uint32, []uint32, []float32, []int) {
		k := cfg.ExpertsUsed
		if m.trace != nil {
			m.emit(fmt.Sprintf("ffn_moe_logits-%d", block), splitRows(logits, cfg.Experts))
		}
		choice := make([]int, n*k)
		weight := make([]float32, n*k)
		probs := make([]float32, cfg.Experts)
		counts := make([]int, cfg.Experts)
		for t := 0; t < n; t++ {
			pick(logits[t*cfg.Experts:(t+1)*cfg.Experts], probs, choice[t*k:(t+1)*k], weight[t*k:(t+1)*k])
			for _, e := range choice[t*k : (t+1)*k] {
				counts[e]++
			}
		}
		if m.trace != nil {
			m.emit(fmt.Sprintf("ffn_moe_weights-%d", block), splitRows(weight, k))
		}
		next := make([]int, cfg.Experts)
		for e := 1; e < cfg.Experts; e++ {
			next[e] = next[e-1] + counts[e-1]
		}
		idx := make([]uint32, n*k)
		dest := make([]uint32, n*k)
		for i, e := range choice {
			idx[next[e]], dest[next[e]] = uint32(i/k), uint32(i)
			next[e]++
		}
		return idx, dest, weight, counts
	}

	if m.trace != nil {
		m.gpu.Trace()
	}
	if err := m.gpu.Encode(flat, at, seg, route, flat); err != nil {
		return err
	}
	if m.trace != nil {
		for i := range m.W.Blocks {
			for _, name := range []string{"kqv_out", "ffn_inp"} {
				w := fmt.Sprintf("%s-%d", name, i)
				m.emit(w, splitRows(m.gpu.Waypoint(w), cfg.Dim))
			}
		}
		m.emit("result_embd", x)
	}
	return nil
}

// splitRows views a flat slice as rows of width.
func splitRows(flat []float32, width int) [][]float32 {
	out := make([][]float32, len(flat)/width)
	for i := range out {
		out[i] = flat[i*width : (i+1)*width]
	}
	return out
}
