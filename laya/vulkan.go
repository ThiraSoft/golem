package laya

// The card. vk/laya.go runs the embedding, the encoder and the head; this
// hands it the weights and the rotation tables and scores the markers that
// come back, which is a few rows.

import (
	"encoding/binary"
	"fmt"

	"github.com/ThiraSoft/golem/nn"
	"github.com/ThiraSoft/golem/vk"
)

type gpu struct {
	dev  *vk.Device
	pipe *vk.LayaPipeline
}

// UseVulkan puts the model on a Vulkan device. It fails rather than falling
// back: a model the caller believed was on the card and is not is a capacity
// plan built on a flag that did nothing.
//
// The card reads fp16 weights and carries every product's operand as two
// fp16 planes, its high part and what is left of it, which keeps it to about
// twenty-two bits of the float32 the processor's path multiplies; the two
// also sum in different orders, and differ by that.
func (m *Model) UseVulkan() error {
	m.turn <- struct{}{}
	defer m.release()
	if m.gpu != nil {
		return nil
	}
	cfg, w := m.Cfg, m.W
	d, err := vk.Open()
	if err != nil {
		return err
	}
	p, err := vk.NewLayaPipeline(d, vk.LayaShape{
		Dim: cfg.Dim, Heads: cfg.Heads, HeadDim: cfg.HeadDim, FF: cfg.FF, HeadFF: cfg.HeadFF,
		Window: cfg.Window, MaxLen: cfg.MaxLen, Eps: cfg.Eps,
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
	if err := p.SetRoPE(ropeTable(cfg, cfg.GlobalBase), ropeTable(cfg, cfg.LocalBase)); err != nil {
		return fail(err)
	}
	if err := p.SetEmbedding(linear(w.TokenEmbd).W, w.EmbNorm.Gain); err != nil {
		return fail(err)
	}
	if err := p.SetScorer(w.ScoreNorm.Gain, w.ScoreNorm.Bias, linear(w.ScoreHid)); err != nil {
		return fail(err)
	}
	for i := range w.Encoder {
		b := &w.Encoder[i]
		if err := p.AddEncoderBlock(vk.LayaEncoderBlock{
			AttnGain: b.AttnNorm.Gain, MLPGain: b.MLPNorm.Gain,
			QKV: linear(b.QKV), O: linear(b.O), Up: linear(b.Up), Down: linear(b.Down),
			Global: cfg.Global(i),
		}); err != nil {
			return fail(fmt.Errorf("laya: block %d: %w", i, err))
		}
	}
	if err := p.SetFinalNorm(w.FinalNorm.Gain); err != nil {
		return fail(err)
	}
	var kinds []float32
	for _, row := range w.TypeEmbd {
		kinds = append(kinds, row...)
	}
	if err := p.SetKinds(kinds); err != nil {
		return fail(err)
	}
	for i := range w.Head {
		b := &w.Head[i]
		if err := p.AddHeadBlock(vk.LayaHeadBlock{
			Norm1Gain: b.Norm1.Gain, Norm1Bias: b.Norm1.Bias, Norm2Gain: b.Norm2.Gain, Norm2Bias: b.Norm2.Bias,
			QKV: linear(b.QKV), O: linear(b.O), Up: linear(b.Up), Down: linear(b.Down),
		}); err != nil {
			return fail(fmt.Errorf("laya: head layer %d: %w", i, err))
		}
	}
	if err := p.Prepare(); err != nil {
		return fail(err)
	}
	m.gpu = &gpu{dev: d, pipe: p}
	return nil
}

// Vulkan says whether the model is on a device.
func (m *Model) Vulkan() bool { return m.gpu != nil }

func (m *Model) closeVulkan() {
	if m.gpu != nil {
		m.gpu.pipe.Close()
		m.gpu.dev.Close()
		m.gpu = nil
	}
}

// ropeTable is every position's cosines and sines for one base, half a head
// of each, computed as the processor's path computes them.
func ropeTable(cfg *Config, base float64) []float32 {
	out := make([]float32, 0, cfg.MaxLen*cfg.HeadDim)
	var t nn.RoPETable
	for p := 0; p < cfg.MaxLen; p++ {
		t.Prepare(cfg.HeadDim, p, base, nil)
		out = append(out, t.Cos...)
		out = append(out, t.Sin...)
	}
	return out
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

// encodeVulkan is the embedding, encode, head and the scorer's hidden layer
// on the card. What comes back is x as score reads it, a row for every
// position of which only each sequence's first is filled, and the scorer's
// hidden layer at every marker, in order.
func (m *Model) encodeVulkan(ids []int32, pos []int, spans []span, seqs []seq) (x, hid [][]float32, err error) {
	n, d := len(ids), m.Cfg.Dim
	tokens := make([]uint32, n)
	at := make([]uint32, n)
	seg := make([]uint32, 2*n)
	kind := make([]uint32, n)
	for t, p := range pos {
		tokens[t] = uint32(ids[t])
		at[t] = uint32(p)
	}
	var want, marks []uint32
	for i, sp := range spans {
		for j := 0; j < sp.length; j++ {
			t := sp.start + j
			seg[2*t], seg[2*t+1] = uint32(sp.start), uint32(sp.length)
			kind[t] = uint32(seqs[i].kind)
		}
		want = append(want, uint32(sp.start))
		for _, mk := range seqs[i].markers {
			marks = append(marks, uint32(sp.start+mk))
		}
	}
	p := m.gpu.pipe
	if m.trace != nil {
		p.Trace()
	}
	flat := make([]float32, len(want)*d)
	hidden := make([]float32, len(marks)*d)
	if err := p.Encode(tokens, at, seg, kind, want, marks, flat, hidden); err != nil {
		return nil, nil, err
	}
	x = make([][]float32, n)
	for r, t := range want {
		x[t] = flat[r*d : (r+1)*d]
	}
	hid = split(hidden, d)
	if m.trace != nil {
		for i := 0; i < m.Cfg.Blocks; i++ {
			name := fmt.Sprintf("layer-%d", i)
			m.emit(name, split(p.Waypoint(name), d))
		}
		m.emit("encoded", split(p.Waypoint("encoded"), d))
		for i := 0; i < m.Cfg.HeadLayers; i++ {
			name := fmt.Sprintf("head-%d", i)
			m.emit(name, split(p.Waypoint(name), d))
		}
	}
	return x, hid, nil
}

func split(flat []float32, width int) [][]float32 {
	out := make([][]float32, len(flat)/width)
	for i := range out {
		out[i] = flat[i*width : (i+1)*width]
	}
	return out
}
