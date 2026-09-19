package krea2

// The DiT: Krea 2's single-stream transformer (comfy/ldm/krea2/model.py),
// from ComfyUI's fp8 checkpoint.
//
// A call is in two parts, because one of them does not depend on the step.
// The text's part — the text fusion over the twelve taps and the text MLP —
// runs once per prompt and leaves its answer on the card; the step's part —
// the timestep vectors, the patches, twenty-eight blocks and the last layer —
// runs at every step, reading that answer.

import (
	"fmt"
	"math"

	"github.com/ThiraSoft/golem/vk"
)

// The DiT's shape (SingleStreamDiT's defaults, which the checkpoint has).
const (
	ditWidth   = 6144
	ditHeads   = 48
	ditKVHeads = 12
	ditHeadDim = 128
	ditFFN     = 16384
	ditBlocks  = 28
	ditTDim    = 256
	ditPatch   = 2
	ditLatent  = 16
	ditIn      = ditLatent * ditPatch * ditPatch
	ditEps     = 1e-5
	ditTheta   = 1000

	txtWidth  = encWidth
	txtHeads  = 20
	txtFFN    = 6912
	txtLayers = 12 // len(Taps)

	// MaxImageTokens bounds the patches of a picture: 1024 × 1024.
	MaxImageTokens = 4096
)

// ropeAxes is how a head's width is shared between the three position axes.
var ropeAxes = [3]int{ditHeadDim - 12*(ditHeadDim/16), 6 * (ditHeadDim / 16), 6 * (ditHeadDim / 16)}

type attnWeights struct {
	q, k, v, gate, o int
	qNorm, kNorm     uint32
}

type ditBlock struct {
	attnWeights
	gate, up, down int
	pre, post      uint32 // norm scales
	mod            uint32 // the block's six modulation vectors
}

type txtBlock struct {
	attnWeights
	gate, up, down int
	pre, post      uint32
}

// DiT is Krea 2's transformer on the card.
type DiT struct {
	k  *vk.K2
	ck *checkpoint

	blocks                   [ditBlocks]ditBlock
	layerwise, refiner       [2]txtBlock
	projector                [txtLayers]float32
	first, tmlp0, tmlp2      int
	tproj, txt1, txt3, lastW int
	firstB, tmlp0B, tmlp2B   uint32
	tprojB, txt1B, txt3B     uint32
	txtNorm, lastB           uint32
	lastNorm, lastMod        uint32

	// The arena.
	temb, tA, t, tg, tvec uint32
	ctx                   uint32
	x, n, scratch         uint32
	table, img, out       uint32

	fusion map[int][2]*vk.Program
	steps  map[[2]int]*vk.Program
	ctxLen int
}

// OpenDiT uploads the DiT to the card: twelve gigabytes, most of the card.
func OpenDiT(d *vk.Device, path string) (*DiT, error) {
	ck, err := openCheckpoint(path)
	if err != nil {
		return nil, err
	}
	m := &DiT{ck: ck, fusion: map[int][2]*vk.Program{}, steps: map[[2]int]*vk.Program{}}
	fail := func(err error) (*DiT, error) {
		m.Close()
		return nil, err
	}
	var par params
	vec := func(name string, n int) (uint32, error) {
		v, err := ck.floats(name, n)
		if err != nil {
			return 0, err
		}
		return par.add(v), nil
	}
	norms := func(pre string, a *attnWeights) (err error) {
		if a.qNorm, err = vec(pre+"attn.qknorm.qnorm.scale", ditHeadDim); err != nil {
			return
		}
		a.kNorm, err = vec(pre+"attn.qknorm.knorm.scale", ditHeadDim)
		return
	}
	for i := range m.blocks {
		b := &m.blocks[i]
		pre := fmt.Sprintf("blocks.%d.", i)
		if err = norms(pre, &b.attnWeights); err != nil {
			return fail(err)
		}
		if b.pre, err = vec(pre+"prenorm.scale", ditWidth); err != nil {
			return fail(err)
		}
		if b.post, err = vec(pre+"postnorm.scale", ditWidth); err != nil {
			return fail(err)
		}
		if b.mod, err = vec(pre+"mod.lin", 6*ditWidth); err != nil {
			return fail(err)
		}
	}
	txt := func(pre string, b *txtBlock) (err error) {
		if err = norms(pre, &b.attnWeights); err != nil {
			return
		}
		if b.pre, err = vec(pre+"prenorm.scale", txtWidth); err != nil {
			return
		}
		b.post, err = vec(pre+"postnorm.scale", txtWidth)
		return
	}
	for i := 0; i < 2; i++ {
		if err = txt(fmt.Sprintf("txtfusion.layerwise_blocks.%d.", i), &m.layerwise[i]); err != nil {
			return fail(err)
		}
		if err = txt(fmt.Sprintf("txtfusion.refiner_blocks.%d.", i), &m.refiner[i]); err != nil {
			return fail(err)
		}
	}
	proj, err := ck.floats("txtfusion.projector.weight", txtLayers)
	if err != nil {
		return fail(err)
	}
	copy(m.projector[:], proj)
	for _, v := range []struct {
		at   *uint32
		name string
		n    int
	}{
		{&m.firstB, "first.bias", ditWidth}, {&m.tmlp0B, "tmlp.0.bias", ditWidth}, {&m.tmlp2B, "tmlp.2.bias", ditWidth},
		{&m.tprojB, "tproj.1.bias", 6 * ditWidth}, {&m.txtNorm, "txtmlp.0.scale", txtWidth},
		{&m.txt1B, "txtmlp.1.bias", ditWidth}, {&m.txt3B, "txtmlp.3.bias", ditWidth},
		{&m.lastB, "last.linear.bias", ditIn}, {&m.lastNorm, "last.norm.scale", ditWidth},
		{&m.lastMod, "last.modulation.lin", 2 * ditWidth},
	} {
		if *v.at, err = vec(v.name, v.n); err != nil {
			return fail(err)
		}
	}

	var a arena
	m.temb = a.take(ditTDim)
	m.tA = a.take(ditWidth)
	m.t = a.take(ditWidth)
	m.tg = a.take(ditWidth)
	m.tvec = a.take(6 * ditWidth)
	m.ctx = a.take(MaxPromptTokens * ditWidth)
	N := MaxPromptTokens + MaxImageTokens
	R := MaxPromptTokens * txtLayers
	m.x = a.take(max(N*ditWidth, R*txtWidth))
	m.n = a.take(max(N*ditWidth, R*txtWidth))
	m.scratch = a.take(max(N*2*ditFFN, R*2*txtFFN, R*5*txtWidth))
	m.table = a.take(N * ditHeadDim)
	m.img = a.take(MaxImageTokens * ditIn)
	m.out = a.take(MaxImageTokens * ditIn)

	if m.k, err = vk.NewK2(d, a.n, par.data); err != nil {
		return fail(err)
	}
	up := func(h *int, name string, rows, cols int) {
		if err == nil {
			*h, err = ck.fp8(m.k, name, rows, cols)
		}
	}
	upAttn := func(pre string, a *attnWeights, width, kv int) {
		up(&a.q, pre+"attn.wq.weight", width, width)
		up(&a.k, pre+"attn.wk.weight", kv, width)
		up(&a.v, pre+"attn.wv.weight", kv, width)
		up(&a.gate, pre+"attn.gate.weight", width, width)
		up(&a.o, pre+"attn.wo.weight", width, width)
	}
	for i := range m.blocks {
		b := &m.blocks[i]
		pre := fmt.Sprintf("blocks.%d.", i)
		upAttn(pre, &b.attnWeights, ditWidth, ditKVHeads*ditHeadDim)
		up(&b.gate, pre+"mlp.gate.weight", ditFFN, ditWidth)
		up(&b.up, pre+"mlp.up.weight", ditFFN, ditWidth)
		up(&b.down, pre+"mlp.down.weight", ditWidth, ditFFN)
	}
	for i := 0; i < 2; i++ {
		for _, tb := range []struct {
			b   *txtBlock
			pre string
		}{{&m.layerwise[i], fmt.Sprintf("txtfusion.layerwise_blocks.%d.", i)}, {&m.refiner[i], fmt.Sprintf("txtfusion.refiner_blocks.%d.", i)}} {
			upAttn(tb.pre, &tb.b.attnWeights, txtWidth, txtWidth)
			up(&tb.b.gate, tb.pre+"mlp.gate.weight", txtFFN, txtWidth)
			up(&tb.b.up, tb.pre+"mlp.up.weight", txtFFN, txtWidth)
			up(&tb.b.down, tb.pre+"mlp.down.weight", txtWidth, txtFFN)
		}
	}
	up(&m.first, "first.weight", ditWidth, ditIn)
	up(&m.tmlp0, "tmlp.0.weight", ditWidth, ditTDim)
	up(&m.tmlp2, "tmlp.2.weight", ditWidth, ditWidth)
	up(&m.tproj, "tproj.1.weight", 6*ditWidth, ditWidth)
	up(&m.txt1, "txtmlp.1.weight", ditWidth, txtWidth)
	up(&m.txt3, "txtmlp.3.weight", ditWidth, ditWidth)
	up(&m.lastW, "last.linear.weight", ditIn, ditWidth)
	if err != nil {
		return fail(err)
	}
	return m, nil
}

func (m *DiT) mm(r *vk.Recorder, w int, outs, ins, cols, x, y, bias, mode uint32) {
	m.k.MM(r, w, true, vk.K2MM{Outputs: outs, Inputs: ins, Cols: cols, X: x, XStride: ins, Y: y, YStride: outs,
		Bias: bias, Scale: 1, Mode: mode})
}

func (m *DiT) gatedMM(r *vk.Recorder, w int, outs, ins, cols, x, y, gateA, gateB uint32) {
	m.k.MM(r, w, true, vk.K2MM{Outputs: outs, Inputs: ins, Cols: cols, X: x, XStride: ins, Y: y, YStride: outs,
		Bias: vk.K2None, Scale: 1, Mode: vk.K2GatedAdd, GateA: gateA, GateB: gateB})
}

// textBlock records a TextFusionBlock over seqs sequences of length each.
func (m *DiT) textBlock(r *vk.Recorder, b *txtBlock, seqs, length uint32) {
	rows := seqs * length
	const w = txtWidth
	q := m.scratch
	k := q + rows*w
	v := k + rows*w
	g := v + rows*w
	att := g + rows*w
	norm := func(src, dst, n, rows, weight uint32) {
		m.k.Norm(r, vk.K2Norm{Src: src, SrcStride: n, Dst: dst, DstStride: n, N: n, Rows: rows, Weight: weight, OnePlus: 1, Eps: ditEps, ScaleA: vk.K2None})
	}
	norm(m.x, m.n, w, rows, b.pre)
	m.mm(r, b.q, w, w, rows, m.n, q, vk.K2None, vk.K2Store)
	m.mm(r, b.k, w, w, rows, m.n, k, vk.K2None, vk.K2Store)
	m.mm(r, b.v, w, w, rows, m.n, v, vk.K2None, vk.K2Store)
	m.mm(r, b.attnWeights.gate, w, w, rows, m.n, g, vk.K2None, vk.K2Store)
	norm(q, q, ditHeadDim, rows*txtHeads, b.qNorm)
	norm(k, k, ditHeadDim, rows*txtHeads, b.kNorm)
	m.k.Attn(r, vk.K2Attn{Q: q, QStride: w, QSeq: length * w, K: k, KStride: w, KSeq: length * w, V: v, VStride: w, VSeq: length * w,
		O: att, OStride: w, OSeq: length * w, Queries: length, Keys: length, Group: 1, Scale: float32(1 / math.Sqrt(ditHeadDim)),
		Heads: txtHeads, Seqs: int(seqs), HeadDim: ditHeadDim})
	m.k.Act(r, vk.K2Act{X: att, XStride: w, Y: g, YStride: w, N: w, Rows: rows, Op: vk.K2Sigmoid})
	m.mm(r, b.o, w, w, rows, att, m.x, vk.K2None, vk.K2Add)
	norm(m.x, m.n, w, rows, b.post)
	h1 := m.scratch
	h2 := h1 + rows*txtFFN
	m.mm(r, b.gate, txtFFN, w, rows, m.n, h1, vk.K2None, vk.K2Store)
	m.mm(r, b.up, txtFFN, w, rows, m.n, h2, vk.K2None, vk.K2Store)
	m.k.Act(r, vk.K2Act{X: h1, XStride: txtFFN, Y: h2, YStride: txtFFN, N: txtFFN, Rows: rows, Op: vk.K2SwiGLU})
	m.mm(r, b.down, w, txtFFN, rows, h1, m.x, vk.K2None, vk.K2Add)
}

// SetText runs the text's part of the DiT on a prompt's conditioning, seq
// tokens of CondWidth, and keeps the answer on the card for Step. It returns
// what the text fusion made of it and the text MLP's answer, for tests.
func (m *DiT) SetText(cond []float32, seq int) error {
	if seq < 1 || seq > MaxPromptTokens || len(cond) != seq*CondWidth {
		return fmt.Errorf("krea2: %d floats of conditioning for %d tokens", len(cond), seq)
	}
	progs, ok := m.fusion[seq]
	if !ok {
		S := uint32(seq)
		a, err := m.k.Compile(func(r *vk.Recorder) {
			for i := range m.layerwise {
				m.textBlock(r, &m.layerwise[i], S, txtLayers)
			}
		})
		if err != nil {
			return err
		}
		b, err := m.k.Compile(func(r *vk.Recorder) {
			for i := range m.refiner {
				m.textBlock(r, &m.refiner[i], 1, S)
			}
			m.k.Norm(r, vk.K2Norm{Src: m.x, SrcStride: txtWidth, Dst: m.n, DstStride: txtWidth, N: txtWidth, Rows: S,
				Weight: m.txtNorm, OnePlus: 1, Eps: ditEps, ScaleA: vk.K2None})
			m.mm(r, m.txt1, ditWidth, txtWidth, S, m.n, m.scratch, m.txt1B, vk.K2Store)
			m.k.Act(r, vk.K2Act{X: m.scratch, XStride: ditWidth, N: ditWidth, Rows: S, Op: vk.K2Gelu})
			m.mm(r, m.txt3, ditWidth, ditWidth, S, m.scratch, m.ctx, m.txt3B, vk.K2Store)
		})
		if err != nil {
			a.Close()
			return err
		}
		progs = [2]*vk.Program{a, b}
		m.fusion[seq] = progs
	}
	if err := m.k.Write(int(m.x), cond); err != nil {
		return err
	}
	if err := progs[0].Run(); err != nil {
		return err
	}
	// The projector folds the twelve layers of a token into one: a weighted
	// sum, done here between the two programs.
	layers, err := m.k.Read(int(m.x), seq*txtLayers*txtWidth)
	if err != nil {
		return err
	}
	folded := make([]float32, seq*txtWidth)
	for s := 0; s < seq; s++ {
		for d := 0; d < txtWidth; d++ {
			var sum float32
			for l := 0; l < txtLayers; l++ {
				sum += m.projector[l] * layers[(s*txtLayers+l)*txtWidth+d]
			}
			folded[s*txtWidth+d] = sum
		}
	}
	if err := m.k.Write(int(m.x), folded); err != nil {
		return err
	}
	if err := progs[1].Run(); err != nil {
		return err
	}
	m.ctxLen = seq
	return nil
}

// Text returns the text MLP's answer from the last SetText, for tests.
func (m *DiT) Text() ([]float32, error) { return m.k.Read(int(m.ctx), m.ctxLen*ditWidth) }

// Fused returns the text fusion's answer from the last SetText, for tests:
// what the text MLP was given, before its norm.
func (m *DiT) Fused() ([]float32, error) { return m.k.Read(int(m.x), m.ctxLen*txtWidth) }

// Step is one call of the DiT on a latent of 16 × h × w (h and w even), at
// noise level sigma, with the text of the last SetText. It returns the
// velocity, the same shape as the latent.
func (m *DiT) Step(latent []float32, h, w int, sigma float32) ([]float32, error) {
	if m.ctxLen == 0 {
		return nil, fmt.Errorf("krea2: a DiT step before any text")
	}
	if h%ditPatch != 0 || w%ditPatch != 0 || len(latent) != ditLatent*h*w {
		return nil, fmt.Errorf("krea2: a latent of %d floats for 16 × %d × %d", len(latent), h, w)
	}
	img := (h / ditPatch) * (w / ditPatch)
	if img > MaxImageTokens {
		return nil, fmt.Errorf("krea2: %d patches is past the %d the DiT holds", img, MaxImageTokens)
	}
	prog, err := m.stepProgram(m.ctxLen, img)
	if err != nil {
		return nil, err
	}
	if err := m.k.Write(int(m.img), patchify(latent, h, w)); err != nil {
		return nil, err
	}
	if err := m.k.Write(int(m.temb), timestepEmbedding(sigma)); err != nil {
		return nil, err
	}
	if err := m.k.Write(int(m.table), ditRope(m.ctxLen, h/ditPatch, w/ditPatch)); err != nil {
		return nil, err
	}
	if err := prog.Run(); err != nil {
		return nil, err
	}
	out, err := m.k.Read(int(m.out), img*ditIn)
	if err != nil {
		return nil, err
	}
	return unpatchify(out, h, w), nil
}

func (m *DiT) stepProgram(txt, img int) (*vk.Program, error) {
	key := [2]int{txt, img}
	if p, ok := m.steps[key]; ok {
		return p, nil
	}
	T, I := uint32(txt), uint32(img)
	N := T + I
	const F = ditWidth
	kv := uint32(ditKVHeads * ditHeadDim)
	p, err := m.k.Compile(func(r *vk.Recorder) {
		// The timestep: tmlp, then tproj over it.
		m.mm(r, m.tmlp0, F, ditTDim, 1, m.temb, m.tA, m.tmlp0B, vk.K2Store)
		m.k.Act(r, vk.K2Act{X: m.tA, XStride: F, N: F, Rows: 1, Op: vk.K2Gelu})
		m.mm(r, m.tmlp2, F, F, 1, m.tA, m.t, m.tmlp2B, vk.K2Store)
		m.k.Act(r, vk.K2Act{X: m.tg, XStride: F, Y: m.t, YStride: F, N: F, Rows: 1, Op: vk.K2Copy})
		m.k.Act(r, vk.K2Act{X: m.tg, XStride: F, N: F, Rows: 1, Op: vk.K2Gelu})
		m.mm(r, m.tproj, 6*F, F, 1, m.tg, m.tvec, m.tprojB, vk.K2Store)

		// The stream: the text, then the patches.
		m.k.Act(r, vk.K2Act{X: m.x, XStride: F, Y: m.ctx, YStride: F, N: F, Rows: T, Op: vk.K2Copy})
		m.mm(r, m.first, F, ditIn, I, m.img, m.x+T*F, m.firstB, vk.K2Store)

		q := m.scratch
		k := q + N*F
		v := k + N*kv
		g := v + N*kv
		att := g + N*F
		h1 := m.scratch
		h2 := h1 + N*ditFFN
		for i := range m.blocks {
			b := &m.blocks[i]
			mod := func(j uint32) (uint32, uint32) { return m.tvec + j*F, b.mod + j*F }
			sa, sb := mod(0)
			ha, hb := mod(1)
			m.k.Norm(r, vk.K2Norm{Src: m.x, SrcStride: F, Dst: m.n, DstStride: F, N: F, Rows: N, Weight: b.pre, OnePlus: 1, Eps: ditEps,
				ScaleA: sa, ScaleB: sb, ShiftA: ha, ShiftB: hb})
			m.mm(r, b.q, F, F, N, m.n, q, vk.K2None, vk.K2Store)
			m.mm(r, b.k, kv, F, N, m.n, k, vk.K2None, vk.K2Store)
			m.mm(r, b.v, kv, F, N, m.n, v, vk.K2None, vk.K2Store)
			m.mm(r, b.attnWeights.gate, F, F, N, m.n, g, vk.K2None, vk.K2Store)
			m.k.Norm(r, vk.K2Norm{Src: q, SrcStride: ditHeadDim, Dst: q, DstStride: ditHeadDim, N: ditHeadDim, Rows: N * ditHeads,
				Weight: b.qNorm, OnePlus: 1, Eps: ditEps, ScaleA: vk.K2None})
			m.k.Norm(r, vk.K2Norm{Src: k, SrcStride: ditHeadDim, Dst: k, DstStride: ditHeadDim, N: ditHeadDim, Rows: N * ditKVHeads,
				Weight: b.kNorm, OnePlus: 1, Eps: ditEps, ScaleA: vk.K2None})
			m.k.Rope(r, vk.K2Rope{X: q, Stride: F, Heads: ditHeads, Dim: ditHeadDim, Tokens: N, Table: m.table})
			m.k.Rope(r, vk.K2Rope{X: k, Stride: kv, Heads: ditKVHeads, Dim: ditHeadDim, Tokens: N, Table: m.table})
			m.k.Attn(r, vk.K2Attn{Q: q, QStride: F, K: k, KStride: kv, V: v, VStride: kv, O: att, OStride: F,
				Queries: N, Keys: N, Group: ditHeads / ditKVHeads, Scale: float32(1 / math.Sqrt(ditHeadDim)),
				Heads: ditHeads, Seqs: 1, HeadDim: ditHeadDim})
			m.k.Act(r, vk.K2Act{X: att, XStride: F, Y: g, YStride: F, N: F, Rows: N, Op: vk.K2Sigmoid})
			ga, gb := mod(2)
			m.gatedMM(r, b.o, F, F, N, att, m.x, ga, gb)

			sa, sb = mod(3)
			ha, hb = mod(4)
			m.k.Norm(r, vk.K2Norm{Src: m.x, SrcStride: F, Dst: m.n, DstStride: F, N: F, Rows: N, Weight: b.post, OnePlus: 1, Eps: ditEps,
				ScaleA: sa, ScaleB: sb, ShiftA: ha, ShiftB: hb})
			m.mm(r, b.gate, ditFFN, F, N, m.n, h1, vk.K2None, vk.K2Store)
			m.mm(r, b.up, ditFFN, F, N, m.n, h2, vk.K2None, vk.K2Store)
			m.k.Act(r, vk.K2Act{X: h1, XStride: ditFFN, Y: h2, YStride: ditFFN, N: ditFFN, Rows: N, Op: vk.K2SwiGLU})
			ga, gb = mod(5)
			m.gatedMM(r, b.down, F, ditFFN, N, h1, m.x, ga, gb)
		}
		// The last layer, on the patches alone: its modulation is the
		// timestep's t plus its own two vectors.
		m.k.Norm(r, vk.K2Norm{Src: m.x + T*F, SrcStride: F, Dst: m.n, DstStride: F, N: F, Rows: I, Weight: m.lastNorm, OnePlus: 1, Eps: ditEps,
			ScaleA: m.t, ScaleB: m.lastMod, ShiftA: m.t, ShiftB: m.lastMod + F})
		m.mm(r, m.lastW, ditIn, F, I, m.n, m.out, m.lastB, vk.K2Store)
	})
	if err != nil {
		return nil, err
	}
	m.steps[key] = p
	return p, nil
}

// patchify is process_img: a 16 × h × w latent to (h/2)·(w/2) patches of
// 64, each channel's 2 × 2 in reading order.
func patchify(latent []float32, h, w int) []float32 {
	ph, pw := h/ditPatch, w/ditPatch
	out := make([]float32, ph*pw*ditIn)
	for y := 0; y < ph; y++ {
		for x := 0; x < pw; x++ {
			at := (y*pw + x) * ditIn
			for c := 0; c < ditLatent; c++ {
				for dy := 0; dy < ditPatch; dy++ {
					for dx := 0; dx < ditPatch; dx++ {
						out[at+c*4+dy*2+dx] = latent[c*h*w+(y*2+dy)*w+x*2+dx]
					}
				}
			}
		}
	}
	return out
}

func unpatchify(p []float32, h, w int) []float32 {
	ph, pw := h/ditPatch, w/ditPatch
	out := make([]float32, ditLatent*h*w)
	for y := 0; y < ph; y++ {
		for x := 0; x < pw; x++ {
			at := (y*pw + x) * ditIn
			for c := 0; c < ditLatent; c++ {
				for dy := 0; dy < ditPatch; dy++ {
					for dx := 0; dx < ditPatch; dx++ {
						out[c*h*w+(y*2+dy)*w+x*2+dx] = p[at+c*4+dy*2+dx]
					}
				}
			}
		}
	}
	return out
}

// timestepEmbedding is Flux's timestep_embedding(sigma, 256): the time
// scaled by a thousand, cosines then sines, frequencies in float32.
func timestepEmbedding(sigma float32) []float32 {
	half := ditTDim / 2
	t := 1000 * sigma
	out := make([]float32, ditTDim)
	for i := 0; i < half; i++ {
		f := float32(math.Exp(float64(float32(-math.Log(10000)) * float32(i) / float32(half))))
		a := float64(t * f)
		out[i] = float32(math.Cos(a))
		out[half+i] = float32(math.Sin(a))
	}
	return out
}

// ditRope is the rotation of every token: the text at position (0, 0, 0),
// patch (y, x) at (0, y, x), each axis turning its share of a head's pairs
// by EmbedND's frequencies, computed in float64.
func ditRope(txt, ph, pw int) []float32 {
	n := txt + ph*pw
	half := ditHeadDim / 2
	out := make([]float32, n*ditHeadDim)
	for tok := 0; tok < n; tok++ {
		pos := [3]float64{}
		if tok >= txt {
			i := tok - txt
			pos[1], pos[2] = float64(i/pw), float64(i%pw)
		}
		pair := 0
		for axis, dim := range ropeAxes {
			for i := 0; i < dim/2; i++ {
				omega := 1 / math.Pow(ditTheta, float64(2*i)/float64(dim))
				a := pos[axis] * omega
				out[tok*ditHeadDim+pair] = float32(math.Cos(a))
				out[tok*ditHeadDim+half+pair] = float32(math.Sin(a))
				pair++
			}
		}
	}
	return out
}

// WeightBytes is what the DiT holds on the card.
func (m *DiT) WeightBytes() int { return m.k.WeightBytes() }

// Close frees the card and the checkpoint.
func (m *DiT) Close() {
	for _, p := range m.fusion {
		p[0].Close()
		p[1].Close()
	}
	m.fusion = nil
	for _, p := range m.steps {
		p.Close()
	}
	m.steps = nil
	if m.k != nil {
		m.k.Close()
		m.k = nil
	}
	if m.ck != nil {
		m.ck.close()
		m.ck = nil
	}
}
