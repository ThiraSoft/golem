package yue2

// The autoregressive half on the card: the AR weight set of the 28 blocks,
// which writes the score and then the semantic codes a token at a time into a
// key and value cache the NAR half reads after it.
//
// The block is Qwen3's, so it runs on the stack golem's Qwen3 and Gemma run
// on (vk/stack.go), assembled the way qwen/vulkan.go assembles it: an
// ordinary pre-norm block, a SiLU gate, the scores scaled by one over the
// square root of the head, the NeoX rotation, an fp16 cache. yue2.cpp's
// src/qwen3-lm.h is the specification. What differs from qwen is around the
// stack rather than in it:
//
//   - The head is untied and only ever read on a window of rows: the score
//     samples rows [0, 151849), the codes rows [151852, 184621). Each window
//     is uploaded as a head of its own, and the two do not overlap, so the
//     card holds what one whole head would and reads only what a phase asks.
//     It reads the hidden state in float32, as yue2.cpp's does: see
//     vk.Yue2Head for what eight bits there cost.
//   - The embedding is looked up on the host, one Q8_0 row of 2048 a token,
//     and written into the stream: the stack's own lookups do not read Q8_0,
//     and a row is microseconds against a token's milliseconds (see
//     BenchmarkARToken).
//   - The cache is cut into slots: slot 0 is the conditional stream and slot
//     1 the unconditional one when the guidance is not one, decoded side by
//     side in one pass.
//   - A token reads its keys through the split attention of vk/yue2_ar.go
//     (UseSplitDecode): attn_one's workgroup a head walked the three thousand
//     keys of the prompt on sixteen workgroups, 12.4 of a token's 15.4 ms.
//     A pass of a prompt reads them through yue2_attn_wide.comp, the same
//     call's, with its mix on the matrix cores (BenchmarkARPrefill).

import (
	"encoding/binary"
	"fmt"
	"math"

	"github.com/ThiraSoft/golem/nn"
	"github.com/ThiraSoft/golem/vk"
)

// arNarrowColumns is the widest pass a prefill takes on a card without
// matrix cores. There the Q8_0 products have no tiled form
// (vk/quantproduct.go), and a pass wider than eight columns takes the tiled
// ladder (vk/attention.go passWidth), so a prompt is read eight positions a
// pass. With matrix cores a pass is as wide as the stack carries.
const arNarrowColumns = 8

// AR is the autoregressive half on one device.
type AR struct {
	cfg     Config
	stack   *vk.Stack
	heads   [2]*vk.Yue2Head // by Phase
	embed   Matrix
	context int
	pos     []int
	// width is the widest pass a prefill takes.
	width int

	// outs are the logits, one buffer a slot, which Prefill and Step hand
	// back and overwrite.
	outs [][]float32
}

// NewAR puts b's AR half on d with a cache of context positions for each of
// slots streams. Only a Q8_0 backbone is read for now. b must stay open while
// the AR runs: the embedding is read from its mapping.
func NewAR(d *vk.Device, b *Backbone, context, slots int) (*AR, error) {
	cfg := b.Cfg
	if context < 1 || context > cfg.Context || slots < 1 {
		return nil, fmt.Errorf("yue2: an AR cache of %d positions for %d slots (the model holds %d)", context, slots, cfg.Context)
	}
	if cfg.Heads*cfg.HeadDim != cfg.Dim || cfg.Heads%cfg.KVHeads != 0 {
		return nil, fmt.Errorf("yue2: AR of %d heads of %d over %d", cfg.Heads, cfg.HeadDim, cfg.Dim)
	}
	for _, m := range []struct {
		what string
		q    nn.Quant
	}{{"the embedding", b.Embed.Quant}, {"the head", b.Head.Quant}} {
		if m.q != nn.Q8_0 {
			return nil, fmt.Errorf("yue2: the AR half on the card reads Q8_0, and %s is %s", m.what, m.q)
		}
	}
	for i, bl := range b.AR {
		for _, q := range []nn.Quant{bl.Q.Quant, bl.K.Quant, bl.V.Quant, bl.O.Quant, bl.Gate.Quant, bl.Up.Quant, bl.Down.Quant} {
			if q != nn.Q8_0 {
				return nil, fmt.Errorf("yue2: the AR half on the card reads Q8_0, and block %d has a %s matrix", i, q)
			}
		}
	}
	if b.Embed.Cols != cfg.Dim || b.Head.Cols != cfg.Dim {
		return nil, fmt.Errorf("yue2: an embedding of %d and a head of %d over a stream of %d", b.Embed.Cols, b.Head.Cols, cfg.Dim)
	}

	a := &AR{cfg: cfg, embed: b.Embed, context: context, pos: make([]int, slots), outs: make([][]float32, slots),
		width: arNarrowColumns}
	fail := func(err error) (*AR, error) {
		a.Close()
		return nil, err
	}

	qDim, kvDim := cfg.Heads*cfg.HeadDim, cfg.KVHeads*cfg.HeadDim
	attn, err := vk.NewAttention(d, cfg.Dim, qDim, kvDim, cfg.Heads, context, slots, 1)
	if err != nil {
		return nil, err
	}
	mix, err := vk.NewMixture(d, cfg.Dim, 0, cfg.FFN, 0, 0, vk.SiLU)
	if err != nil {
		attn.Close()
		return nil, err
	}
	if a.stack, err = vk.NewStack(d, cfg.Dim, cfg.Eps, attn, mix); err != nil {
		attn.Close()
		mix.Close()
		return nil, err
	}
	if err := a.stack.SetGeometry(0, cfg.HeadDim, cfg.RoPEBase, nil); err != nil {
		return fail(err)
	}
	for i := range b.AR {
		bl := &b.AR[i]
		shape := vk.BlockShape{
			Heads: cfg.Heads, KVHeads: cfg.KVHeads, HeadDim: cfg.HeadDim,
			RoPEDims: cfg.HeadDim, Capacity: context, OwnsKV: true, Eps: cfg.Eps,
			Formats: vk.BlockFormats{Q: nn.Q8_0, K: nn.Q8_0, V: nn.Q8_0, O: nn.Q8_0},
			// ggml_flash_attn_ext's scale, as qw3lm_build_attn passes it.
			Scale: float32(1 / math.Sqrt(float64(cfg.HeadDim))),
		}
		if err := attn.AddBlock(shape, bl.Q.Data, bl.K.Data, bl.V.Data, bl.O.Data, bl.QNorm, bl.KNorm); err != nil {
			return fail(err)
		}
		if err := mix.AddBlock(vk.MixtureFormats{GateUp: nn.Q8_0, Down: nn.Q8_0}, nil, nil, bl.Gate.Data, bl.Up.Data, bl.Down.Data); err != nil {
			return fail(err)
		}
		if err := a.stack.AddBlock(vk.BlockNorms{Attn: bl.AttnNorm, FFN: bl.FFNNorm, OutScale: 1, Layout: vk.LayoutPreNorm}); err != nil {
			return fail(err)
		}
	}
	// A token reads its whole prefix: attn_one's workgroup a head leaves the
	// card three quarters idle at three thousand keys (BenchmarkARToken).
	if err := attn.UseSplitDecode(); err != nil {
		return fail(err)
	}
	if err := a.stack.SetOutputNorm(b.Norm); err != nil {
		return fail(err)
	}
	if err := a.stack.Ready(); err != nil {
		return fail(err)
	}
	if d.Coopmat() {
		a.width = a.stack.Columns()
	}

	rowBytes := cfg.Dim / nn.QuantBlock * 34
	for _, p := range []Phase{PhaseABC, PhaseSemantic} {
		row0, rows := PhaseRows(p)
		if row0+rows > b.Head.Rows {
			return fail(fmt.Errorf("yue2: the head has %d rows and the phase reads up to %d", b.Head.Rows, row0+rows))
		}
		h, err := vk.NewYue2Head(d, b.Head.Data[row0*rowBytes:(row0+rows)*rowBytes], rows, cfg.Dim, slots)
		if err != nil {
			return fail(err)
		}
		a.heads[p] = h
	}
	_, abcRows := PhaseRows(PhaseABC)
	_, semRows := PhaseRows(PhaseSemantic)
	for s := range a.outs {
		a.outs[s] = make([]float32, max(abcRows, semRows))
	}
	return a, nil
}

// Pos is how many positions slot holds.
func (a *AR) Pos(slot int) int { return a.pos[slot] }

// Trim keeps the first n positions of slot; the next token goes at n.
func (a *AR) Trim(slot, n int) {
	if n < 0 || n > a.pos[slot] {
		panic(fmt.Sprintf("yue2: trim slot %d to %d of %d", slot, n, a.pos[slot]))
	}
	a.pos[slot] = n
}

// Cache is layer's keys and values on the card: slots rings of capacity
// positions laid end to end, a position [KVHeads][HeadDim] fp16.
func (a *AR) Cache(layer int) (k, v *vk.Buffer, capacity int) {
	k, v, capacity, err := a.stack.Cache(layer)
	if err != nil {
		panic(fmt.Sprintf("yue2: %v", err))
	}
	return k, v, capacity
}

// Close gives the card back.
func (a *AR) Close() {
	for i, h := range a.heads {
		if h != nil {
			h.Close()
			a.heads[i] = nil
		}
	}
	if a.stack != nil {
		a.stack.Close()
		a.stack = nil
	}
}

// embedRow writes token id's row of the embedding into dst, dequantized the
// way ggml_get_rows does: the int8 times the block's fp16 scale, in float32.
func (a *AR) embedRow(id int32, dst []float32) error {
	if id < 0 || int(id) >= a.embed.Rows {
		return fmt.Errorf("yue2: token %d outside a vocabulary of %d", id, a.embed.Rows)
	}
	cols := a.cfg.Dim
	rowBytes := cols / nn.QuantBlock * 34
	src := a.embed.Data[int(id)*rowBytes : (int(id)+1)*rowBytes]
	for b := 0; b < cols/nn.QuantBlock; b++ {
		blk := src[34*b:]
		d := nn.Widen(binary.LittleEndian.Uint16(blk))
		for i := range nn.QuantBlock {
			dst[nn.QuantBlock*b+i] = float32(int8(blk[2+i])) * d
		}
	}
	return nil
}

// run carries one pass: column c is token ids[c] of slot slots[c] at the
// slot's next position. The columns of one slot are contiguous, as the stack
// wants. The positions advance.
func (a *AR) run(ids []int32, slots []int) error {
	dim := a.cfg.Dim
	stream := a.stack.Stream().Floats()
	at := make([]vk.Position, len(ids))
	next := map[int]int{}
	for c, id := range ids {
		s := slots[c]
		if s < 0 || s >= len(a.pos) {
			return fmt.Errorf("yue2: slot %d of %d", s, len(a.pos))
		}
		if c > 0 && slots[c-1] != s {
			if _, seen := next[s]; seen {
				return fmt.Errorf("yue2: slot %d's columns are not contiguous", s)
			}
		}
		p, ok := next[s]
		if !ok {
			p = a.pos[s]
		}
		if p >= a.context {
			return fmt.Errorf("yue2: slot %d is full at %d positions", s, a.context)
		}
		next[s] = p + 1
		if err := a.embedRow(id, stream[c*dim:(c+1)*dim]); err != nil {
			return err
		}
		first, last := make([]int, a.cfg.Layers), make([]int, a.cfg.Layers)
		for l := range last {
			last[l] = p
		}
		at[c] = vk.Position{Slot: s, Pos: p, First: first, Last: last}
	}
	if err := a.stack.Run(at, 0, 0); err != nil {
		return err
	}
	for s, p := range next {
		a.pos[s] = p
	}
	return nil
}

// logits reads the head of phase p over the normed hidden states of the
// pass's columns cols, into the slots' buffers.
func (a *AR) logits(p Phase, cols, slots []int) ([][]float32, error) {
	dim := a.cfg.Dim
	_, rows := PhaseRows(p)
	stream := a.stack.Stream().Floats()
	xs, outs := make([][]float32, len(cols)), make([][]float32, len(cols))
	for i, c := range cols {
		xs[i] = stream[c*dim : (c+1)*dim]
		outs[i] = a.outs[slots[i]][:rows]
	}
	if err := a.heads[p].Logits(xs, outs); err != nil {
		return nil, err
	}
	return outs, nil
}

// Prefill reads ids into slot behind what it holds, and returns the logits
// of the last position on phase p's rows (PhaseRows). The slice is the
// slot's and the next Prefill or Step of that slot overwrites it.
func (a *AR) Prefill(slot int, ids []int32, p Phase) ([]float32, error) {
	if len(ids) == 0 {
		return nil, fmt.Errorf("yue2: an empty prefill")
	}
	if slot < 0 || slot >= len(a.pos) {
		return nil, fmt.Errorf("yue2: slot %d of %d", slot, len(a.pos))
	}
	if a.pos[slot]+len(ids) > a.context {
		return nil, fmt.Errorf("yue2: %d positions behind %d in a cache of %d", len(ids), a.pos[slot], a.context)
	}
	width := a.width
	slots := make([]int, min(width, len(ids)))
	for i := range slots {
		slots[i] = slot
	}
	n := 0
	for at := 0; at < len(ids); at += width {
		chunk := ids[at:min(at+width, len(ids))]
		if err := a.run(chunk, slots[:len(chunk)]); err != nil {
			return nil, err
		}
		n = len(chunk)
	}
	outs, err := a.logits(p, []int{n - 1}, []int{slot})
	if err != nil {
		return nil, err
	}
	return outs[0], nil
}

// Step decodes one token for each listed slot in one pass and returns their
// logits on phase p's rows, in the order given. Each slice is its slot's,
// overwritten by the slot's next Prefill or Step.
func (a *AR) Step(tokens []int32, slots []int, p Phase) ([][]float32, error) {
	if len(tokens) == 0 || len(tokens) != len(slots) {
		return nil, fmt.Errorf("yue2: %d tokens for %d slots", len(tokens), len(slots))
	}
	seen := map[int]bool{}
	for _, s := range slots {
		if seen[s] {
			return nil, fmt.Errorf("yue2: slot %d twice in one step", s)
		}
		seen[s] = true
	}
	if err := a.run(tokens, slots); err != nil {
		return nil, err
	}
	cols := make([]int, len(tokens))
	for i := range cols {
		cols[i] = i
	}
	return a.logits(p, cols, slots)
}
