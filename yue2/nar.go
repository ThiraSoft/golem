package yue2

// The flow matching half on the card: the NAR weight set of the 28 blocks,
// which reads the AR half's keys and values and predicts the velocity of 64
// latent channels a frame. It runs on the Krea 2 machine the way flux3's DiT
// does: one float32 arena, token-major, one row of 2048 a token; the weights
// fp16 one buffer per tensor, dequantized on the host at load; one program
// recorded per shape and replayed for every evaluation of the ODE.
//
// yue2.cpp's src/nar.h is the specification. The block is [zero row, the
// frames, zero row], N = frames + 2 tokens at positions arLen .. arLen + N - 1,
// which continue the AR prefix's; every query sees the arLen prefix keys and
// the N fresh ones, unmasked. The prefix lives in a cache beside the arena
// (vk.K2Yue2Cache) in the layout the attention reads, written once by
// SetPrefix; each evaluation writes the block's own keys behind it.

import (
	"encoding/binary"
	"fmt"
	"math"
	"runtime"
	"sync"
	"unsafe"

	"github.com/ThiraSoft/golem/nn"
	"github.com/ThiraSoft/golem/vk"
)

// narTimeDim is the width of the timestep's sinusoidal features.
const narTimeDim = 256

type narLayer struct {
	qkv, o, gateUp, down int
	attnNorm, ffnNorm    uint32
	qNorm, kNorm         uint32
}

// NAR is the flow matching half on one device.
type NAR struct {
	cfg    Config
	k      *vk.K2
	cache  *vk.K2Yue2Cache
	layers []narLayer

	vae2llm, llm2vae, time0, time1 int
	vae2llmB, llm2vaeB             uint32
	time0B, time1B, norm           uint32

	maxFrames, maxPrefix int
	arLen                int

	// The arena, in floats: the latent block, the timestep's features and
	// its embedding, the frame embedding and the rotation table of the
	// current shape, the stream x, the norms' fp16 answer, the fused q|k|v
	// product, the attention's answer, the MLP's fp16 inner activation, the
	// velocity.
	xin, feats, tA, temb uint32
	pos, table           uint32
	x, nh, qkv, att, ffn uint32
	out, state           uint32
	floats               int

	progs map[int]*vk.Program
	// tables is the frame count the frame embedding and the rotation table
	// in the arena were written for; 0 for none.
	tables int

	// timeline, when set, gets a stamp after every dispatch: what the
	// benchmark reads.
	timeline *vk.Timeline
}

// NewNAR puts b's NAR half on d, with an arena for up to maxFrames frames and
// a cache for a prefix of up to maxPrefix keys.
func NewNAR(d *vk.Device, b *Backbone, maxFrames, maxPrefix int) (*NAR, error) {
	return newNAR(d, b, maxFrames, maxPrefix, maxPrefix+maxFrames+2)
}

// newNAR is NewNAR with a cache of keys rows, which a prefix and the block
// behind it share: the pipeline's chunks are long prefixes with short blocks
// or the other way round, never both, and the context bounds their sum.
func newNAR(d *vk.Device, b *Backbone, maxFrames, maxPrefix, keys int) (*NAR, error) {
	cfg := b.Cfg
	if cfg.HeadDim != 128 || cfg.Heads*cfg.HeadDim != cfg.Dim || cfg.Heads%cfg.KVHeads != 0 {
		return nil, fmt.Errorf("yue2: NAR of %d heads of %d over %d, which the attention does not take", cfg.Heads, cfg.HeadDim, cfg.Dim)
	}
	if maxFrames < 1 || maxPrefix < 1 {
		return nil, fmt.Errorf("yue2: NAR of %d frames over %d keys", maxFrames, maxPrefix)
	}
	n := &NAR{cfg: cfg, maxFrames: maxFrames, maxPrefix: maxPrefix, progs: map[int]*vk.Program{},
		layers: make([]narLayer, cfg.Layers)}

	maxN := maxFrames + 2
	dim, kvDim, ffn := cfg.Dim, cfg.KVHeads*cfg.HeadDim, cfg.FFN
	at := 0
	room := func(floats int) uint32 {
		o := at
		at += (floats + 63) / 64 * 64
		return uint32(o)
	}
	n.xin = room(cfg.LatentDim * maxN)
	n.feats = room(narTimeDim)
	n.tA = room(dim)
	n.temb = room(dim)
	n.pos = room(dim * maxN)
	n.table = room(cfg.HeadDim * maxN)
	n.x = room(dim * maxN)
	n.nh = room(dim * maxN / 2)
	n.qkv = room((dim + 2*kvDim) * maxN)
	n.att = room(dim * maxN)
	n.ffn = room(ffn * maxN / 2)
	n.out = room(cfg.LatentDim * maxN)
	n.state = room(cfg.LatentDim * maxN)
	n.floats = at

	var params []float32
	param := func(p []float32) uint32 {
		o := uint32(len(params))
		params = append(params, p...)
		return o
	}
	for l := range n.layers {
		bl, nl := &b.NAR[l], &n.layers[l]
		nl.attnNorm, nl.ffnNorm = param(bl.AttnNorm), param(bl.FFNNorm)
		nl.qNorm, nl.kNorm = param(bl.QNorm), param(bl.KNorm)
	}
	n.norm = param(b.Norm)
	n.vae2llmB, n.llm2vaeB = param(b.Vae2LLMBias), param(b.LLM2VaeBias)
	n.time0B, n.time1B = param(b.Time0Bias), param(b.Time1Bias)

	k, err := vk.NewK2(d, n.floats, params)
	if err != nil {
		return nil, err
	}
	n.k = k
	fail := func(err error) (*NAR, error) {
		n.Close()
		return nil, fmt.Errorf("yue2: NAR weights: %w", err)
	}
	// Every weight goes to the card as fp16 rows, each widened from a row of
	// one of the file's matrices on its way (vk.K2.AddWeightRows): row r of
	// the tensor is row of src(r).
	q80 := q80Halves()
	upload := func(h *int, rows, cols int, src func(r int) (*Matrix, int)) error {
		var err error
		*h, err = k.AddWeightRows(rows, 2*cols, func(dst []byte, first, count int) {
			out := unsafe.Slice((*uint16)(unsafe.Pointer(&dst[0])), len(dst)/2)
			spread(count, func(lo, hi int) {
				for i := lo; i < hi; i++ {
					m, row := src(first + i)
					narrowRow(m, row, out[i*cols:(i+1)*cols], q80)
				}
			})
		})
		return err
	}
	single := func(h *int, m *Matrix) error {
		if err := widenable(*m, m.Cols); err != nil {
			return err
		}
		return upload(h, m.Rows, m.Cols, func(r int) (*Matrix, int) { return m, r })
	}
	for _, w := range []struct {
		h *int
		m *Matrix
	}{{&n.vae2llm, &b.Vae2LLM}, {&n.llm2vae, &b.LLM2Vae}, {&n.time0, &b.Time0}, {&n.time1, &b.Time1}} {
		if err := single(w.h, w.m); err != nil {
			return fail(err)
		}
	}
	// The gate's rows then the up's, in the runs of 32 the SwiGLU epilogue
	// wants: the card's row c is the source row swiGLU[c].
	swiGLU := make([]int, 2*ffn)
	for r := range swiGLU {
		swiGLU[vk.SwiGLURow(r, ffn)] = r
	}
	for l := range n.layers {
		bl, nl := &b.NAR[l], &n.layers[l]
		for _, m := range []Matrix{bl.Q, bl.K, bl.V, bl.Gate, bl.Up} {
			if err := widenable(m, dim); err != nil {
				return fail(err)
			}
		}
		if bl.Q.Rows != dim || bl.K.Rows != kvDim || bl.V.Rows != kvDim || bl.Gate.Rows != ffn || bl.Up.Rows != ffn {
			return fail(fmt.Errorf("block %d: q, k, v of %d, %d, %d rows and a gate and up of %d, %d", l,
				bl.Q.Rows, bl.K.Rows, bl.V.Rows, bl.Gate.Rows, bl.Up.Rows))
		}
		// q, k and v in one product, their rows stacked.
		err := upload(&nl.qkv, dim+2*kvDim, dim, func(r int) (*Matrix, int) {
			switch {
			case r < dim:
				return &bl.Q, r
			case r < dim+kvDim:
				return &bl.K, r - dim
			}
			return &bl.V, r - dim - kvDim
		})
		if err != nil {
			return fail(err)
		}
		if err := single(&nl.o, &bl.O); err != nil {
			return fail(err)
		}
		err = upload(&nl.gateUp, 2*ffn, dim, func(c int) (*Matrix, int) {
			r := swiGLU[c]
			if r < ffn {
				return &bl.Gate, r
			}
			return &bl.Up, r - ffn
		})
		if err != nil {
			return fail(err)
		}
		if err := single(&nl.down, &bl.Down); err != nil {
			return fail(err)
		}
	}
	if n.cache, err = k.NewYue2Cache(cfg.Layers, cfg.KVHeads, keys); err != nil {
		return fail(err)
	}
	return n, nil
}

// widenable says whether m is a matrix of cols columns narrowRow reads.
func widenable(m Matrix, cols int) error {
	var rowBytes int
	switch m.Quant {
	case nn.Q8_0:
		if m.Cols%32 != 0 {
			return fmt.Errorf("a Q8_0 matrix of %d columns", m.Cols)
		}
		rowBytes = m.Cols / 32 * 34
	case nn.BF16, nn.F16:
		rowBytes = 2 * m.Cols
	case nn.F32:
		rowBytes = 4 * m.Cols
	default:
		return fmt.Errorf("weights in %v", m.Quant)
	}
	if m.Cols != cols {
		return fmt.Errorf("a matrix of %d columns where %d are read", m.Cols, cols)
	}
	if len(m.Data) < rowBytes*m.Rows {
		return fmt.Errorf("%d bytes for %d rows of %d", len(m.Data), m.Rows, m.Cols)
	}
	return nil
}

// q80Halves is narrow(float32(q) * d) for every q of a Q8_0 block and every
// fp16 scale d of sign zero: entry s<<8 | byte(q) for the scale of bits s.
// Narrowing commutes with the sign, so a negative scale's entry is its
// magnitude's with the sign bit flipped, zeros included (q = 0 against a
// negative scale is minus zero in float32, and narrows to 0x8000), for every
// finite scale. Sixteen
// megabytes, built in a few milliseconds over the cores, against 1.4 billion
// products to narrow for the NAR half: a lookup and an exclusive or in place
// of the rounding. TestQ80RowsMatchNarrow holds narrowRow to narrow on
// every scale and byte.
func q80Halves() []uint16 {
	t := make([]uint16, 1<<15<<8)
	spread(1<<15, func(lo, hi int) {
		for s := lo; s < hi; s++ {
			d := nn.Widen(uint16(s))
			for q := range 256 {
				t[s<<8|q] = narrow(float32(int8(q)) * d)
			}
		}
	})
	return t
}

// narrowRow writes row r of m into dst in fp16: a Q8_0 row as each int8
// times its block's scale, rounded once (looked up in q80, q80Halves), the
// others rounded from their own precision. widenable has checked m.
func narrowRow(m *Matrix, r int, dst []uint16, q80 []uint16) {
	switch m.Quant {
	case nn.Q8_0:
		rowBytes := m.Cols / 32 * 34
		src := m.Data[r*rowBytes : (r+1)*rowBytes]
		for b := 0; b < m.Cols/32; b++ {
			blk := src[34*b : 34*b+34]
			s := binary.LittleEndian.Uint16(blk)
			q := blk[2:34]
			out := dst[32*b : 32*b+32]
			if s&0x7c00 == 0x7c00 {
				// An infinite or NaN scale, which no quantizer writes: zero
				// times infinity is a NaN whose sign the table's flip would
				// get wrong, so these round as they are.
				d := nn.Widen(s)
				for i := range out {
					out[i] = narrow(float32(int8(q[i])) * d)
				}
				continue
			}
			table := q80[int(s&0x7fff)<<8:][:256]
			sign := s & 0x8000
			for i := range out {
				out[i] = table[q[i]] ^ sign
			}
		}
	case nn.BF16:
		src := m.Data[r*2*m.Cols : (r+1)*2*m.Cols]
		for i := range dst {
			dst[i] = narrow(math.Float32frombits(uint32(binary.LittleEndian.Uint16(src[2*i:])) << 16))
		}
	case nn.F16:
		src := m.Data[r*2*m.Cols : (r+1)*2*m.Cols]
		for i := range dst {
			dst[i] = binary.LittleEndian.Uint16(src[2*i:])
		}
	case nn.F32:
		src := m.Data[r*4*m.Cols : (r+1)*4*m.Cols]
		for i := range dst {
			dst[i] = narrow(math.Float32frombits(binary.LittleEndian.Uint32(src[4*i:])))
		}
	}
}

// narrow is nn.FloatToHalf, rounding to the nearest fp16 with ties to even,
// in the form of Fabian Giesen's float_to_half_fast3_rtne: the exponent
// rebiased by an integer add and the rounding carried into it, with the
// subnormals rounded by the float adder. nn's version branches on every
// case, and widening the NAR half calls it 1.4 billion times.
// TestNarrowMatchesNN holds the two to the same bits on every product of an
// int8 and an fp16 scale.
func narrow(f float32) uint16 {
	u := math.Float32bits(f)
	sign := u & 0x80000000
	u ^= sign
	var o uint32
	switch {
	case u >= (127+16)<<23: // past fp16's range: infinity, or NaN kept quiet
		o = 0x7c00
		if u > 0x7f800000 {
			o = 0x7e00
		}
	case u < 113<<23: // an fp16 subnormal or zero, rounded by the adder
		const magic = ((127 - 15) + (23 - 10) + 1) << 23
		o = math.Float32bits(math.Float32frombits(u)+math.Float32frombits(magic)) - magic
	default:
		odd := u >> 13 & 1
		u -= (127 - 15) << 23
		u += 0xfff + odd
		o = u >> 13
	}
	return uint16(o | sign>>16)
}

// spread runs work over [0, n) cut into one contiguous range a core.
func spread(n int, work func(lo, hi int)) {
	workers := min(runtime.GOMAXPROCS(0), n)
	if workers <= 1 {
		work(0, n)
		return
	}
	var wg sync.WaitGroup
	per := (n + workers - 1) / workers
	for lo := 0; lo < n; lo += per {
		wg.Add(1)
		go func(lo, hi int) {
			defer wg.Done()
			work(lo, hi)
		}(lo, min(lo+per, n))
	}
	wg.Wait()
}

// WeightBytes is what the weights hold on the card; CacheBytes the prefix
// cache; ArenaBytes the arena.
func (n *NAR) WeightBytes() int { return n.k.WeightBytes() }
func (n *NAR) CacheBytes() int  { return n.cache.Bytes() }
func (n *NAR) ArenaBytes() int  { return 4 * n.floats }

// SetPrefix puts the AR half's keys and values on the card, every layer's
// arLen rows of [KV heads][128] fp16, as the AR cache holds them: k[l] and
// v[l] are layer l's, position-major. It holds until the next SetPrefix.
func (n *NAR) SetPrefix(k, v [][]uint16, arLen int) error {
	c := n.cfg
	if arLen < 1 || arLen > n.maxPrefix {
		return fmt.Errorf("yue2: a prefix of %d keys, past the %d the cache holds", arLen, n.maxPrefix)
	}
	if len(k) != c.Layers || len(v) != c.Layers {
		return fmt.Errorf("yue2: a prefix of %d and %d layers for %d", len(k), len(v), c.Layers)
	}
	width := c.KVHeads * c.HeadDim
	np := n.cache.Rows()
	blk := make([]uint16, 2*width*np)
	for l := range c.Layers {
		if len(k[l]) < arLen*width || len(v[l]) < arLen*width {
			return fmt.Errorf("yue2: layer %d's prefix is %d and %d halves, not %d", l, len(k[l]), len(v[l]), arLen*width)
		}
		// The keys [head][np][128], the values [head][128][np], the rows
		// past the prefix zero until an evaluation writes the block there.
		clear(blk)
		vt := blk[width*np:]
		for t := range arLen {
			for h := range c.KVHeads {
				src := t*width + h*c.HeadDim
				copy(blk[(h*np+t)*c.HeadDim:], k[l][src:src+c.HeadDim])
				for d := range c.HeadDim {
					vt[(h*c.HeadDim+d)*np+t] = v[l][src+d]
				}
			}
		}
		if err := n.cache.WriteLayer(l, blk); err != nil {
			return err
		}
	}
	if arLen != n.arLen {
		n.forget()
		n.arLen = arLen
	}
	return nil
}

// SetPrefixFromStack is SetPrefix from the AR half's own cache: the first
// arLen positions of ar's slot, every layer, copied on the card. ar and n
// must share a device. It holds until the next SetPrefix.
func (n *NAR) SetPrefixFromStack(ar *AR, slot, arLen int) error {
	c := n.cfg
	if arLen < 1 || arLen > n.maxPrefix {
		return fmt.Errorf("yue2: a prefix of %d keys, past the %d the cache holds", arLen, n.maxPrefix)
	}
	if slot < 0 || slot >= len(ar.pos) || arLen > ar.Pos(slot) {
		return fmt.Errorf("yue2: a prefix of %d keys from slot %d of the AR", arLen, slot)
	}
	if ar.cfg.KVHeads != c.KVHeads || ar.cfg.HeadDim != 128 || ar.cfg.Layers != c.Layers {
		return fmt.Errorf("yue2: an AR of %d layers of %d heads of %d for a NAR of %d layers of %d heads",
			ar.cfg.Layers, ar.cfg.KVHeads, ar.cfg.HeadDim, c.Layers, c.KVHeads)
	}
	ks, vs := make([]*vk.Buffer, c.Layers), make([]*vk.Buffer, c.Layers)
	var capacity int
	for l := range ks {
		ks[l], vs[l], capacity = ar.Cache(l)
	}
	// A slot's ring starts at slot·capacity positions, and the context never
	// wraps: Prefill and Step refuse a position past it.
	if err := n.cache.ImportStack(ks, vs, slot*capacity, arLen); err != nil {
		return err
	}
	if arLen != n.arLen {
		n.forget()
		n.arLen = arLen
	}
	return nil
}

// forget drops the programs and tables of the previous prefix length.
func (n *NAR) forget() {
	for f, p := range n.progs {
		p.Close()
		delete(n.progs, f)
	}
	n.tables = 0
}

// shiftT is the timestep of a raw one: its sigmoid, through the release's
// shift curve. Float32 step by step, as nar_shift_t.
func (n *NAR) shiftT(raw float32) float32 {
	s := n.cfg.TimestepShift
	sig := 1 / (1 + float32(math.Exp(float64(-raw))))
	return s * sig / (1 + (s-1)*sig)
}

// timeFeatures is nar_time_features: the cosines first, then the sines,
// frequencies exp(-ln(1e4)·i/128), in float32 the way the C++ computes them.
func timeFeatures(t float32) []float32 {
	out := make([]float32, narTimeDim)
	half := narTimeDim / 2
	ln := float32(math.Log(10000))
	for i := range half {
		freq := float32(math.Exp(float64(-ln * float32(i) / float32(half))))
		arg := t * freq
		out[i] = float32(math.Cos(float64(arg)))
		out[half+i] = float32(math.Sin(float64(arg)))
	}
	return out
}

// frameEmbedding is nar_pos_features: over the block's own frame index
// 0 .. tokens - 1, sin and cos interleaved over dim, in float32 as the C++.
func frameEmbedding(tokens, dim int) []float32 {
	out := make([]float32, tokens*dim)
	step := -float32(math.Log(10000)) / float32(dim)
	for i := 0; 2*i < dim; i++ {
		div := float32(math.Exp(float64(float32(2*i) * step)))
		for p := range tokens {
			arg := float32(p) * div
			out[p*dim+2*i] = float32(math.Sin(float64(arg)))
			out[p*dim+2*i+1] = float32(math.Cos(float64(arg)))
		}
	}
	return out
}

// ropeTable is a token's 64 cosines then 64 sines for positions first ..
// first + tokens - 1: ggml's angle pos·θ^i with θ = base^(-2/128), the power
// in float32 as its shader takes it.
func ropeTable(first, tokens int, base float64) []float32 {
	out := make([]float32, 128*tokens)
	scale := float32(math.Pow(base, -2.0/128))
	for t := range tokens {
		pos := float32(first + t)
		for i := range 64 {
			theta := pos * float32(math.Pow(float64(scale), float64(i)))
			out[128*t+i] = float32(math.Cos(float64(theta)))
			out[128*t+64+i] = float32(math.Sin(float64(theta)))
		}
	}
	return out
}

func (n *NAR) mm(r *vk.Recorder, w int, label string, m vk.K2MM) {
	m.Scale = 1
	n.k.MM(r, w, false, m)
	n.timeline.Stamp(r, label)
}

func (n *NAR) rmsNorm(r *vk.Recorder, src, weight uint32, tokens int) {
	d := uint32(n.cfg.Dim)
	n.k.Norm(r, vk.K2Norm{Src: src, SrcStride: d, Dst: 2 * n.nh, DstStride: d, N: d, Rows: uint32(tokens),
		Weight: weight, Eps: n.cfg.Eps, ScaleA: vk.K2None, ScaleB: vk.K2None, ShiftA: vk.K2None, ShiftB: vk.K2None,
		Halves: 1})
	n.timeline.Stamp(r, "norm")
}

func (n *NAR) act(r *vk.Recorder, a vk.K2Act) {
	n.k.Act(r, a)
	n.timeline.Stamp(r, "act")
}

// recordInput is the block's first hidden state: vae2llm of the latents,
// plus the timestep's embedding, plus the frame embedding.
func (n *NAR) recordInput(r *vk.Recorder, tokens int) {
	d, toks := uint32(n.cfg.Dim), uint32(tokens)
	n.mm(r, n.time0, "time", vk.K2MM{Outputs: d, Inputs: narTimeDim, Cols: 1, X: n.feats, XStride: narTimeDim,
		Y: n.tA, YStride: d, Bias: n.time0B, Mode: vk.K2Store})
	n.act(r, vk.K2Act{X: n.tA, XStride: d, N: d, Rows: 1, Op: vk.K2SiLU})
	n.mm(r, n.time1, "time", vk.K2MM{Outputs: d, Inputs: d, Cols: 1, X: n.tA, XStride: d,
		Y: n.temb, YStride: d, Bias: n.time1B, Mode: vk.K2Store})
	lat := uint32(n.cfg.LatentDim)
	n.mm(r, n.vae2llm, "input", vk.K2MM{Outputs: d, Inputs: lat, Cols: toks, X: n.xin, XStride: lat,
		Y: n.x, YStride: d, Bias: n.vae2llmB, Mode: vk.K2Store})
	n.act(r, vk.K2Act{X: n.x, XStride: d, Y: n.temb, YStride: 0, N: d, Rows: toks, Op: vk.K2Sum})
	n.act(r, vk.K2Act{X: n.x, XStride: d, Y: n.pos, YStride: d, N: d, Rows: toks, Op: vk.K2Sum})
}

// recordLayer is block l over the latent block. With sa set, the attention's
// output projection goes to the q|k|v room, which it has finished with, and
// is added to x after: the same sum, kept for the waypoint.
func (n *NAR) recordLayer(r *vk.Recorder, l, tokens int, sa bool) {
	c, bl := n.cfg, &n.layers[l]
	d, toks := uint32(c.Dim), uint32(tokens)
	kvDim := uint32(c.KVHeads * c.HeadDim)
	stride := d + 2*kvDim
	n.rmsNorm(r, n.x, bl.attnNorm, tokens)
	n.mm(r, bl.qkv, "mm qkv", vk.K2MM{Outputs: stride, Inputs: d, Cols: toks, X: 2 * n.nh, XStride: d,
		Y: n.qkv, YStride: stride, Bias: vk.K2None, Mode: vk.K2Store, XHalves: 1})
	n.k.Yue2QKRope(r, vk.K2Yue2QKRope{Q: n.qkv, K: n.qkv + d, Stride: stride,
		QHeads: uint32(c.Heads), KHeads: uint32(c.KVHeads), Tokens: toks, Table: n.table,
		QWeight: bl.qNorm, KWeight: bl.kNorm, Eps: c.Eps})
	n.timeline.Stamp(r, "qkrope")
	n.cache.Import(r, l, n.qkv+d, n.qkv+d+kvDim, stride, toks, uint32(n.arLen))
	n.timeline.Stamp(r, "kv import")
	n.cache.Attn(r, l, vk.K2Attn{Q: n.qkv, QStride: stride, O: n.att, OStride: d,
		Queries: toks, Keys: uint32(n.arLen) + toks, Group: uint32(c.Heads / c.KVHeads),
		Scale: float32(1 / math.Sqrt(float64(c.HeadDim))), Heads: c.Heads, Seqs: 1, HeadDim: c.HeadDim})
	n.timeline.Stamp(r, "attn")
	if sa {
		n.mm(r, bl.o, "mm o", vk.K2MM{Outputs: d, Inputs: d, Cols: toks, X: n.att, XStride: d,
			Y: n.qkv, YStride: d, Bias: vk.K2None, Mode: vk.K2Store})
		n.act(r, vk.K2Act{X: n.x, XStride: d, Y: n.qkv, YStride: d, N: d, Rows: toks, Op: vk.K2Sum})
	} else {
		n.mm(r, bl.o, "mm o", vk.K2MM{Outputs: d, Inputs: d, Cols: toks, X: n.att, XStride: d,
			Y: n.x, YStride: d, Bias: vk.K2None, Mode: vk.K2Add})
	}
	ffn := uint32(c.FFN)
	n.rmsNorm(r, n.x, bl.ffnNorm, tokens)
	n.mm(r, bl.gateUp, "mm ffn in", vk.K2MM{Outputs: 2 * ffn, Inputs: d, Cols: toks, X: 2 * n.nh, XStride: d,
		Y: 2 * n.ffn, YStride: ffn, Bias: vk.K2None, Mode: vk.K2SwiGLUHalves, XHalves: 1})
	n.mm(r, bl.down, "mm ffn out", vk.K2MM{Outputs: d, Inputs: ffn, Cols: toks, X: 2 * n.ffn, XStride: ffn,
		Y: n.x, YStride: d, Bias: vk.K2None, Mode: vk.K2Add, XHalves: 1})
}

// recordHead is the final norm and llm2vae, every token's velocity.
func (n *NAR) recordHead(r *vk.Recorder, tokens int) {
	d, lat := uint32(n.cfg.Dim), uint32(n.cfg.LatentDim)
	n.rmsNorm(r, n.x, n.norm, tokens)
	n.mm(r, n.llm2vae, "head", vk.K2MM{Outputs: lat, Inputs: d, Cols: uint32(tokens), X: 2 * n.nh, XStride: d,
		Y: n.out, YStride: lat, Bias: n.llm2vaeB, Mode: vk.K2Store, XHalves: 1})
}

// logitClamped is nar_logit_clamped: logit(t) saturated to ±20.
func logitClamped(t float32) float32 {
	if t >= 1.0 {
		return 20.0
	}
	if t <= 0.0 {
		return -20.0
	}
	v := float32(math.Log(float64(t / (1.0 - t))))
	if v > 20.0 {
		return 20.0
	}
	if v < -20.0 {
		return -20.0
	}
	return v
}

// prepareTables writes the frame embedding and the rotation table on a new shape.
func (n *NAR) prepareTables(frames int) error {
	if n.arLen == 0 {
		return fmt.Errorf("yue2: NAR called before SetPrefix")
	}
	if frames < 1 || frames > n.maxFrames {
		return fmt.Errorf("yue2: %d frames, past the %d the arena holds", frames, n.maxFrames)
	}
	if n.arLen+frames+2 > n.cache.Rows() {
		return fmt.Errorf("yue2: %d frames behind a prefix of %d, past the %d keys the cache holds", frames, n.arLen, n.cache.Rows())
	}
	tokens := frames + 2
	if n.tables != frames {
		if err := n.k.Write(int(n.pos), frameEmbedding(tokens, n.cfg.Dim)); err != nil {
			return err
		}
		if err := n.k.Write(int(n.table), ropeTable(n.arLen, tokens, n.cfg.RoPEBase)); err != nil {
			return err
		}
		n.tables = frames
	}
	return nil
}

// prepare writes the latent block, the timestep's features and, on a new
// shape, the frame embedding and the rotation table.
func (n *NAR) prepare(x []float32, frames int, rawT float32) error {
	if err := n.prepareTables(frames); err != nil {
		return err
	}
	lat := n.cfg.LatentDim
	if len(x) < lat*frames {
		return fmt.Errorf("yue2: %d latents for %d frames", len(x), frames)
	}
	tokens := frames + 2
	// LATENT_START and LATENT_END are zero rows around the frames.
	in := make([]float32, lat*tokens)
	copy(in[lat:], x[:lat*frames])
	if err := n.k.Write(int(n.xin), in); err != nil {
		return err
	}
	return n.k.Write(int(n.feats), timeFeatures(n.shiftT(rawT)))
}

// Velocity is one evaluation of the network: the velocity of frames frames of
// latents x, [t][64], at raw timestep rawT (the logit of t, saturated at
// ±20), over the prefix SetPrefix put on the card. It comes back as x is.
func (n *NAR) Velocity(x []float32, frames int, rawT float32) ([]float32, error) {
	if err := n.prepare(x, frames, rawT); err != nil {
		return nil, err
	}
	p, err := n.program(frames)
	if err != nil {
		return nil, err
	}
	if err := p.Run(); err != nil {
		return nil, err
	}
	lat := n.cfg.LatentDim
	return n.k.Read(int(n.out)+lat, lat*frames)
}

// program is the whole evaluation for frames frames, recorded once.
func (n *NAR) program(frames int) (*vk.Program, error) {
	if p, ok := n.progs[frames]; ok {
		return p, nil
	}
	tokens := frames + 2
	p, err := n.k.Compile(func(r *vk.Recorder) {
		n.recordInput(r, tokens)
		for l := range n.layers {
			n.recordLayer(r, l, tokens, false)
		}
		n.recordHead(r, tokens)
	})
	if err != nil {
		return nil, err
	}
	n.progs[frames] = p
	return p, nil
}

// Solve integrates the ODE on the card: noise is frames frames of 64 channels
// time-major, steps midpoint steps from t = 1 down to 0. It returns the
// latents at t = 0 in the layout of noise.
//
// The state stays resident on the card across all 2*steps evaluations; only the
// 256 sinusoidal timestep features cross the bus per evaluation.
func (n *NAR) Solve(noise []float32, frames, steps int, cancel func() bool) ([]float32, error) {
	return n.solveWithHook(noise, frames, steps, cancel, nil)
}

func (n *NAR) solveWithHook(noise []float32, frames, steps int, cancel func() bool, hook func(step int, state []float32)) ([]float32, error) {
	if err := n.prepareTables(frames); err != nil {
		return nil, err
	}
	lat := n.cfg.LatentDim
	if len(noise) < lat*frames {
		return nil, fmt.Errorf("yue2: %d noise floats for %d frames", len(noise), frames)
	}
	if steps < 1 {
		return nil, fmt.Errorf("yue2: %d steps", steps)
	}
	tokens := frames + 2
	// LATENT_START and LATENT_END are zero rows around the frames.
	in := make([]float32, lat*tokens)
	copy(in[lat:], noise[:lat*frames])
	if err := n.k.Write(int(n.state), in); err != nil {
		return nil, err
	}
	if err := n.k.Write(int(n.xin), in); err != nil {
		return nil, err
	}

	pEval, err := n.program(frames)
	if err != nil {
		return nil, err
	}

	dt := float32(1.0) / float32(steps)
	pMid, err := n.k.Compile(func(r *vk.Recorder) {
		n.k.Yue2Axpy(r, vk.K2Yue2Axpy{
			Dst: n.xin + uint32(lat),
			A:   n.state + uint32(lat),
			B:   n.out + uint32(lat),
			N:   uint32(frames * lat),
			S:   -0.5 * dt,
		})
	})
	if err != nil {
		return nil, err
	}
	defer pMid.Close()

	pStep, err := n.k.Compile(func(r *vk.Recorder) {
		n.k.Yue2Axpy(r, vk.K2Yue2Axpy{
			Dst:     n.state + uint32(lat),
			Dst2:    n.xin + uint32(lat),
			A:       n.state + uint32(lat),
			B:       n.out + uint32(lat),
			N:       uint32(frames * lat),
			S:       -dt,
			HasDst2: 1,
		})
	})
	if err != nil {
		return nil, err
	}
	defer pStep.Close()

	for step := range steps {
		if cancel != nil && cancel() {
			return nil, fmt.Errorf("%w: the flow at step %d of %d", ErrCancelled, step, steps)
		}
		t := float32(1.0) - float32(step)*dt
		rawT := logitClamped(t)
		if err := n.k.Write(int(n.feats), timeFeatures(n.shiftT(rawT))); err != nil {
			return nil, err
		}
		if err := pEval.Run(); err != nil {
			return nil, err
		}
		if err := pMid.Run(); err != nil {
			return nil, err
		}

		rawTMid := logitClamped(t - 0.5*dt)
		if err := n.k.Write(int(n.feats), timeFeatures(n.shiftT(rawTMid))); err != nil {
			return nil, err
		}
		if err := pEval.Run(); err != nil {
			return nil, err
		}
		if err := pStep.Run(); err != nil {
			return nil, err
		}

		if hook != nil {
			s, err := n.k.Read(int(n.state)+lat, lat*frames)
			if err != nil {
				return nil, err
			}
			hook(step, s)
		}
	}

	return n.k.Read(int(n.state)+lat, lat*frames)
}

// waypoints is one evaluation in pieces, a program each, reading back what
// yue2.cpp's dumper records of its first evaluation: temb_t, and token-major
// [N][2048] hidden_after_input, layer0_sa_output and hidden_after_layer
// {0, 7, 14, 21, 27}, and the velocity.
func (n *NAR) waypoints(x []float32, frames int, rawT float32) (map[string][]float32, error) {
	if err := n.prepare(x, frames, rawT); err != nil {
		return nil, err
	}
	tokens := frames + 2
	d := n.cfg.Dim
	got := map[string][]float32{}
	run := func(record func(r *vk.Recorder)) error {
		p, err := n.k.Compile(record)
		if err != nil {
			return err
		}
		defer p.Close()
		return p.Run()
	}
	read := func(name string, at uint32, count int) error {
		v, err := n.k.Read(int(at), count)
		got[name] = v
		return err
	}
	if err := run(func(r *vk.Recorder) { n.recordInput(r, tokens) }); err != nil {
		return nil, err
	}
	if err := read("temb_t", n.temb, d); err != nil {
		return nil, err
	}
	if err := read("hidden_after_input", n.x, d*tokens); err != nil {
		return nil, err
	}
	for l := range n.layers {
		if err := run(func(r *vk.Recorder) { n.recordLayer(r, l, tokens, l == 0) }); err != nil {
			return nil, err
		}
		if l == 0 {
			if err := read("layer0_sa_output", n.qkv, d*tokens); err != nil {
				return nil, err
			}
		}
		if l%7 == 0 || l == len(n.layers)-1 {
			if err := read(fmt.Sprintf("hidden_after_layer%d", l), n.x, d*tokens); err != nil {
				return nil, err
			}
		}
	}
	if err := run(func(r *vk.Recorder) { n.recordHead(r, tokens) }); err != nil {
		return nil, err
	}
	lat := n.cfg.LatentDim
	if err := read("velocity", n.out+uint32(lat), lat*frames); err != nil {
		return nil, err
	}
	return got, nil
}

// Close releases the card's memory.
func (n *NAR) Close() {
	n.forget()
	if n.cache != nil {
		n.cache.Close()
		n.cache = nil
	}
	if n.k != nil {
		n.k.Close()
		n.k = nil
	}
}
