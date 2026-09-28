package laya

// The pass. Every question of a request is its own sequence, and they go
// through together: every matrix is read once for all their positions, and the
// attention is what keeps them apart: a position sees its own sequence and no
// other, and in a local block only sixty-four positions of it either side.

import (
	"fmt"
	"math"
	"unsafe"

	"github.com/ThiraSoft/golem/nn"
)

// queryRows is how many queries of one head one unit of the attention takes:
// six columns of the tile, six times.
const queryRows = 36

// seq is one question's pass: its identifiers, where its option markers are,
// and its kind.
type seq struct {
	ids     []int32
	markers []int
	kind    int
}

// output is what the pass gives back for one sequence: the raw logit of each
// option, and the act head's two probabilities.
type output struct {
	logits []float32
	// actLogits is what the act head writes, act its softmax.
	actLogits, act [2]float32
}

type span struct{ start, length int }

type unit struct{ span, head, first int }

func (m *Model) forward(seqs []seq) ([]output, error) {
	cfg, w := m.Cfg, m.W
	d := cfg.Dim
	var ids []int32
	var pos []int
	var spans []span
	for i, s := range seqs {
		if len(s.ids) == 0 || len(s.ids) > cfg.MaxLen {
			return nil, fmt.Errorf("laya: sequence %d has %d tokens, outside 1..%d", i, len(s.ids), cfg.MaxLen)
		}
		spans = append(spans, span{len(ids), len(s.ids)})
		for p, id := range s.ids {
			if id < 0 || int(id) >= w.TokenEmbd.Rows {
				return nil, fmt.Errorf("laya: token %d is outside a vocabulary of %d", id, w.TokenEmbd.Rows)
			}
			ids = append(ids, id)
			pos = append(pos, p)
		}
	}
	n := len(ids)
	if m.gpu != nil {
		// The card looks the embeddings up itself and sends back only the
		// rows score reads.
		x, hid, err := m.encodeVulkan(ids, pos, spans, seqs)
		if err != nil {
			return nil, err
		}
		return m.score(x, hid, spans, seqs), nil
	}
	m.pack()
	x := rows(n, d)
	// An fp16 table is widened eight halves at a time.
	var table []uint16
	if w.TokenEmbd.Quant == nn.F16 {
		table = unsafe.Slice((*uint16)(unsafe.Pointer(&w.TokenEmbd.Data[0])), len(w.TokenEmbd.Data)/2)
	}
	nn.InParallel(n, n*d*4, func(first, last int) {
		for t := first; t < last; t++ {
			if table != nil {
				at := int(ids[t]) * d
				nn.AxpyHalf(x[t], table[at:at+d], 1)
				continue
			}
			w.TokenEmbd.Row(int(ids[t]), x[t])
		}
	})
	normRows(x, x, w.EmbNorm)
	m.emit("embed", x)

	m.encode(x, pos, spans)
	m.head(x, spans, seqs)
	return m.score(x, nil, spans, seqs), nil
}

// encode runs the encoder's blocks and its last norm over x in place.
func (m *Model) encode(x [][]float32, pos []int, spans []span) {
	cfg, w := m.Cfg, m.W
	n, d := len(x), cfg.Dim
	h := rows(n, d)
	qkv := rows(n+16, 3*d)[:n] // attention's room past the end
	mixed := rows(n, d)
	proj := rows(n, d)
	up := rows(n, 2*cfg.FF)
	gated := rows(n, cfg.FF)
	flat := make([]float32, n*2*cfg.FF)

	longest := 0
	for _, sp := range spans {
		longest = max(longest, sp.length)
	}
	global := make([]nn.RoPETable, longest)
	local := make([]nn.RoPETable, longest)
	for p := range global {
		global[p].Prepare(cfg.HeadDim, p, cfg.GlobalBase, nil)
		local[p].Prepare(cfg.HeadDim, p, cfg.LocalBase, nil)
	}

	for i := range w.Encoder {
		b := &w.Encoder[i]
		in := x
		if b.AttnNorm.Gain != nil {
			normRows(h, x, b.AttnNorm)
			in = h
		}
		m.product(b.QKV, in, qkv, flat)
		ropes, window := local, cfg.Window
		if cfg.Global(i) {
			ropes, window = global, 0
		}
		attention(qkv, mixed, spans, cfg.Heads, cfg.HeadDim, window, ropes, pos)
		m.productAdd(b.O, mixed, x, proj, flat)
		normRows(h, x, b.MLPNorm)
		m.product(b.Up, h, up, flat)
		ff := cfg.FF
		nn.InParallel(n, n*ff*16, func(first, last int) {
			for t := first; t < last; t++ {
				copy(gated[t], up[t][:ff])
				nn.GELUErf(gated[t])
				for j, g := range up[t][ff:] {
					gated[t][j] *= g
				}
			}
		})
		m.productAdd(b.Down, gated, x, proj, flat)
		m.emit(fmt.Sprintf("layer-%d", i), x)
	}
	normRows(x, x, w.FinalNorm)
	m.emit("encoded", x)
}

// head adds each sequence's kind row and runs the two head layers over x in
// place: PyTorch's TransformerEncoderLayer with norm_first, which is a
// pre-norm block with biases and a ReLU.
func (m *Model) head(x [][]float32, spans []span, seqs []seq) {
	cfg, w := m.Cfg, m.W
	n, d := len(x), cfg.Dim
	for i, sp := range spans {
		row := w.TypeEmbd[seqs[i].kind]
		for t := sp.start; t < sp.start+sp.length; t++ {
			for j, v := range row {
				x[t][j] += v
			}
		}
	}
	h := rows(n, d)
	qkv := rows(n+16, 3*d)[:n] // attention's room past the end
	mixed := rows(n, d)
	proj := rows(n, d)
	up := rows(n, cfg.HeadFF)
	flat := make([]float32, n*cfg.HeadFF)
	for i := range w.Head {
		b := &w.Head[i]
		normRows(h, x, b.Norm1)
		m.product(b.QKV, h, qkv, flat)
		attention(qkv, mixed, spans, cfg.Heads, cfg.HeadDim, 0, nil, nil)
		m.productAdd(b.O, mixed, x, proj, flat)
		normRows(h, x, b.Norm2)
		m.product(b.Up, h, up, flat)
		nn.InParallel(n, n*cfg.HeadFF*8, func(first, last int) {
			for t := first; t < last; t++ {
				for j, v := range up[t] {
					up[t][j] = max(v, 0)
				}
			}
		})
		m.productAdd(b.Down, up, x, proj, flat)
		m.emit(fmt.Sprintf("head-%d", i), x)
	}
}

// score reads each sequence's markers into logits, and its first position
// with a summary of its own answer into the act head.
//
// hid, when the card has already run the scorer's norm and hidden layer over
// the markers, is that, a row a marker in order; nil runs them here.
func (m *Model) score(x, hid [][]float32, spans []span, seqs []seq) []output {
	cfg, w := m.Cfg, m.W
	d := cfg.Dim
	if hid == nil {
		var marked [][]float32
		for i, sp := range spans {
			for _, at := range seqs[i].markers {
				r := make([]float32, d)
				copy(r, x[sp.start+at])
				marked = append(marked, r)
			}
		}
		k := len(marked)
		normRows(marked, marked, w.ScoreNorm)
		hid = rows(k, d)
		m.product(w.ScoreHid, marked, hid, make([]float32, k*d))
		nn.InParallel(k, k*d*16, func(first, last int) {
			for _, r := range hid[first:last] {
				nn.GELUErf(r)
			}
		})
	}
	k := len(hid)
	one := rows(k, 1)
	m.product(w.ScoreOut, hid, one, make([]float32, k*d))

	out := make([]output, len(seqs))
	feats := rows(len(seqs), d+4)
	at := 0
	for i, s := range seqs {
		o := &out[i]
		o.logits = make([]float32, len(s.markers))
		for j := range o.logits {
			o.logits[j] = one[at+j][0]
		}
		at += len(s.markers)

		// The act head's features, from the distribution as the model gave
		// it: no temperature.
		p := softmax(o.logits)
		kk := float64(max(len(p), 2))
		var ent float64
		top1, top2 := float32(0), float32(0)
		for _, v := range p {
			ent -= float64(v) * math.Log(math.Max(float64(v), 1e-9))
			if v > top1 {
				top1, top2 = v, top1
			} else if v > top2 {
				top2 = v
			}
		}
		in := feats[i]
		copy(in, x[spans[i].start])
		in[d], in[d+1], in[d+2], in[d+3] = top1, top1-top2, float32(ent/math.Log(kk)), float32(kk/255)
	}

	// The act head, every sequence of the pass at once. Its first matrix
	// reads Dim+4, which no fp16 kernel takes, so it was widened once when
	// the model was opened.
	ah := rows(len(seqs), w.ActHid.Rows)
	m.actHidden(feats, ah)
	for _, r := range ah {
		nn.GELUErf(r)
	}
	act := rows(len(seqs), w.ActOut.Rows)
	m.product(w.ActOut, ah, act, make([]float32, len(seqs)*w.ActOut.Cols))
	for i := range out {
		copy(out[i].actLogits[:], act[i])
		copy(out[i].act[:], softmax(act[i]))
	}
	if m.trace != nil && len(seqs) == 1 {
		m.emit("logits", [][]float32{out[0].logits})
	}
	return out
}

// actHidden is the act head's first product, hid[t] = ActHid·feats[t] + bias,
// on the widened copy of its weights: the first Dim columns on the tile
// kernel, the four summary features after them one by one.
func (m *Model) actHidden(feats, hid [][]float32) {
	w := m.W.ActHid
	width, k := w.Cols, len(feats)
	wide := width &^ 7
	in := make([]float32, k*width)
	for t, r := range feats {
		copy(in[t*width:], r)
	}
	out := make([]float32, k*w.Rows)
	nn.GemmF32NT(in, width, m.actW, width, wide, k, w.Rows, out, w.Rows)
	for t := range feats {
		for r := 0; r < w.Rows; r++ {
			sum := out[t*w.Rows+r]
			for j := wide; j < width; j++ {
				sum += m.actW[r*width+j] * in[t*width+j]
			}
			hid[t][r] = sum + w.Bias[r]
		}
	}
}

// attention mixes each sequence's values for every position of it, head by
// head, after turning its queries and keys by ropes (nil for none). window > 0
// limits a query to the keys at most window positions away.
//
// Each sequence's keys are transposed once per head, a row a dimension padded
// to a multiple of sixteen, so that the scores are nn.GemmPanelsF32 with the
// keys along its sixteen lanes and the queries read out of the fused
// projection in place; the answer is the same kernel with the values'
// dimensions along the lanes, read in place too. qkv must have sixteen rows
// of room past its last: the answer reads up to fifteen values past a
// sequence's end, which the probabilities meet with zero.
func attention(qkv, out [][]float32, spans []span, heads, hd, window int, ropes []nn.RoPETable, pos []int) {
	n := len(qkv)
	dim := heads * hd
	stride := 3 * dim
	qkvFlat := qkv[0][: (n+16)*stride : (n+16)*stride]
	outFlat := flatOf(out, n)
	scale := float32(1 / math.Sqrt(float64(hd)))

	padded := make([]int, len(spans))
	base := make([]int, len(spans))
	total := 0
	for i, sp := range spans {
		padded[i] = (sp.length + 15) &^ 15
		base[i] = total
		total += heads * hd * padded[i]
	}
	kt := make([]float32, total)
	nn.InParallel(len(spans)*heads, n*dim*16, func(first, last int) {
		for u := first; u < last; u++ {
			i, h := u/heads, u%heads
			sp, lp := spans[i], padded[i]
			dst := kt[base[i]+h*hd*lp : base[i]+(h+1)*hd*lp]
			for j := 0; j < sp.length; j++ {
				t := sp.start + j
				q := qkvFlat[t*stride+h*hd : t*stride+(h+1)*hd]
				k := qkvFlat[t*stride+dim+h*hd : t*stride+dim+(h+1)*hd]
				if ropes != nil {
					ropes[pos[t]].Apply(q)
					ropes[pos[t]].Apply(k)
				}
				for c, v := range k {
					dst[c*lp+j] = v
				}
			}
		}
	})

	var units []unit
	longest := 0
	for i, sp := range spans {
		longest = max(longest, padded[i])
		for h := 0; h < heads; h++ {
			for q := 0; q < sp.length; q += queryRows {
				units = append(units, unit{i, h, q})
			}
		}
	}
	inf := float32(math.Inf(-1))
	nn.InParallel(len(units), len(units)*queryRows*min(longest, 2*window+queryRows+16)*hd*4, func(first, last int) {
		scores := make([]float32, queryRows*longest)
		for _, u := range units[first:last] {
			sp, lp := spans[u.span], padded[u.span]
			count := min(queryRows, sp.length-u.first)
			from, to := 0, sp.length
			if window > 0 {
				from = max(0, u.first-window) &^ 15
				to = min(sp.length, u.first+count+window)
			}
			width := (to - from + 15) &^ 15
			sc := scores[:count*width]
			q0 := sp.start + u.first
			nn.GemmPanelsF32(kt[base[u.span]+u.head*hd*lp+from:], lp,
				qkvFlat[q0*stride+u.head*hd:], stride, hd, width, count, sc, width)
			for r := 0; r < count; r++ {
				row := sc[r*width : (r+1)*width]
				q := u.first + r
				for j := range row {
					key := from + j
					if key >= to || window > 0 && (key < q-window || key > q+window) {
						row[j] = inf
						continue
					}
					row[j] *= scale
				}
				nn.SoftmaxF32(row)
			}
			nn.GemmPanelsF32(qkvFlat[(sp.start+from)*stride+2*dim+u.head*hd:], stride,
				sc, width, width, hd, count, outFlat[q0*dim+u.head*hd:], dim)
		}
	})
}

// productAdd is x += m·xs + bias, the residual stream taking a block's
// output: in the blocked product itself when it has the matrix, through
// proj otherwise.
func (model *Model) productAdd(m nn.Matrix, xs, x, proj [][]float32, flat []float32) {
	if model.blocked(m, xs, x, flat, true) {
		return
	}
	model.product(m, xs, proj, flat)
	addRows(x, proj)
}

// blocked runs ys = m·xs + bias (or adds it into ys) on nn's blocked product,
// if the matrix was packed for it and the rows are laid out as it reads them.
func (model *Model) blocked(m nn.Matrix, xs, ys [][]float32, flat []float32, add bool) bool {
	if len(xs) == 0 || len(m.Data) == 0 {
		return false
	}
	p, ok := model.packs[&m.Data[0]]
	if !ok {
		return false
	}
	out := contiguous(ys, m.Rows)
	if out == nil {
		return false
	}
	x := contiguous(xs, m.Cols)
	if x == nil {
		x = flat[:len(xs)*m.Cols]
		for t, r := range xs {
			copy(x[t*m.Cols:(t+1)*m.Cols], r[:m.Cols])
		}
	}
	p.GemmF16(x, len(xs), out, m.Bias, add)
	return true
}

// product is ys = m·xs + bias, in float32 activations: the reference is
// PyTorch in float32, which rounds nothing on the way in.
func (model *Model) product(m nn.Matrix, xs, ys [][]float32, flat []float32) {
	if model.blocked(m, xs, ys, flat, false) {
		return
	}
	width := m.Cols
	if m.Quant == nn.F16 && width%8 == 0 && nn.TiledProducts() {
		x := flat[:len(xs)*width]
		for t, r := range xs {
			copy(x[t*width:(t+1)*width], r[:width])
		}
		if m.MatMulF16(x, ys) {
			return
		}
	}
	if width%32 != 0 {
		// The act head's first matrix reads 1028: no kernel wants that
		// width, and it is one row of a pass.
		row := make([]float32, width)
		for r := 0; r < m.Rows; r++ {
			m.Row(r, row)
			for t, x := range xs {
				var sum float32
				for j, v := range row {
					sum += v * x[j]
				}
				if m.Bias != nil {
					sum += m.Bias[r]
				}
				ys[t][r] = sum
			}
		}
		return
	}
	b := nn.NewBatch(width, len(xs))
	for t, r := range xs {
		copy(b.F[t], r[:width])
	}
	switch m.Quant {
	case nn.F32, nn.F16:
	case nn.BF16:
		b.BF16 = true
		for t := range b.F {
			b.QuantizeColumnRange(t, 0, b.Width)
		}
	default:
		b.Quantize()
	}
	m.MatVecBatch(b, ys)
}

// contiguous is rs as one slice when they are width-long rows back to back in
// memory, as rows makes them, and nil otherwise.
func contiguous(rs [][]float32, width int) []float32 {
	if len(rs[0]) != width || cap(rs[0]) < len(rs)*width {
		return nil
	}
	flat := rs[0][:len(rs)*width]
	for i, r := range rs {
		if len(r) != width || &r[0] != &flat[i*width] {
			return nil
		}
	}
	return flat
}

// pack lays every matrix the processor multiplies a wide batch by out for
// nn's blocked product, once, the first time the processor runs a pass. It
// is a second copy of the weights, and the card never needs it.
func (m *Model) pack() {
	m.packOnce.Do(func() {
		m.packs = map[*byte]nn.PackedF16{}
		add := func(mat nn.Matrix) {
			if p, ok := nn.PackF16(mat); ok {
				m.packs[&mat.Data[0]] = p
			}
		}
		for i := range m.W.Encoder {
			b := &m.W.Encoder[i]
			add(b.QKV)
			add(b.O)
			add(b.Up)
			add(b.Down)
		}
		for i := range m.W.Head {
			b := &m.W.Head[i]
			add(b.QKV)
			add(b.O)
			add(b.Up)
			add(b.Down)
		}
		add(m.W.ScoreHid)
	})
}

// normRows writes norm(src) into dst row by row; they may be the same rows.
func normRows(dst, src [][]float32, n nn.LayerNorm) {
	nn.InParallel(len(src), len(src)*len(src[0])*64, func(first, last int) {
		for t := first; t < last; t++ {
			if &dst[t][0] != &src[t][0] {
				copy(dst[t], src[t])
			}
			n.Apply(dst[t])
		}
	})
}

// addRows adds b into a, row by row.
func addRows(a, b [][]float32) {
	nn.InParallel(len(a), len(a)*len(a[0])*16, func(first, last int) {
		for t := first; t < last; t++ {
			for j, v := range b[t] {
				a[t][j] += v
			}
		}
	})
}

// softmax is a fresh softmax of x, in float64 on the way.
func softmax(x []float32) []float32 {
	out := make([]float32, len(x))
	if len(x) == 0 {
		return out
	}
	peak := x[0]
	for _, v := range x {
		peak = max(peak, v)
	}
	var sum float64
	for i, v := range x {
		e := math.Exp(float64(v - peak))
		out[i] = float32(e)
		sum += e
	}
	for i := range out {
		out[i] = float32(float64(out[i]) / sum)
	}
	return out
}

func rows(n, width int) [][]float32 {
	flat := make([]float32, n*width)
	out := make([][]float32, n)
	for i := range out {
		out[i] = flat[i*width : (i+1)*width]
	}
	return out
}

func flatOf(r [][]float32, n int) []float32 {
	width := len(r[0])
	return r[0][: n*width : n*width]
}

func (m *Model) emit(name string, rows [][]float32) {
	if m.trace != nil {
		m.trace(name, rows)
	}
}
