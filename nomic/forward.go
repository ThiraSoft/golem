package nomic

// The pass. Several texts go through at once: every matrix is read once for
// all their positions, and the attention is the only thing that keeps them
// apart — a position sees the positions of its own text and no other. That is
// what makes a batch of short texts cheap, and an embedding request is usually
// a batch of short texts.
//
// Everything between two products is spread over the cores as well, and every
// buffer is the model's and kept from one pass to the next. Both were measured
// before they were done: with the norms on one thread and the buffers made
// afresh, seven cores spent half of a long text spinning while the eighth
// zeroed memory and normed rows.

import (
	"context"
	"fmt"
	"math"

	"github.com/ThiraSoft/golem/nn"
)

// passTokens is the most positions one pass carries. It bounds the scratch —
// the widest buffer is a feed forward's, FF floats a position — and it is well
// past where a longer pass stops paying for itself.
const passTokens = 4096

// queryRows is how many queries of one head one unit of the attention takes.
// A single long text has twelve heads for eight cores; cut by queries as well
// it has a hundred and ninety-two units, and nobody waits for a straggler.
const queryRows = 32

// Tokenize is the text as this model reads it: framed by <s> and </s>, and cut
// to the context the checkpoint was trained on. The cut keeps the </s>,
// because the pooled vector is an average over every position and the
// checkpoint never saw a text without one.
func (m *Model) Tokenize(text string) []int32 {
	ids := m.Vocab.Encode(text, true, false)
	if len(ids) > m.Cfg.Context {
		last := ids[len(ids)-1]
		ids = append(ids[:m.Cfg.Context-1], last)
	}
	return ids
}

// Embed returns one vector per text, pooled the way the file says and not
// normalized: that is the caller's choice, and Normalize is the usual one.
// It is safe to call from several goroutines; the passes take turns.
func (m *Model) Embed(texts [][]int32) ([][]float32, error) {
	return m.EmbedContext(context.Background(), texts)
}

// EmbedContext is Embed with a context, looked at while waiting for the
// model and between two passes. A pass that has started runs to its end: it
// is milliseconds, and cutting one short would save nothing a caller notices.
// Checking costs nanoseconds against those milliseconds, and the passes are
// the same, so the vectors are the same floats.
func (m *Model) EmbedContext(ctx context.Context, texts [][]int32) ([][]float32, error) {
	if err := m.acquire(ctx); err != nil {
		return nil, err
	}
	defer m.release()
	out := make([][]float32, len(texts))
	for from := 0; from < len(texts); {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		to, width := from, 0
		for to < len(texts) && (to == from || width+len(texts[to]) <= passTokens) {
			width += len(texts[to])
			to++
		}
		rows, err := m.hidden(texts[from:to])
		if err != nil {
			return nil, err
		}
		for i, r := range rows {
			out[from+i] = m.pool(r)
		}
		from = to
	}
	return out, nil
}

// Hidden is what the last block wrote for every position of every text, one
// row a position. The rows are the model's own and the next pass overwrites
// them; copy them to keep them.
func (m *Model) Hidden(texts [][]int32) ([][][]float32, error) {
	m.turn <- struct{}{}
	defer m.release()
	return m.hidden(texts)
}

// acquire takes the model's turn, or gives up when ctx ends first.
func (m *Model) acquire(ctx context.Context) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	select {
	case m.turn <- struct{}{}:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

func (m *Model) release() { <-m.turn }

// span is one text's stretch of the pass.
type span struct{ start, length int }

// unit is a stretch of one head's queries, the attention's unit of work.
type unit struct{ text, head, first int }

// scratch is every buffer a pass needs, sized for the widest pass so far.
type scratch struct {
	positions                    int
	x, qkv, mixed, proj, ffn, up [][]float32
	answers                      [][]float32 // ExpertsUsed a position
	flat                         []float32   // a product's rounded operand
	transposed                   []float32   // each text's values, head by head, transposed
}

func (m *Model) reserve(n, texts int) *scratch {
	s := m.scratch
	if s != nil && s.positions >= n && len(s.transposed) >= m.Cfg.Dim*(n+8*texts) {
		return s
	}
	cfg := m.Cfg
	n = max(n, 64)
	s = &scratch{
		positions:  n,
		x:          rows(n, cfg.Dim),
		qkv:        rows(n, 3*cfg.Dim),
		mixed:      rows(n, cfg.Dim),
		proj:       rows(n, cfg.Dim),
		ffn:        rows(n, cfg.Dim),
		up:         rows(n, cfg.FF),
		answers:    rows(n*max(cfg.ExpertsUsed, 1), cfg.Dim),
		flat:       make([]float32, n*max(cfg.FF, 3*cfg.Dim)),
		transposed: make([]float32, cfg.Dim*(n+8*max(texts, n/8))),
	}
	m.scratch = s
	return s
}

func (m *Model) hidden(texts [][]int32) ([][][]float32, error) {
	cfg, w := m.Cfg, m.W
	var ids []int32
	var pos []int
	var spans []span
	for i, text := range texts {
		if len(text) == 0 {
			return nil, fmt.Errorf("nomic: text %d has no tokens", i)
		}
		if len(text) > cfg.Context {
			return nil, fmt.Errorf("nomic: text %d has %d tokens, past the context of %d", i, len(text), cfg.Context)
		}
		spans = append(spans, span{len(ids), len(text)})
		for p, id := range text {
			if id < 0 || int(id) >= w.TokenEmbd.Rows {
				return nil, fmt.Errorf("nomic: token %d is outside a vocabulary of %d", id, w.TokenEmbd.Rows)
			}
			ids = append(ids, id)
			pos = append(pos, p)
		}
	}
	n := len(ids)
	s := m.reserve(n, len(texts))
	x, qkv, mixed, proj, ffn := s.x[:n], s.qkv[:n], s.mixed[:n], s.proj[:n], s.ffn[:n]

	// The embedding, the sentence-A row, and a norm. llama.cpp names the sum
	// inp_embd and the normed stream inp_norm.
	nn.InParallel(n, n*cfg.Dim*4, func(first, last int) {
		for t := first; t < last; t++ {
			w.TokenEmbd.Row(int(ids[t]), x[t])
			for j, v := range w.TypeEmbd {
				x[t][j] += v
			}
		}
	})
	m.emit("inp_embd", x)
	if m.gpu != nil {
		if err := m.hiddenVulkan(x, pos, spans); err != nil {
			return nil, err
		}
		out := make([][][]float32, len(texts))
		for i, sp := range spans {
			out[i] = x[sp.start : sp.start+sp.length]
		}
		return out, nil
	}
	normRows(x, w.EmbNorm)
	m.emit("inp_norm-0", x)

	longest := 0
	for _, sp := range spans {
		longest = max(longest, sp.length)
	}
	ropes := make([]nn.RoPETable, longest)
	for p := range ropes {
		ropes[p].Prepare(cfg.HeadDim, p, cfg.RoPEBase, nil)
	}
	var units []unit
	for i, sp := range spans {
		for h := 0; h < cfg.Heads; h++ {
			for q := 0; q < sp.length; q += queryRows {
				units = append(units, unit{i, h, q})
			}
		}
	}

	for i := range w.Blocks {
		b := &w.Blocks[i]
		suffix := fmt.Sprintf("-%d", i)

		product(b.QKV, x, qkv, s.flat)
		// NeoX, over the whole of each head, positions counted from each
		// text's own start.
		nn.InParallel(n*cfg.Heads, n*cfg.Dim*8, func(first, last int) {
			for u := first; u < last; u++ {
				t, h := u/cfg.Heads, u%cfg.Heads
				at := h * cfg.HeadDim
				ropes[pos[t]].Apply(qkv[t][at : at+cfg.HeadDim])
				ropes[pos[t]].Apply(qkv[t][cfg.Dim+at : cfg.Dim+at+cfg.HeadDim])
			}
		})
		if m.trace != nil {
			m.emit("Qcur"+suffix, columns(qkv, 0, cfg.Dim))
			m.emit("Kcur"+suffix, columns(qkv, cfg.Dim, 2*cfg.Dim))
			m.emit("Vcur"+suffix, columns(qkv, 2*cfg.Dim, 3*cfg.Dim))
		}
		m.attention(s, spans, units, n)
		product(b.O, mixed, proj, s.flat)
		m.emit("kqv_out"+suffix, proj)

		// Post-norm: the residual goes in first, and the sum is normed.
		addNorm(proj, x, proj, b.AttnNorm)
		m.emit("ffn_inp"+suffix, proj)

		if cfg.Mixture(i) {
			m.mixture(b, s, proj, ffn, suffix)
			m.emit("ffn_moe_out"+suffix, ffn)
		} else {
			up := s.up[:n]
			product(b.Up, proj, up, s.flat)
			gelu(up)
			product(b.Down, up, ffn, s.flat)
			m.emit("ffn_out"+suffix, ffn)
		}
		addNorm(x, ffn, proj, b.OutNorm)
	}
	m.emit("result_embd", x)

	out := make([][][]float32, len(texts))
	for i, sp := range spans {
		out[i] = x[sp.start : sp.start+sp.length]
	}
	return out, nil
}

// attention mixes the values of each text's positions for every position of
// it, head by head, as two products: the queries against the keys, and the
// probabilities against the values turned on their side. There is no mask
// beyond the text's own bounds: an encoder reads both ways.
//
// The queries and keys are read out of the fused projection in place, a head
// at a time, by stride. The values are transposed once per text and head, so
// that the second product is rows against rows as well; their length is padded
// to a multiple of eight with zeros, which the probabilities meet as zeros.
func (m *Model) attention(s *scratch, spans []span, units []unit, n int) {
	cfg := m.Cfg
	hd, dim := cfg.HeadDim, cfg.Dim
	stride := 3 * dim
	qkvFlat := flatOf(s.qkv, n)
	outFlat := flatOf(s.mixed, n)
	scale := float32(1 / math.Sqrt(float64(hd)))

	padded := make([]int, len(spans))
	base := make([]int, len(spans)) // where each text's transposed values begin
	at := 0
	for i, sp := range spans {
		padded[i] = (sp.length + 7) / 8 * 8
		base[i] = at
		at += cfg.Heads * hd * padded[i]
	}
	vt := s.transposed[:at]

	nn.InParallel(len(spans)*cfg.Heads, n*dim, func(first, last int) {
		for u := first; u < last; u++ {
			i, h := u/cfg.Heads, u%cfg.Heads
			sp, lp := spans[i], padded[i]
			dst := vt[base[i]+h*hd*lp : base[i]+(h+1)*hd*lp]
			for j := 0; j < sp.length; j++ {
				v := qkvFlat[(sp.start+j)*stride+2*dim+h*hd:]
				for d := 0; d < hd; d++ {
					dst[d*lp+j] = v[d]
				}
			}
			for d := 0; d < hd; d++ {
				clear(dst[d*lp+sp.length : (d+1)*lp])
			}
		}
	})

	longest := 0
	for _, lp := range padded {
		longest = max(longest, lp)
	}
	nn.InParallel(len(units), len(units)*queryRows*longest*hd*4, func(first, last int) {
		scores := make([]float32, queryRows*longest)
		for _, u := range units[first:last] {
			sp, lp := spans[u.text], padded[u.text]
			rowsHere := min(queryRows, sp.length-u.first)
			q0 := sp.start + u.first
			sc := scores[:rowsHere*lp]
			nn.GemmF32NT(
				qkvFlat[q0*stride+u.head*hd:], stride,
				qkvFlat[sp.start*stride+dim+u.head*hd:], stride,
				hd, rowsHere, sp.length, sc, lp)
			for r := 0; r < rowsHere; r++ {
				row := sc[r*lp : r*lp+sp.length]
				// ggml scales inside the softmax; a product and then a
				// scale is the same float either way.
				for j := range row {
					row[j] *= scale
				}
				// ggml's exponential, which is a polynomial and not expf.
				nn.SoftmaxGGML(row)
				clear(sc[r*lp+sp.length : (r+1)*lp])
			}
			v := vt[base[u.text]+u.head*hd*lp:]
			nn.GemmF32NT(sc, lp, v, lp, lp, rowsHere, hd, outFlat[q0*dim+u.head*hd:], dim)
		}
	})
}

// mixture is a mixture block's feed forward. The router scores the experts,
// its softmax picks two, and each expert reads, in one batch, the positions
// that picked it. What a position gets back is the two answers weighted by
// the probabilities the softmax gave them — which do not sum to one, and are
// not made to: build_moe_ffn is called with norm_w false for this model.
func (m *Model) mixture(b *BlockWeights, s *scratch, in, out [][]float32, suffix string) {
	cfg := m.Cfg
	n, k := len(in), cfg.ExpertsUsed
	logits := rows(n, cfg.Experts)
	product(b.Router, in, logits, s.flat)
	m.emit("ffn_moe_logits"+suffix, logits)

	choice := make([]int, n*k)
	weight := rows(n, k)
	probs := make([]float32, cfg.Experts)
	for t := range logits {
		pick(logits[t], probs, choice[t*k:(t+1)*k], weight[t])
	}
	m.emit("ffn_moe_weights"+suffix, weight)

	// answers[t*k+s] is what the s-th expert position t chose wrote for it.
	answers := s.answers[:n*k]
	picked := make([][]float32, 0, n)
	written := make([][]float32, 0, n)
	for e := 0; e < cfg.Experts; e++ {
		picked, written = picked[:0], written[:0]
		for i, c := range choice {
			if c == e {
				picked = append(picked, in[i/k])
				written = append(written, answers[i])
			}
		}
		if len(picked) == 0 {
			continue
		}
		hidden := s.up[:len(picked)]
		product(b.UpExps[e], picked, hidden, s.flat)
		gelu(hidden)
		product(b.DownExps[e], hidden, written, s.flat)
	}

	// Weighted, then summed in slot order, as the graph adds its views.
	nn.InParallel(n, n*cfg.Dim*k*2, func(first, last int) {
		for t := first; t < last; t++ {
			clear(out[t])
			for slot := 0; slot < k; slot++ {
				ws := weight[t][slot]
				for j, v := range answers[t*k+slot] {
					out[t][j] += v * ws
				}
			}
		}
	})
}

// pick is the router's choice for one position: ggml's softmax over its
// logits, then the len(choice) best, the lower index first between equals,
// with the probability each was given. probs is scratch as wide as logits.
func pick(logits, probs []float32, choice []int, weight []float32) {
	copy(probs, logits)
	nn.SoftmaxGGML(probs)
	for slot := range choice {
		best := -1
		for e, p := range probs {
			if p < 0 {
				continue
			}
			if best < 0 || p > probs[best] {
				best = e
			}
		}
		choice[slot] = best
		weight[slot] = probs[best]
		probs[best] = -1
	}
}

// pool makes one vector of a text's rows.
func (m *Model) pool(rows [][]float32) []float32 {
	out := make([]float32, m.Cfg.Dim)
	switch m.Cfg.Pooling {
	case PoolCLS:
		copy(out, rows[0])
	case PoolLast:
		copy(out, rows[len(rows)-1])
	default:
		// llama.cpp averages with a product against a column of 1/n, so each
		// row is scaled before it is added rather than the sum divided after.
		inv := 1 / float32(len(rows))
		for _, r := range rows {
			for j, v := range r {
				out[j] += v * inv
			}
		}
	}
	return out
}

// Normalize scales a vector to unit length, as llama.cpp's
// common_embd_normalize does for its default of 2, and as ollama does to
// every embedding it returns. A zero vector stays zero.
func Normalize(v []float32) {
	var sum float64
	for _, x := range v {
		sum += float64(x) * float64(x)
	}
	if sum == 0 {
		return
	}
	inv := 1 / math.Sqrt(sum)
	for i, x := range v {
		v[i] = float32(float64(x) * inv)
	}
}

// product is ys = m·xs + bias. An fp16 matrix goes through the tiled kernel,
// its operand rounded to fp16 first as ggml rounds it, into flat; anything
// else, and any machine without the kernel, goes through MatVecBatch.
func product(m nn.Matrix, xs, ys [][]float32, flat []float32) {
	if m.Quant == nn.F16 && m.Cols%8 == 0 && nn.TiledProducts() {
		width := m.Cols
		x := flat[:len(xs)*width]
		nn.InParallel(len(xs), len(xs)*width*4, func(first, last int) {
			for t := first; t < last; t++ {
				dst := x[t*width : (t+1)*width]
				copy(dst, xs[t][:width])
				nn.RoundHalfRange(dst)
			}
		})
		if m.MatMulF16(x, ys) {
			return
		}
	}
	m.MatVecBatch(input(xs, m.Quant), ys)
}

// input is a batch holding rows, in the form a matrix of format q reads.
//
// ggml converts an activation to what the weight's dot product wants: fp16 for
// an fp16 weight, bfloat16 for a bfloat16 one, the Q8_0 blocks for the
// quantized formats. The fp16 rounding is not in nn.Batch, which has only ever
// fed fp16 weights from a vision projector, so it is done here.
func input(rows [][]float32, q nn.Quant) *nn.Batch {
	b := nn.NewBatch(len(rows[0]), len(rows))
	nn.InParallel(len(rows), len(rows)*len(rows[0]), func(first, last int) {
		for t := first; t < last; t++ {
			copy(b.F[t], rows[t])
			if q == nn.F16 {
				nn.RoundHalfRange(b.F[t])
			}
		}
	})
	switch q {
	case nn.F32, nn.F16:
	case nn.BF16:
		b.BF16 = true
		for t := range b.F {
			b.QuantizeColumnRange(t, 0, b.Width)
		}
	default:
		b.Quantize()
	}
	return b
}

// gelu is ggml's, looked up in its fp16 table, over every row.
func gelu(rows [][]float32) {
	nn.InParallel(len(rows), len(rows)*len(rows[0])*4, func(first, last int) {
		for t := first; t < last; t++ {
			nn.GELUTable(rows[t])
		}
	})
}

// normRows norms every row in place.
func normRows(x [][]float32, n nn.LayerNorm) {
	nn.InParallel(len(x), len(x)*len(x[0])*8, func(first, last int) {
		for t := first; t < last; t++ {
			layerNorm(x[t], n)
		}
	})
}

// addNorm writes norm(a + b) into dst, row by row; dst may be either operand.
// ggml adds the block's output to the residual in that order, and a float sum
// of two terms does not care.
func addNorm(dst, a, b [][]float32, n nn.LayerNorm) {
	nn.InParallel(len(dst), len(dst)*len(dst[0])*8, func(first, last int) {
		for t := first; t < last; t++ {
			for j := range dst[t] {
				dst[t][j] = a[t][j] + b[t][j]
			}
			layerNorm(dst[t], n)
		}
	})
}

// layerNorm is ggml_norm followed by the gain and the bias, to the bit.
//
// nn.LayerNormGGML is the same function written the obvious way, and it is off
// by an ulp here and there: ggml sums the squares eight at a time with SSE's
// horizontal reduction, and takes the reciprocal of a float square root rather
// than the square root of a double. An ulp is nothing to an fp16 checkpoint.
// To a Q8_0 one it is a whole quantization step in the next product, once in a
// while, and a gap that grows block by block from there. Twelve post-norm
// blocks make two dozen of these, so this one follows ggml's order exactly:
// vec.cpp's ggml_vec_cvar_f32 on AVX2, and ops.cpp's
// ggml_compute_forward_norm_f32 around it.
func layerNorm(x []float32, n nn.LayerNorm) {
	var total float64
	for _, v := range x {
		total += float64(v)
	}
	mean := float32(total) / float32(len(x))

	var sum float64
	i := 0
	for ; i+7 < len(x); i += 8 {
		var sq [8]float32
		for k := range sq {
			d := x[i+k] - mean
			x[i+k] = d
			sq[k] = d * d
		}
		// _mm_add_ps of the two halves, then movehl, then movehdup.
		a0, a1, a2, a3 := sq[0]+sq[4], sq[1]+sq[5], sq[2]+sq[6], sq[3]+sq[7]
		sum += float64((a0 + a2) + (a1 + a3))
	}
	for ; i < len(x); i++ {
		d := x[i] - mean
		x[i] = d
		sum += float64(d * d)
	}
	variance := float32(sum / float64(len(x)))
	scale := 1 / float32(math.Sqrt(float64(variance+n.Eps)))
	// Three graph nodes, three roundings. The conversions keep the compiler
	// from fusing the last multiply into the add, which the spec allows it to.
	for j, v := range x {
		x[j] = float32(float32(v*scale)*n.Gain[j]) + n.Bias[j]
	}
}

func rows(n, width int) [][]float32 {
	flat := make([]float32, n*width)
	out := make([][]float32, n)
	for i := range out {
		out[i] = flat[i*width : (i+1)*width]
	}
	return out
}

// flatOf is the first n rows of a buffer made by rows, as the one slice they
// are views of.
func flatOf(r [][]float32, n int) []float32 {
	width := len(r[0])
	return r[0][: n*width : n*width]
}

// columns is the same stretch of every row, as views.
func columns(src [][]float32, from, to int) [][]float32 {
	out := make([][]float32, len(src))
	for i, r := range src {
		out[i] = r[from:to]
	}
	return out
}

// emit hands a waypoint to the trace, when there is one.
func (m *Model) emit(name string, rows [][]float32) {
	if m.trace != nil {
		m.trace(name, rows)
	}
}
