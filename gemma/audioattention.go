package gemma

// Blocked local attention with a relative position encoding.
//
// The geometry is the conformer's, not the language model's. Positions are cut
// into chunks of twelve; a query in chunk b is shown a window of twenty-four
// keys — its own chunk and the twelve frames before the chunk begins — and the
// mask then takes back everything further than eleven frames away, which is
// what llama.cpp's condition (gq - gk) < max_past says and is easy to misread
// as twelve. So the window is twenty-four wide and no query ever sees more
// than twelve of it.
//
// On top of the content scores sits a relative term: thirteen sinusoidal rows,
// one per distance from zero to twelve, projected by attn_k_rel and dotted
// with the queries. llama.cpp then shifts that table the way Transformer-XL's
// appendix B does — pad each row to twenty-five, read the whole thing back
// twenty-four at a time, and every row has slid one place. Working the two
// indices out on paper says what the slide amounts to: on every key a query
// can actually see, the entry it lands on is that query's own row at the
// distance between the two. That is what the loop below adds, and there is a
// test that says the arithmetic agrees.

import (
	"math"

	"github.com/ThiraSoft/golem/nn"
)

// attention reads x — n positions of Cfg.Dim, the position the slower index —
// and writes what the block adds to its residual into out.
func (a *AudioTower) attention(b *AudioBlock, x, out []float32, n int, s *audioScratch) {
	cfg := a.Cfg
	dim, hd, heads := cfg.Dim, cfg.HeadDim, cfg.Heads
	C := cfg.Chunk
	blocks := (n + C - 1) / C

	norm := s.norm[:n*dim]
	copy(norm, x[:n*dim])
	nn.InParallel(n, n*dim, func(first, last int) {
		for p := first; p < last; p++ {
			nn.RMSNormPlain(norm[p*dim:(p+1)*dim], b.AttnPreNorm, cfg.Eps)
		}
	})

	q, k, v := s.q[:n*dim], s.k[:n*dim], s.v[:n*dim]
	ApplyShared([]VisionLinear{b.Q, b.K, b.V}, norm, s.wide, [][]float32{q, k, v}, n)

	// The scales llama.cpp applies to the operands rather than to the scores:
	// the usual one over the root of the head width, divided by ln 2 because
	// the softmax that follows was written in base two, and a matching factor
	// on the keys. The per-dimension gain rides along on the query.
	qScale := float32(1/math.Sqrt(float64(hd))) / float32(math.Ln2)
	kScale := float32(math.Log(1+math.E) / math.Ln2)
	nn.InParallel(n, n*dim, func(first, last int) {
		for p := first; p < last; p++ {
			for h := 0; h < heads; h++ {
				qh := q[p*dim+h*hd : p*dim+(h+1)*hd]
				for i := range qh {
					qh[i] *= qScale * b.PerDimScale[i]
				}
			}
			kp := k[p*dim : (p+1)*dim]
			for i := range kp {
				kp[i] *= kScale
			}
		}
	})

	rel := s.rel[:heads*cfg.RPE*hd]
	a.relativeRows(b, rel, s)

	ctx := s.ctx[:blocks*C*dim]
	nn.InParallel(heads*blocks, heads*blocks*C*cfg.Context*hd, func(first, last int) {
		for job := first; job < last; job++ {
			h, blk := job/blocks, job%blocks
			a.attendOneChunk(b, q, k, v, rel, ctx, s.scores[job*C*cfg.Context:], n, blk, h)
		}
	})

	// The chunk the blocking rounded up is dropped, the projection applied,
	// and the norm that closes the branch. The residual itself is the
	// caller's.
	b.Out.Apply(ctx[:n*dim], s.wide, out[:n*dim], n)
	nn.InParallel(n, n*dim, func(first, last int) {
		for p := first; p < last; p++ {
			nn.RMSNormPlain(out[p*dim:(p+1)*dim], b.AttnPostNorm, cfg.Eps)
		}
	})
}

// attendOneChunk computes one chunk of twelve queries for one head.
func (a *AudioTower) attendOneChunk(b *AudioBlock, q, k, v, rel, ctx, scores []float32, n, blk, h int) {
	cfg := a.Cfg
	dim, hd := cfg.Dim, cfg.HeadDim
	C, P, S, R := cfg.Chunk, cfg.Past, cfg.Context, cfg.RPE
	relH := rel[h*R*hd:]

	for qi := 0; qi < C; qi++ {
		gq := blk*C + qi
		if gq >= n {
			// A query past the end of the signal: its chunk exists because the
			// blocking rounded up, and nothing reads what it produces.
			continue
		}
		row := scores[qi*S : (qi+1)*S]
		qh := q[gq*dim+h*hd : gq*dim+(h+1)*hd]
		cap := cfg.Softcap
		for ki := 0; ki < S; ki++ {
			gk := blk*C - P + ki
			if !audioVisible(gq, gk, n, P) {
				row[ki] = float32(math.Inf(-1))
				continue
			}
			// The content score, and on top of it the relative one: the
			// distance between the two is the row of the relative table this
			// pair reads, counted down from the past horizon.
			sum := nn.DotF32(qh, k[gk*dim+h*hd:][:hd]) +
				nn.DotF32(qh, relH[(P-(gq-gk))*hd:][:hd])
			row[ki] = cap * float32(math.Tanh(float64(sum/cap)))
		}
		nn.SoftmaxGGML(row)

		dst := ctx[gq*dim+h*hd:][:hd]
		for d := range dst {
			dst[d] = 0
		}
		for ki := 0; ki < S; ki++ {
			gk := blk*C - P + ki
			w := row[ki]
			if w == 0 || !audioVisible(gq, gk, n, P) {
				continue
			}
			vh := v[gk*dim+h*hd:][:hd]
			for d := range dst {
				dst[d] += w * vh[d]
			}
		}
	}
}

// audioVisible is the mask, as one condition. clip.cpp writes it as four:
// the query is inside the signal, the key is inside the signal, the key is
// not in the future, and the two are less than the past horizon apart.
func audioVisible(gq, gk, n, past int) bool {
	return gq < n && gk >= 0 && gk < n && gk <= gq && gq-gk < past
}

// relativeRows builds the thirteen sinusoidal position rows and projects them
// through attn_k_rel, leaving rel[(h*R+r)*hd+d].
//
// The sinusoid is llama.cpp's own, filled in the GEMMA4A arm of clip.cpp's
// set_input: distances counted down from the past horizon to zero, the first
// half of the width sine and the second cosine, over timescales spread
// logarithmically to ten thousand. Inventing a different one would give a
// plausible-looking encoding the weights have never seen.
func (a *AudioTower) relativeRows(b *AudioBlock, rel []float32, s *audioScratch) {
	cfg := a.Cfg
	dim, hd, R := cfg.Dim, cfg.HeadDim, cfg.RPE
	timescales := dim / 2
	increment := math.Log(10000) / math.Max(float64(timescales-1), 1)

	pos := s.pos[:R*dim]
	for p := 0; p < R; p++ {
		position := float64(cfg.Past - p)
		for i := 0; i < timescales; i++ {
			scaled := position * math.Exp(-float64(i)*increment)
			pos[p*dim+i] = float32(math.Sin(scaled))
			pos[p*dim+i+timescales] = float32(math.Cos(scaled))
		}
	}

	// attn_k_rel is applied without its clamps: clip.cpp reaches for
	// ggml_mul_mat here rather than for the clamping build_mm every other
	// product in this tower goes through, and the file carries no ranges
	// beside this weight either.
	plain := b.KRel
	inf := float32(math.Inf(1))
	plain.Clamp = Clamp{InMin: -inf, InMax: inf, OutMin: -inf, OutMax: inf}
	plain.Apply(pos, s.posTmp[:R*dim], s.posOut[:R*dim], R)

	for h := 0; h < cfg.Heads; h++ {
		for r := 0; r < R; r++ {
			copy(rel[(h*R+r)*hd:(h*R+r+1)*hd], s.posOut[r*dim+h*hd:r*dim+(h+1)*hd])
		}
	}
}
