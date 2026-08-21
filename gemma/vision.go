package gemma

// The vision tower: sixteen blocks over the patches of one image, then a
// pooling and a projection that put the result in the language model's own
// width.
//
// It runs once per image rather than once per token, and that is what decides
// its shape. A language model draws one token at a time, so its products are
// matrix-vector and the bus is what limits them. A tower is handed a thousand
// patches at once, so its products are matrix-matrix: every projection here
// takes the whole grid in one call, the weight is read once for all of it, and
// what limits the pass is arithmetic rather than the bus.
//
// The activations are therefore flat — one buffer of patches by width, the
// patch the slower index — rather than a slice per patch. That is the layout
// nn's kernels read, and going through a slice of slices would mean copying
// into it and out of it around every product.
//
// The feed forward is the exception that is tiled: its intermediate is four
// times as wide as the stream, and a whole grid of it would be tens of
// megabytes that no cache holds and nothing reads twice.

import (
	"math"

	"github.com/ThiraSoft/golem/imageio"
	"github.com/ThiraSoft/golem/nn"
)

// ffnTile is how many patches go through the feed forward together. Bigger is
// faster — every tile is a fresh sweep over three matrices, and four sections
// the cores have to meet at — and the whole grid at once is fastest of all.
// What stops it there is memory: the intermediate is four times as wide as the
// stream, so a thousand patches of it are already twelve megabytes, and an
// image is allowed to be twice this one.
const ffnTile = 1024

type VisionTower struct {
	Cfg *VisionConfig
	W   *VisionWeights
}

func NewVisionTower(cfg *VisionConfig, w *VisionWeights) *VisionTower {
	return &VisionTower{Cfg: cfg, W: w}
}

// scratch is one image's working memory, made once and used by every block.
type scratch struct {
	normed, q, k, v, attn []float32 // patches by Dim
	// qh, kh and vh are the same three, head by head: one head's patches laid
	// end to end, so that a head's attention walks contiguous memory. The
	// attention is half of what this tower costs and it is a thousand rows of
	// sixty-four floats against a thousand more; strided through a
	// patch-major buffer, every one of those rows would start a new cache
	// line for eight useful floats.
	qh, kh, vh     []float32 // heads by patches by HeadDim
	tmp            []float32 // the clamped input of whichever product is running
	gate, up, down []float32 // ffnTile by FFN, and by Dim for the last
}

func (v *VisionTower) newScratch(n int) *scratch {
	cfg := v.Cfg
	tile := min(n, ffnTile)
	return &scratch{
		normed: make([]float32, n*cfg.Dim),
		q:      make([]float32, n*cfg.Dim),
		k:      make([]float32, n*cfg.Dim),
		v:      make([]float32, n*cfg.Dim),
		attn:   make([]float32, n*cfg.Dim),
		qh:     make([]float32, n*cfg.Dim),
		kh:     make([]float32, n*cfg.Dim),
		vh:     make([]float32, n*cfg.Dim),
		tmp:    make([]float32, max(n*cfg.Dim, tile*cfg.FFN)),
		gate:   make([]float32, tile*cfg.FFN),
		up:     make([]float32, tile*cfg.FFN),
		down:   make([]float32, tile*cfg.Dim),
	}
}

// Patches cuts the image into patches, projects each of them, and adds the two
// learned position embeddings — one indexed by the patch's column, one by its
// row. What comes back is the grid, flat: patches by Dim.
//
// The projection is a 16x16 convolution with a stride of 16, which is to say a
// product against a patch that overlaps nothing. It is written as a product,
// and like every other one here it is taken over the whole grid at once: the
// split is over the output rows, so the weight is read once and the patches
// stream past it.
func (v *VisionTower) Patches(pixels []float32, cols, rows int) []float32 {
	cfg, w := v.Cfg, v.W
	ps := cfg.PatchSize
	width := cols * ps
	plane := width * rows * ps
	n := cols * rows
	area := ps * ps * 3

	// The patches, cut out and scaled to -1..1, in the order the weight was
	// written in: x fastest, then y, then the channel.
	flat := make([]float32, n*area)
	nn.InParallel(n, n*area, func(first, last int) {
		for p := first; p < last; p++ {
			px, py := p%cols, p/cols
			patch := flat[p*area : (p+1)*area]
			for c := 0; c < 3; c++ {
				for y := 0; y < ps; y++ {
					src := c*plane + (py*ps+y)*width + px*ps
					dst := (c*ps + y) * ps
					for x := 0; x < ps; x++ {
						patch[dst+x] = pixels[src+x]*2 - 1
					}
				}
			}
		}
	})

	out := make([]float32, n*cfg.Dim)
	nn.InParallel(cfg.Dim, n*area*cfg.Dim, func(start, end int) {
		for o := start; o < end; o++ {
			row := w.PatchEmbd[o*area : (o+1)*area]
			for p := 0; p < n; p++ {
				out[p*cfg.Dim+o] = nn.DotF32(flat[p*area:(p+1)*area], row)
			}
		}
	})

	// Two lookups, both added: x by column, y by row.
	nn.InParallel(n, n*cfg.Dim, func(first, last int) {
		for p := first; p < last; p++ {
			px, py := p%cols, p/cols
			xs := w.PosX[px*cfg.Dim : (px+1)*cfg.Dim]
			ys := w.PosY[py*cfg.Dim : (py+1)*cfg.Dim]
			row := out[p*cfg.Dim : (p+1)*cfg.Dim]
			for i := range row {
				row[i] += xs[i] + ys[i]
			}
		}
	})
	return out
}

// Block runs one block over every patch, in place. xs is the grid, flat.
//
// What separates it from the text model's block: the rotation is
// two-dimensional — the low half of each head turns with the patch's column
// and the high half with its row, both at base 100 — the values are
// RMS-normed before attention, the scores are not divided by anything, and
// every patch sees every other one. There is no cache: an image is encoded
// whole, once.
//
// The feed forward is a quick-GELU gate rather than a SwiGLU: the projector
// declares neither use_gelu nor use_silu, and that is the default clip.cpp
// falls to.
func (v *VisionTower) Block(i int, xs []float32, cols int, s *scratch) {
	cfg, w := v.Cfg, &v.W.Blocks[i]
	n := len(xs) / cfg.Dim
	hd := cfg.HeadDim
	half := hd / 2

	// The norm that opens the block, then the three projections, each over the
	// whole grid.
	nn.InParallel(n, n*cfg.Dim, func(first, last int) {
		for p := first; p < last; p++ {
			row := s.normed[p*cfg.Dim : (p+1)*cfg.Dim]
			copy(row, xs[p*cfg.Dim:(p+1)*cfg.Dim])
			nn.RMSNormPlain(row, w.LN1, cfg.Eps)
		}
	})
	w.Q.Apply(s.normed, s.tmp, s.q, n)
	w.K.Apply(s.normed, s.tmp, s.k, n)
	w.V.Apply(s.normed, s.tmp, s.v, n)

	// Then the per-head business: the norms, and a rotation that depends on
	// where the patch is in the grid and on nothing else. What comes out is
	// written head-major, which is the layout the attention below reads.
	nn.InParallel(n, n*cfg.Dim, func(first, last int) {
		var rx, ry nn.RoPETable
		for p := first; p < last; p++ {
			rx.Prepare(half, p%cols, cfg.RoPEBase, nil)
			ry.Prepare(half, p/cols, cfg.RoPEBase, nil)
			for h := 0; h < cfg.Heads; h++ {
				from, to := p*cfg.Dim+h*hd, (h*n+p)*hd
				q := s.qh[to : to+hd]
				k := s.kh[to : to+hd]
				v := s.vh[to : to+hd]
				copy(q, s.q[from:from+hd])
				copy(k, s.k[from:from+hd])
				copy(v, s.v[from:from+hd])
				nn.RMSNormPlain(q, w.QNorm, cfg.Eps)
				nn.RMSNormPlain(k, w.KNorm, cfg.Eps)
				// The value carries no gain of its own and is not rotated.
				nn.RMSNormPlain(v, nil, cfg.Eps)
				rx.Apply(q[:half])
				ry.Apply(q[half:])
				rx.Apply(k[:half])
				ry.Apply(k[half:])
			}
		}
	})

	// The attention itself, four patches of a head to a core. Every patch
	// attends to every patch: an image has no past.
	//
	// Four at a time because a head's keys are a quarter of a megabyte and
	// every query reads all of them: taken one at a time they would be read a
	// thousand times over, and the values with them. The four share what they
	// read, and the sums they scatter into are four rows of one buffer rather
	// than four places in the grid — which is why the block below puts them
	// back afterwards.
	groups := (n + 3) / 4
	nn.InParallel(groups*cfg.Heads, 2*n*n*cfg.Heads*hd, func(first, last int) {
		scores := make([]float32, 4*n)
		mixed := make([]float32, 4*hd)
		for u := first; u < last; u++ {
			h, g := u/groups, u%groups
			head := h * n * hd
			p := g * 4
			k := s.kh[head : head+n*hd]
			v := s.vh[head : head+n*hd]
			// No 1/sqrt(head_dim): gemma4v scales the scores by one, and the
			// query norm is what keeps them in range.
			if p+4 <= n && nn.Scores4(s.qh[head+p*hd:], k, hd, n, scores, n) {
				for i := 0; i < 4; i++ {
					nn.SoftmaxGGML(scores[i*n : (i+1)*n])
				}
				clear(mixed)
				nn.Mix4(mixed, hd, v, scores, n, hd, n)
				for i := 0; i < 4; i++ {
					copy(s.attn[(p+i)*cfg.Dim+h*hd:], mixed[i*hd:(i+1)*hd])
				}
				continue
			}
			// The tail of a grid that is not a whole number of fours.
			for i := 0; p+i < n && i < 4; i++ {
				nn.Scores(s.qh[head+(p+i)*hd:head+(p+i+1)*hd], k, hd, n, scores[:n])
				nn.SoftmaxGGML(scores[:n])
				dst := s.attn[(p+i)*cfg.Dim+h*hd : (p+i)*cfg.Dim+(h+1)*hd]
				clear(dst)
				nn.Mix(dst, v, scores[:n], hd, n)
			}
		}
	})

	// The output projection, its post-norm and the residual, over the whole
	// grid; then the feed forward, which is tiled because its intermediate is
	// four times as wide as the stream.
	w.O.Apply(s.attn, s.tmp, s.normed, n)
	nn.InParallel(n, n*cfg.Dim, func(first, last int) {
		for i := first * cfg.Dim; i < last*cfg.Dim; i += cfg.Dim {
			row := s.normed[i : i+cfg.Dim]
			nn.RMSNormPlain(row, w.PostAttnNorm, cfg.Eps)
			for j, val := range row {
				xs[i+j] += val
			}
		}
	})

	for at := 0; at < n; at += ffnTile {
		to := min(at+ffnTile, n)
		batch := to - at
		grid := xs[at*cfg.Dim : to*cfg.Dim]
		normed := s.normed[:batch*cfg.Dim]
		nn.InParallel(batch, batch*cfg.Dim, func(first, last int) {
			for i := first * cfg.Dim; i < last*cfg.Dim; i += cfg.Dim {
				row := normed[i : i+cfg.Dim]
				copy(row, grid[i:i+cfg.Dim])
				nn.RMSNormPlain(row, w.LN2, cfg.Eps)
			}
		})
		w.Gate.Apply(normed, s.tmp, s.gate, batch)
		w.Up.Apply(normed, s.tmp, s.up, batch)
		nn.InParallel(batch, batch*cfg.FFN, func(first, last int) {
			nn.GEGLUQuickRange(s.gate, s.up, first*cfg.FFN, last*cfg.FFN)
		})
		w.Down.Apply(s.gate[:batch*cfg.FFN], s.tmp, s.down, batch)
		nn.InParallel(batch, batch*cfg.Dim, func(first, last int) {
			for i := first * cfg.Dim; i < last*cfg.Dim; i += cfg.Dim {
				row := s.down[i : i+cfg.Dim]
				nn.RMSNormPlain(row, w.PostFFNNorm, cfg.Eps)
				for j, val := range row {
					grid[i+j] += val
				}
			}
		})
	}
}

// Encode is the whole tower: an image in, one row per soft token out, each as
// wide as the language model's embedding.
func (v *VisionTower) Encode(im *imageio.Image) [][]float32 {
	cfg := v.Cfg
	pixels, cols, rows := cfg.Prepare(im)
	xs := v.Patches(pixels, cols, rows)
	s := v.newScratch(cols * rows)
	for i := 0; i < cfg.Blocks; i++ {
		v.Block(i, xs, cols, s)
	}
	return v.Project(v.Pool(xs, cols, rows))
}

// Pool averages each Merge by Merge square of patches into one token and
// scales by the square root of the width, which is where the tower stops
// having a grid.
func (v *VisionTower) Pool(xs []float32, cols, rows int) []float32 {
	cfg := v.Cfg
	m := cfg.Merge
	outCols, outRows := cols/m, rows/m
	scale := float32(math.Sqrt(float64(cfg.Dim))) / float32(m*m)

	out := make([]float32, outCols*outRows*cfg.Dim)
	for t := 0; t < outCols*outRows; t++ {
		row := out[t*cfg.Dim : (t+1)*cfg.Dim]
		ox, oy := t%outCols, t/outCols
		for dy := 0; dy < m; dy++ {
			for dx := 0; dx < m; dx++ {
				at := ((oy*m+dy)*cols + ox*m + dx) * cfg.Dim
				for i, val := range xs[at : at+cfg.Dim] {
					row[i] += val
				}
			}
		}
		for i := range row {
			row[i] *= scale
		}
	}
	return out
}

// Project standardizes, norms and projects into the language model's width.
// What comes back is a row per soft token, which is the shape the language
// model reads them in.
//
// The standardisation is skipped when the file carries no ranges for it, which
// is the case for E2B: llama.cpp treats those two tensors as a pair and so
// does this.
func (v *VisionTower) Project(tokens []float32) [][]float32 {
	cfg, w := v.Cfg, v.W
	n := len(tokens) / cfg.Dim
	normed := make([]float32, len(tokens))
	nn.InParallel(n, len(tokens), func(first, last int) {
		for p := first; p < last; p++ {
			row := normed[p*cfg.Dim : (p+1)*cfg.Dim]
			copy(row, tokens[p*cfg.Dim:(p+1)*cfg.Dim])
			if w.StdBias != nil {
				for i := range row {
					row[i] = (row[i] - w.StdBias[i]) * w.StdScale[i]
				}
			}
			nn.RMSNormPlain(row, nil, cfg.Eps)
		}
	})
	flat := make([]float32, n*cfg.ProjDim)
	w.Proj.Apply(normed, make([]float32, len(normed)), flat, n)

	out := make([][]float32, n)
	for i := range out {
		out[i] = flat[i*cfg.ProjDim : (i+1)*cfg.ProjDim]
	}
	return out
}
