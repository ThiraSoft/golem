package gemma

// The vision tower: sixteen blocks over the patches of one image, then a
// pooling and a projection that put the result in the language model's own
// width.
//
// It runs once per image rather than once per token, which is why nothing here
// is repacked, quantized or cached. What it costs is read once and forgotten.

import (
	"math"

	"github.com/ThiraSoft/golem/imageio"
	"github.com/ThiraSoft/golem/nn"
)

type VisionTower struct {
	Cfg *VisionConfig
	W   *VisionWeights
}

func NewVisionTower(cfg *VisionConfig, w *VisionWeights) *VisionTower {
	return &VisionTower{Cfg: cfg, W: w}
}

// Patches cuts the image into patches, projects each of them, and adds the two
// learned position embeddings — one indexed by the patch's column, one by its
// row.
//
// The projection is a 16x16 convolution with a stride of 16, which is to say a
// product against a patch that overlaps nothing. It is written as a product.
// The weight is laid out as the file wrote it: x fastest, then y, then the
// channel, then the output.
func (v *VisionTower) Patches(pixels []float32, cols, rows int) [][]float32 {
	cfg, w := v.Cfg, v.W
	ps := cfg.PatchSize
	width := cols * ps
	plane := width * rows * ps
	out := make([][]float32, cols*rows)

	nn.InParallel(len(out), len(out)*ps*ps*3*cfg.Dim, func(first, last int) {
		patch := make([]float32, ps*ps*3)
		for p := first; p < last; p++ {
			px, py := p%cols, p/cols
			for c := 0; c < 3; c++ {
				for y := 0; y < ps; y++ {
					src := c*plane + (py*ps+y)*width + px*ps
					dst := (c*ps + y) * ps
					// The graph scales the pixels to -1..1 before the
					// convolution, and the scaling is a patch's own business.
					for x := 0; x < ps; x++ {
						patch[dst+x] = pixels[src+x]*2 - 1
					}
				}
			}
			row := make([]float32, cfg.Dim)
			for o := 0; o < cfg.Dim; o++ {
				row[o] = nn.DotF32(patch, w.PatchEmbd[o*len(patch):(o+1)*len(patch)])
			}
			// Two lookups, both added: x by column, y by row.
			xs := w.PosX[px*cfg.Dim : (px+1)*cfg.Dim]
			ys := w.PosY[py*cfg.Dim : (py+1)*cfg.Dim]
			for i := range row {
				row[i] += xs[i] + ys[i]
			}
			out[p] = row
		}
	})
	return out
}

// Block runs one block over every patch, in place.
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
//
// Every product is clipped to the range it was quantized under, which is what
// VisionLinear.Apply does, and it runs on the caller's thread: the work is
// already split over the patches and nn's sections do not nest.
func (v *VisionTower) Block(i int, xs [][]float32, cols int) {
	cfg, w := v.Cfg, &v.W.Blocks[i]
	n := len(xs)
	hd := cfg.HeadDim
	half := hd / 2

	normed := make([][]float32, n)
	q := make([][]float32, n)
	k := make([][]float32, n)
	val := make([][]float32, n)
	attn := make([][]float32, n)
	for p := range xs {
		normed[p] = make([]float32, cfg.Dim)
		q[p] = make([]float32, cfg.Dim)
		k[p] = make([]float32, cfg.Dim)
		val[p] = make([]float32, cfg.Dim)
		attn[p] = make([]float32, cfg.Dim)
	}

	// The projections, and everything that is one patch's own business.
	nn.InParallel(n, n*3*cfg.Dim*cfg.Dim, func(first, last int) {
		var rx, ry nn.RoPETable
		tmp := make([]float32, cfg.Dim)
		for p := first; p < last; p++ {
			copy(normed[p], xs[p])
			nn.RMSNormPlain(normed[p], w.LN1, cfg.Eps)
			w.Q.Apply(normed[p], tmp, q[p])
			w.K.Apply(normed[p], tmp, k[p])
			w.V.Apply(normed[p], tmp, val[p])

			rx.Prepare(half, p%cols, cfg.RoPEBase, nil)
			ry.Prepare(half, p/cols, cfg.RoPEBase, nil)
			for h := 0; h < cfg.Heads; h++ {
				qh := q[p][h*hd : (h+1)*hd]
				kh := k[p][h*hd : (h+1)*hd]
				vh := val[p][h*hd : (h+1)*hd]
				nn.RMSNormPlain(qh, w.QNorm, cfg.Eps)
				nn.RMSNormPlain(kh, w.KNorm, cfg.Eps)
				// The value carries no gain of its own and is not rotated.
				nn.RMSNormPlain(vh, nil, cfg.Eps)
				rx.Apply(qh[:half])
				ry.Apply(qh[half:])
				rx.Apply(kh[:half])
				ry.Apply(kh[half:])
			}
		}
	})

	// The attention itself, a head of a patch to a core. Every patch attends to
	// every patch: an image has no past.
	nn.InParallel(n*cfg.Heads, 2*n*n*cfg.Heads*hd, func(first, last int) {
		scores := make([]float32, n)
		for u := first; u < last; u++ {
			p, h := u/cfg.Heads, u%cfg.Heads
			query := q[p][h*hd : (h+1)*hd]
			// No 1/sqrt(head_dim): gemma4v scales the scores by one, and the
			// query norm is what keeps them in range.
			for j := 0; j < n; j++ {
				scores[j] = nn.DotF32(query, k[j][h*hd:(h+1)*hd])
			}
			nn.SoftmaxInPlace(scores)
			dst := attn[p][h*hd : (h+1)*hd]
			clear(dst)
			for j := 0; j < n; j++ {
				nn.Axpy(dst, val[j][h*hd:(h+1)*hd], scores[j])
			}
		}
	})

	// The output projection, the post-norm, the residual, and the same shape
	// again for the feed forward.
	nn.InParallel(n, n*(cfg.Dim*cfg.Dim+3*cfg.Dim*cfg.FFN), func(first, last int) {
		out := make([]float32, cfg.Dim)
		gate := make([]float32, cfg.FFN)
		up := make([]float32, cfg.FFN)
		tmp := make([]float32, cfg.FFN)
		for p := first; p < last; p++ {
			w.O.Apply(attn[p], tmp[:cfg.Dim], out)
			nn.RMSNormPlain(out, w.PostAttnNorm, cfg.Eps)
			for j := range out {
				xs[p][j] += out[j]
			}

			copy(normed[p], xs[p])
			nn.RMSNormPlain(normed[p], w.LN2, cfg.Eps)
			w.Gate.Apply(normed[p], tmp[:cfg.Dim], gate)
			w.Up.Apply(normed[p], tmp[:cfg.Dim], up)
			nn.GEGLUQuickRange(gate, up, 0, cfg.FFN)
			w.Down.Apply(gate, tmp, out)
			nn.RMSNormPlain(out, w.PostFFNNorm, cfg.Eps)
			for j := range out {
				xs[p][j] += out[j]
			}
		}
	})
}

// Encode is the whole tower: an image in, one row per soft token out, each as
// wide as the language model's embedding.
func (v *VisionTower) Encode(im *imageio.Image) [][]float32 {
	cfg := v.Cfg
	pixels, cols, rows := cfg.Prepare(im)
	xs := v.Patches(pixels, cols, rows)
	for i := 0; i < cfg.Blocks; i++ {
		v.Block(i, xs, cols)
	}
	return v.Project(v.Pool(xs, cols, rows))
}

// Pool averages each Merge by Merge square of patches into one token and
// scales by the square root of the width, which is where the tower stops
// having a grid.
func (v *VisionTower) Pool(xs [][]float32, cols, rows int) [][]float32 {
	cfg := v.Cfg
	m := cfg.Merge
	outCols, outRows := cols/m, rows/m
	scale := float32(math.Sqrt(float64(cfg.Dim))) / float32(m*m)

	out := make([][]float32, outCols*outRows)
	for t := range out {
		row := make([]float32, cfg.Dim)
		ox, oy := t%outCols, t/outCols
		for dy := 0; dy < m; dy++ {
			for dx := 0; dx < m; dx++ {
				for i, val := range xs[(oy*m+dy)*cols+ox*m+dx] {
					row[i] += val
				}
			}
		}
		for i := range row {
			row[i] *= scale
		}
		out[t] = row
	}
	return out
}

// Project standardizes, norms and projects into the language model's width.
//
// The standardisation is skipped when the file carries no ranges for it, which
// is the case for E2B: llama.cpp treats those two tensors as a pair and so
// does this.
func (v *VisionTower) Project(tokens [][]float32) [][]float32 {
	cfg, w := v.Cfg, v.W
	out := make([][]float32, len(tokens))
	nn.InParallel(len(tokens), len(tokens)*cfg.Dim*cfg.ProjDim, func(first, last int) {
		tmp := make([]float32, cfg.Dim)
		for t := first; t < last; t++ {
			row := append([]float32(nil), tokens[t]...)
			if w.StdBias != nil {
				for i := range row {
					row[i] = (row[i] - w.StdBias[i]) * w.StdScale[i]
				}
			}
			nn.RMSNormPlain(row, nil, cfg.Eps)
			dst := make([]float32, cfg.ProjDim)
			w.Proj.Apply(row, tmp, dst)
			out[t] = dst
		}
	})
	return out
}
