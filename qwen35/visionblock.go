package qwen35

import (
	"math"

	"github.com/ThiraSoft/golem/nn"
)

// visionScratch is one image's working memory. The tower runs once per image
// and not once per token, so this is made per encode and not kept — but it is
// made once for the whole tower rather than per block, because a grid of four
// thousand patches is twenty megabytes and twenty-seven of those would be the
// collector's afternoon.
type visionScratch struct {
	normed []float32 // one patch under a norm
	qkv    []float32 // the fused projection of every patch
	q, k   []float32 // rotated, patch-major
	scores []float32 // one head's row of scores
	mixed  []float32 // the attention output before the projection
	attn   []float32 // after it
	up     []float32 // the feed forward's wide half
	branch []float32 // its answer
	merged []float32 // one merged group, for the projection
}

func newVisionScratch(cfg *VisionConfig, patches int) *visionScratch {
	return &visionScratch{
		normed: make([]float32, cfg.Dim),
		qkv:    make([]float32, patches*3*cfg.Dim),
		q:      make([]float32, patches*cfg.Dim),
		k:      make([]float32, patches*cfg.Dim),
		scores: make([]float32, patches),
		mixed:  make([]float32, patches*cfg.Dim),
		attn:   make([]float32, cfg.Dim),
		up:     make([]float32, cfg.FFN),
		branch: make([]float32, cfg.Dim),
		merged: make([]float32, cfg.Dim*cfg.Merge*cfg.Merge),
	}
}

// VisionBlockForward runs one block over the whole grid, in place.
//
// The order is qwen3vl.cpp's: a layer norm, the fused projection, the rotation
// on the queries and the keys, full attention with no mask, the output
// projection and a residual; then a second norm, a gateless feed forward under
// GELU, and a second residual.
//
// The norm is ggml_norm — the mean subtracted, a gain and a bias — where every
// norm of the text model is an RMS with a gain alone. clip.cpp calls it
// NORM_TYPE_NORMAL and nn.LayerNormGGML is what Gemma's unified embedder
// already reads it with.
func (cfg *VisionConfig) VisionBlockForward(w *VisionWeights, i int, xs []float32, at [][2]int, s *visionScratch) {
	b := &w.Blocks[i]
	patches := len(at)
	dim, heads, hd := cfg.Dim, cfg.Heads, cfg.HeadDim
	scale := float32(1 / math.Sqrt(float64(hd)))

	// The fused projection of every patch, then the rotation. Both are per
	// patch and neither reads another patch, so they go in one sweep.
	var table nn.RoPETable
	sections := nn.Sections{hd / 4, hd / 4, hd / 4, hd / 4}
	for p := 0; p < patches; p++ {
		x := xs[p*dim : (p+1)*dim]
		copy(s.normed, x)
		nn.LayerNormGGML(s.normed, b.LN1.Gain, b.LN1.Bias, cfg.Eps)
		b.QKV.Apply(s.normed, s.qkv[p*3*dim:(p+1)*3*dim])

		// The three are laid end to end in the fused output: the queries, then
		// the keys, then the values, each heads*hd wide.
		fused := s.qkv[p*3*dim : (p+1)*3*dim]
		table.PrepareVision(hd, [2]int{at[p][0], at[p][1]}, 10000, sections, nil)
		for h := 0; h < heads; h++ {
			q := fused[h*hd : (h+1)*hd]
			k := fused[dim+h*hd : dim+(h+1)*hd]
			table.Apply(q)
			table.Apply(k)
			copy(s.q[p*dim+h*hd:], q)
			copy(s.k[p*dim+h*hd:], k)
		}
	}

	// Full attention: every patch sees every patch, and there is no cache
	// because an image is encoded whole, once.
	nn.InParallel(patches, patches*patches*dim, func(first, last int) {
		scores := make([]float32, patches)
		for p := first; p < last; p++ {
			for h := 0; h < heads; h++ {
				q := s.q[p*dim+h*hd : p*dim+(h+1)*hd]
				for o := 0; o < patches; o++ {
					scores[o] = nn.DotF32(q, s.k[o*dim+h*hd:o*dim+(h+1)*hd]) * scale
				}
				nn.SoftmaxGGML(scores)
				mixed := s.mixed[p*dim+h*hd : p*dim+(h+1)*hd]
				for i := range mixed {
					mixed[i] = 0
				}
				for o := 0; o < patches; o++ {
					v := s.qkv[o*3*dim+2*dim+h*hd : o*3*dim+2*dim+(h+1)*hd]
					nn.Axpy(mixed, v, scores[o])
				}
			}
		}
	})

	// The output projection, the residual, then the feed forward and its own.
	for p := 0; p < patches; p++ {
		x := xs[p*dim : (p+1)*dim]
		b.O.Apply(s.mixed[p*dim:(p+1)*dim], s.attn)
		for d := range x {
			x[d] += s.attn[d]
		}

		copy(s.normed, x)
		nn.LayerNormGGML(s.normed, b.LN2.Gain, b.LN2.Bias, cfg.Eps)
		b.Up.Apply(s.normed, s.up)
		nn.GELUTable(s.up)
		b.Dn.Apply(s.up, s.branch)
		for d := range x {
			x[d] += s.branch[d]
		}
	}
}
