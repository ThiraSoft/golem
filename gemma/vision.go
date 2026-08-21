package gemma

// The vision tower: sixteen blocks over the patches of one image, then a
// pooling and a projection that put the result in the language model's own
// width.
//
// It runs once per image rather than once per token, which is why nothing here
// is repacked, quantized or cached. What it costs is read once and forgotten.

import (
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
