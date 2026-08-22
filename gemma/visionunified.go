package gemma

// The other projector: an embedder with no tower behind it.
//
// The 12B declares gemma4uv where E2B declares gemma4v, and the file says what
// that means before any code does — eleven tensors, no block, no head. There
// is nothing to attend with: the patches are embedded here and every block
// they go through afterwards is the language model's, which is why the
// projection lands on the 12B's own width without changing it.
//
// What the embedder does, in the order models/gemma4uv.cpp does it: cut the
// image into patches three times as wide as the tower's, because the merging
// of three by three is the convolution's here; normalize the raw patch;
// project it and add a bias; normalize that; add the two position lookups;
// normalize again. Then the projection, which is Project below, shared with
// the tower.
//
// The three norms are layer norms — a mean subtracted, a gain and a bias —
// where every norm in the tower is an RMS norm with a gain alone. The comment
// in the reference says why: this part of the model is written in PyTorch's
// own nn.LayerNorm, whose default epsilon is 1e-5 and is not the file's.

import "github.com/ThiraSoft/golem/nn"

// patchNormEps is that default. It is hardcoded in gemma4uv.cpp for the same
// reason it is hardcoded here: no key of the file carries it.
const patchNormEps = 1e-5

// patchesUnified is Patches for the unified embedder: the grid with its
// positions added, and then the last of the three norms.
func (v *VisionTower) patchesUnified(pixels []float32, cols, rows int) []float32 {
	cfg, w := v.Cfg, v.W
	out := v.embedPatches(pixels, cols, rows)
	n := cols * rows
	nn.InParallel(n, n*cfg.Dim, func(first, last int) {
		for p := first; p < last; p++ {
			nn.LayerNormGGML(out[p*cfg.Dim:(p+1)*cfg.Dim], w.PosNorm.Gain, w.PosNorm.Bias, patchNormEps)
		}
	})
	return out
}

// embedPatches stops where llama.cpp's pos_embd node is: the patches embedded,
// normalized, and the two lookups added — the norm that follows is applied by
// the caller, and having it apart is what lets a test stand on that node.
func (v *VisionTower) embedPatches(pixels []float32, cols, rows int) []float32 {
	cfg, w := v.Cfg, v.W
	n := cols * rows
	area := cfg.PatchSize * cfg.PatchSize * 3
	flat := cutPatches(pixels, cols, rows, cfg.PatchSize, false)

	// The norm over the raw patch, in place: six thousand values a patch, and
	// the projection below reads them straight after.
	nn.InParallel(n, n*area, func(first, last int) {
		for p := first; p < last; p++ {
			nn.LayerNormGGML(flat[p*area:(p+1)*area], w.PatchNorm.Gain, w.PatchNorm.Bias, patchNormEps)
		}
	})

	// The projection. Four output rows at a time, against the whole grid: the
	// weight is a hundred megabytes and is read once either way, but the grid
	// is three megabytes that no core's own cache holds, and one row at a time
	// would pull it through the shared cache once per row — three thousand
	// eight hundred times for one picture.
	//
	// nn.Scores4 is that product: four rows against every column, two columns
	// at a time, each column loaded once for four multiply-accumulates. It was
	// written for the attention of E2B's tower, where the shape of the problem
	// is the same one — a few queries against a great many keys — and the
	// arithmetic it wants here is the arithmetic ggml uses for this tensor,
	// which is fused.
	//
	// It writes rows, and the grid is held in columns, so a block's four rows
	// land in a staging buffer and are turned as they are copied out. That is
	// four floats written per patch against six thousand nine hundred read,
	// and it does not show.
	out := make([]float32, n*cfg.Dim)
	nn.InParallel(cfg.Dim, n*area*cfg.Dim, func(start, end int) {
		block := make([]float32, 4*n)
		for o := start; o < end; o += 4 {
			rows := min(4, end-o)
			if rows == 4 && nn.Scores4(w.PatchEmbd[o*area:], flat, area, n, block, n) {
				for p := 0; p < n; p++ {
					dst := out[p*cfg.Dim+o:]
					for i := 0; i < 4; i++ {
						dst[i] = block[i*n+p] + w.PatchBias[o+i]
					}
				}
				continue
			}
			for i := 0; i < rows; i++ {
				row := w.PatchEmbd[(o+i)*area : (o+i+1)*area]
				bias := w.PatchBias[o+i]
				for p := 0; p < n; p++ {
					out[p*cfg.Dim+o+i] = nn.DotF32(flat[p*area:(p+1)*area], row) + bias
				}
			}
		}
	})

	nn.InParallel(n, n*cfg.Dim, func(first, last int) {
		for p := first; p < last; p++ {
			nn.LayerNormGGML(out[p*cfg.Dim:(p+1)*cfg.Dim], w.EmbdNorm.Gain, w.EmbdNorm.Bias, patchNormEps)
		}
	})
	v.addPositions(out, cols, n)
	return out
}
