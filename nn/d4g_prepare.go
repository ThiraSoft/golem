package nn

// What the activations go through before they meet D4G weights.
//
// Two things happen to a matrix before it is quantized, and neither can be
// stored in the weights: its columns are scaled by how much signal the
// activations actually put through them, and each group of columns is rotated
// so that no coefficient in the group is an outlier. Both are undone on the
// other operand at inference — y = Wx = (W·diag(s)·Aᵀ)(A·(x/s)) — which costs
// one pass over the activation and nothing over the weights.
//
// The sign flips the rotation begins with are elementwise too, so they are
// folded into the same vector: Pre carries ±1/sⱼ, one fp16 a column, and the
// transform that follows it is the plain Walsh–Hadamard.

import "math"

// PrepareD4G applies a matrix's Pre vector to one activation and then the
// normalised Walsh–Hadamard transform over each group of `group` values, in
// place. group must be a power of two dividing len(x), or zero to leave the
// activation unrotated.
func PrepareD4G(x, pre []float32, group int) {
	for i := range x {
		x[i] *= pre[i]
	}
	if group <= 1 {
		return
	}
	inv := float32(1 / math.Sqrt(float64(group)))
	for base := 0; base+group <= len(x); base += group {
		blk := x[base : base+group]
		for l := 1; l < group; l <<= 1 {
			for i := 0; i < group; i += l << 1 {
				for j := i; j < i+l; j++ {
					a, b := blk[j], blk[j+l]
					blk[j], blk[j+l] = a+b, a-b
				}
			}
		}
		for i := range blk {
			blk[i] *= inv
		}
	}
}

// UnprepareD4G recovers a weight row from its rotated, scaled form: it is
// PrepareD4G run backwards, and the order matters. The normalised transform is
// its own inverse, so the rotation is undone by applying it again — but the
// scaling was applied first, so it must be undone last. pre here is the
// activation-side vector, the reciprocal of what the weights met, which is
// exactly the factor a row has to be multiplied by to come back.
//
// This is what a tied head costs the input path: the embedding table is stored
// the way the logit product wants it, and reading one token's row means one
// transform of the model's width. That is nothing beside the row itself.
func UnprepareD4G(x, pre []float32, group int) {
	if group > 1 {
		inv := float32(1 / math.Sqrt(float64(group)))
		for base := 0; base+group <= len(x); base += group {
			blk := x[base : base+group]
			for l := 1; l < group; l <<= 1 {
				for i := 0; i < group; i += l << 1 {
					for j := i; j < i+l; j++ {
						a, b := blk[j], blk[j+l]
						blk[j], blk[j+l] = a+b, a-b
					}
				}
			}
			for i := range blk {
				blk[i] *= inv
			}
		}
	}
	for i := range x {
		x[i] *= pre[i]
	}
}
