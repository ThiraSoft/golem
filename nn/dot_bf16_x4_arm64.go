//go:build arm64

package nn

// hasBF16x4 says the four-column kernel exists here. On arm64 it always does:
// the widening it rests on is baseline NEON, with no feature to probe for.
const hasBF16x4 = true

//go:noescape
func dotBF16x4NEON(row *uint16, x *float32, stride, n int, out *float32)

// dotBF16x4 computes four dot products of one weight row against four
// activation columns. On arm64 the kernel always applies: the widening it
// depends on is baseline NEON, with no feature to probe for.
func dotBF16x4(row []uint16, x []float32, stride, n int, out *[4]float32) bool {
	var lanes [20]float32
	dotBF16x4NEON(&row[0], &x[0], stride, n, &lanes[0])
	for c := 0; c < 4; c++ {
		out[c] = (lanes[c*4] + lanes[c*4+1]) + (lanes[c*4+2] + lanes[c*4+3]) + lanes[16+c]
	}
	return true
}
