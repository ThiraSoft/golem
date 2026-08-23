//go:build arm64

package nn

//go:noescape
func dotF32LanesNEON(a, b *float32, n int, out *float32)

//go:noescape
func axpyNEON(dst, src *float32, n int, a float32)

// fastDotF32 is the NEON dot product. The kernel leaves four accumulators of
// four lanes; lane j of each holds the elements the portable loop puts in s[j],
// so folding them this way keeps the two forms within a rounding of each other.
func fastDotF32(a, b *float32, n int) (float32, bool) {
	var lanes [17]float32
	dotF32LanesNEON(a, b, n, &lanes[0])
	var s [4]float32
	for k := 0; k < 4; k++ {
		for j := 0; j < 4; j++ {
			s[j] += lanes[k*4+j]
		}
	}
	return (s[0] + s[1]) + (s[2] + s[3]) + lanes[16], true
}

func fastAxpy(dst, src *float32, n int, a float32) bool {
	axpyNEON(dst, src, n, a)
	return true
}

//go:noescape
func dotF32HalfLanesNEON(a *float32, b *uint16, n int, out *float32)

// fastDotF32Half folds its lanes exactly as fastDotF32 does, which is what
// makes the two agree to the bit — see dot_half_arm64.s for why that matters.
func fastDotF32Half(a *float32, b *uint16, n int) (float32, bool) {
	var lanes [17]float32
	dotF32HalfLanesNEON(a, b, n, &lanes[0])
	var s [4]float32
	for k := 0; k < 4; k++ {
		for j := 0; j < 4; j++ {
			s[j] += lanes[k*4+j]
		}
	}
	return (s[0] + s[1]) + (s[2] + s[3]) + lanes[16], true
}
