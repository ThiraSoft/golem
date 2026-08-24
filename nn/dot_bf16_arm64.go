//go:build arm64

package nn

//go:noescape
func dotBF16LanesNEON(row *uint16, x *float32, n int, out *float32)

// fastDotBF16 folds the kernel's lanes the way fastDotF32 folds its own: lane j
// of every accumulator holds the elements the portable loop puts in s[j], so
// summing down the lanes and then across gives what that loop gives, to a
// rounding.
func fastDotBF16(row *uint16, x *float32, n int) (float32, bool) {
	var lanes [17]float32
	dotBF16LanesNEON(row, x, n, &lanes[0])
	var s [4]float32
	for k := 0; k < 4; k++ {
		for j := 0; j < 4; j++ {
			s[j] += lanes[k*4+j]
		}
	}
	return (s[0] + s[1]) + (s[2] + s[3]) + lanes[16], true
}
