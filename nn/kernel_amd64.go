//go:build amd64

package nn

func dotBF16AVX2(row *uint16, x *float32, n int) float32

func hasAVX2() bool

var avx2 = hasAVX2()

func axpyAVX2(dst, src *float32, n int, a float32)

func dotF32AVX2(a, b *float32, n int) float32
