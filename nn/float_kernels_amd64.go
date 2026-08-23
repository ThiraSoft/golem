//go:build amd64

package nn

func fastDotF32(a, b *float32, n int) (float32, bool) {
	if !avx2 {
		return 0, false
	}
	return dotF32AVX2(a, b, n), true
}

func fastAxpy(dst, src *float32, n int, a float32) bool {
	if !avx2 {
		return false
	}
	axpyAVX2(dst, src, n, a)
	return true
}

func fastDotF32Half(a *float32, b *uint16, n int) (float32, bool) {
	if !avx2 {
		return 0, false
	}
	return dotF32HalfAVX2(a, b, n), true
}

func fastAxpyHalf(dst *float32, src *uint16, n int, a float32) bool {
	if !avx2 {
		return false
	}
	axpyHalfAVX2(dst, src, n, a)
	return true
}
