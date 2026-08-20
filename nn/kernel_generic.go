//go:build !amd64

package nn

const avx2 = false

func dotBF16AVX2(row *uint16, x *float32, n int) float32 { panic("unavailable") }

func axpyAVX2(dst, src *float32, n int, a float32) { panic("unavailable") }

func dotF32AVX2(a, b *float32, n int) float32 { panic("unavailable") }
