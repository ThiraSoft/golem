//go:build amd64

package nn

//go:noescape
func dotBF16x4AVX2(row *uint16, x *float32, stride, n int, out *float32)

// dotBF16x4 computes four dot products of one weight row against four
// activation columns, and reports whether it did: without AVX2 the caller
// takes the columns one at a time.
func dotBF16x4(row []uint16, x []float32, stride, n int, out *[4]float32) bool {
	if !avx2 {
		return false
	}
	dotBF16x4AVX2(&row[0], &x[0], stride, n, &out[0])
	return true
}

//go:noescape
func dotBF16x2x4AVX2(row *uint16, rowStride int, x *float32, colStride, n int, out *float32)

//go:noescape
func dotBF16x2x4IlvAVX2(row *uint16, rowStride int, x *float32, colStride, n int, out *float32)

//go:noescape
func prepareIlvAVX2(dst, src *float32, n int, lo, hi float32)

// PrepareIlv writes src into dst clamped to [lo, hi], rounded to bfloat16 and
// laid out for the interleaved kernel. It reports whether it could: the length
// has to be a whole number of sixteens, which every width in this tower is.
func PrepareIlv(dst, src []float32, lo, hi float32) bool {
	if !avx2 || len(src)%16 != 0 || len(src) == 0 {
		return false
	}
	prepareIlvAVX2(&dst[0], &src[0], len(src), lo, hi)
	return true
}

//go:noescape
func clampAVX2(x *float32, n int, lo, hi float32)

// Clamp holds every value of x inside [lo, hi].
func Clamp(x []float32, lo, hi float32) {
	if len(x) == 0 {
		return
	}
	if avx2 {
		clampAVX2(&x[0], len(x), lo, hi)
		return
	}
	for i, v := range x {
		if v < lo {
			x[i] = lo
		} else if v > hi {
			x[i] = hi
		}
	}
}
