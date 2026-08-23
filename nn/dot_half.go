package nn

// Products against a cache kept in fp16.
//
// The key-value cache is the one thing in the engine that is written once and
// read on every token afterwards, and llama.cpp keeps it in fp16. Golem rounded
// its entries through fp16 and then stored the result in a float32 — four bytes
// carrying two bytes of information, and four bytes crossing the memory bus on
// every read. These two kernels let the cache hold the sixteen bits it actually
// has.
//
// The conversion is not free, but it is paid in the currency the machine has
// spare: at one position per token the attention loop waits on memory, and the
// arithmetic to widen a half runs in that shadow. What halves is the traffic.

// DotF32Half returns the dot product of a float32 vector with an fp16 one.
func DotF32Half(a []float32, b []uint16) float32 {
	n := min(len(a), len(b))
	if n == 0 {
		return 0
	}
	if avx2 {
		return dotF32HalfAVX2(&a[0], &b[0], n)
	}
	var s0, s1, s2, s3 float32
	i := 0
	for ; i+3 < n; i += 4 {
		s0 += a[i] * halfToFloat(b[i])
		s1 += a[i+1] * halfToFloat(b[i+1])
		s2 += a[i+2] * halfToFloat(b[i+2])
		s3 += a[i+3] * halfToFloat(b[i+3])
	}
	for ; i < n; i++ {
		s0 += a[i] * halfToFloat(b[i])
	}
	return (s0 + s1) + (s2 + s3)
}

// AxpyHalf adds a times an fp16 vector into a float32 one.
func AxpyHalf(dst []float32, src []uint16, a float32) {
	n := min(len(dst), len(src))
	if n == 0 {
		return
	}
	if avx2 {
		axpyHalfAVX2(&dst[0], &src[0], n, a)
		return
	}
	for i := 0; i < n; i++ {
		dst[i] += a * halfToFloat(src[i])
	}
}

// Half rounds a float32 to the nearest fp16, and Widen puts one back. They are
// the cache's two ends, and they are the same pair RoundHalf is built from.
func Half(v float32) uint16  { return floatToHalf(v) }
func Widen(h uint16) float32 { return halfToFloat(h) }
