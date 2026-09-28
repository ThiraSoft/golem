//go:build !amd64

package nn

func geluErfAVX2(x *float32, n int) { panic("nn: no GELU kernel on this architecture") }

func expSubAVX2(x *float32, n int, m float32) {
	panic("nn: no exponential kernel on this architecture")
}

func softmaxF32AVX2(x *float32, n int) { panic("nn: no softmax kernel on this architecture") }
