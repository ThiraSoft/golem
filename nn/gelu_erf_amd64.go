//go:build amd64

package nn

//go:noescape
func geluErfAVX2(x *float32, n int)

//go:noescape
func expSubAVX2(x *float32, n int, m float32)

//go:noescape
func softmaxF32AVX2(x *float32, n int)
