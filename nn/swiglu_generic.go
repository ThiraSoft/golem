//go:build !amd64

package nn

// The activation kernels of nn/softmax_amd64.s, which nn/swiglu.go reaches for
// when avx2 is true. It never is off amd64, so these exist to be compiled and
// not to be called: Go type-checks a dead branch like any other.

func gegluQuickAVX2(gate, up *float32, n int) { panic("unavailable") }

func gatedSigmoidAVX2(dst, num, arg *float32, n int) { panic("unavailable") }
