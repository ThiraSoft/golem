//go:build amd64

package nn

//go:noescape
func dotF32HalfAVX2(a *float32, b *uint16, n int) float32

//go:noescape
func axpyHalfAVX2(dst *float32, src *uint16, n int, a float32)
