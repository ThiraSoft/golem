//go:build !amd64

package nn

func dotF32HalfAVX2(a *float32, b *uint16, n int) float32 { panic("unavailable") }

func axpyHalfAVX2(dst *float32, src *uint16, n int, a float32) { panic("unavailable") }
