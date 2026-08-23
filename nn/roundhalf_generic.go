//go:build !amd64

package nn

func roundHalfAVX2(v *float32, n int) { panic("unavailable") }
