//go:build !amd64 && !arm64

package nn

// fastDotBF16 declines: without a kernel the caller takes the portable loop.
func fastDotBF16(row *uint16, x *float32, n int) (float32, bool) { return 0, false }
