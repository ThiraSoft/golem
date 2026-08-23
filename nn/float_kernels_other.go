//go:build !amd64 && !arm64

package nn

// No kernels here, so every caller takes its portable loop.

func fastDotF32(a, b *float32, n int) (float32, bool) { return 0, false }

func fastAxpy(dst, src *float32, n int, a float32) bool { return false }

func fastDotF32Half(a *float32, b *uint16, n int) (float32, bool) { return 0, false }
