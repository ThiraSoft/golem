//go:build !amd64

package nn

func tilesAvailable() bool { return false }

func matF16Tile(w []uint16, wStride int, x []float32, xStride, n int, out *[12]float32) {
	panic("nn: no tiled fp16 kernel on this architecture")
}

func matF32Tile(w []float32, wStride int, x []float32, xStride, n int, out *[12]float32) {
	panic("nn: no tiled float32 kernel on this architecture")
}
