package nn

// Products for a wide batch, tiled: an fp16 matrix against float32 columns, and
// float32 against float32 for an encoder's attention.
//
// The F16 case of Matrix.rows reads a row, widens it, and takes one dot
// product a column, then widens the same row again for the next column. That
// is the right shape for a projector that sees a few hundred patches once an
// image, and it was 74 % of an encoder's time: nomic-embed-text reads every
// weight of the model once for every position of every text it is handed, and
// it is handed thousands at a time. So these read four rows against three
// columns in registers and keep them there, which is the shape of llama.cpp's
// own kernels on the same machine.
//
// They are separate entry points rather than cases of rows, because they sum
// in another order and fuse the multiply-add: gemma's projector is held to its
// fixtures by the order rows has, and moving it for a speed it does not need
// would move those.
//
// Every entry comes out of the tile kernel, the ragged edges included: a column
// left over past the last three is read with a column stride of zero, which
// makes the kernel's three columns one, and a row left over past the last four
// with a row stride of zero. Each accumulator of the kernel depends on its own
// row and column alone, so an entry is the same float whichever tile it lands
// in. That is what keeps a text's embedding from depending on which other
// texts shared its pass: with the edges on DotF32Half, which sums in another
// order, it moved in the fourth digit.

import "unsafe"

// TiledProducts says whether this machine has the tiled kernels. Without them
// MatMulF16 declines and GemmF32NT takes a dot at a time.
func TiledProducts() bool { return tilesAvailable() }

// MatMulF16 computes ys[c] = W·x[c] + bias for an fp16 matrix, where x holds
// len(ys) columns of Cols floats back to back. It reports false, having done
// nothing, when the machine has no kernel for it or the width is not a whole
// number of eights; the caller then takes MatVecBatch.
//
// The floats are read as they are. ggml rounds an activation to fp16 before an
// fp16 product, and so should the caller if it wants ggml's answer.
func (m Matrix) MatMulF16(x []float32, ys [][]float32) bool {
	cols := len(ys)
	if m.Quant != F16 || m.Cols%8 != 0 || cols == 0 || !tilesAvailable() {
		return false
	}
	if len(x) < cols*m.Cols {
		panic("nn: MatMulF16 was handed fewer floats than its columns hold")
	}
	weights := unsafe.Slice((*uint16)(unsafe.Pointer(&m.Data[0])), len(m.Data)/2)
	groups := (m.Rows + 3) / 4
	InParallel(groups, m.Rows*m.Cols*cols, func(first, last int) {
		m.f16Rows(weights, x, cols, ys, first*4, min(last*4, m.Rows))
	})
	return true
}

// f16Rows computes rows [start, end) for every column. The columns are taken
// in blocks that fit the second-level cache, so that the rows a worker owns
// are read once per block rather than once per column.
func (m Matrix) f16Rows(w []uint16, x []float32, cols int, ys [][]float32, start, end int) {
	n := m.Cols
	block := max(3, (128<<10)/(4*n)/3*3)
	var tile [12]float32
	for c0 := 0; c0 < cols; c0 += block {
		c1 := min(c0+block, cols)
		for r := start; r < end; r += 4 {
			rows := min(4, end-r)
			for c := c0; c < c1; c += 3 {
				width := min(3, c1-c)
				switch {
				case rows == 4 && width == 3:
					matF16Tile(w[r*n:], n, x[c*n:], n, n, &tile)
					for i := 0; i < 4; i++ {
						for j := 0; j < 3; j++ {
							ys[c+j][r+i] = tile[i*3+j]
						}
					}
				case rows == 4:
					for j := 0; j < width; j++ {
						matF16Tile(w[r*n:], n, x[(c+j)*n:], 0, n, &tile)
						for i := 0; i < 4; i++ {
							ys[c+j][r+i] = tile[i*3]
						}
					}
				case width == 3:
					for i := 0; i < rows; i++ {
						matF16Tile(w[(r+i)*n:], 0, x[c*n:], n, n, &tile)
						for j := 0; j < 3; j++ {
							ys[c+j][r+i] = tile[j]
						}
					}
				default:
					for i := 0; i < rows; i++ {
						for j := 0; j < width; j++ {
							matF16Tile(w[(r+i)*n:], 0, x[(c+j)*n:], 0, n, &tile)
							ys[c+j][r+i] = tile[0]
						}
					}
				}
			}
		}
	}
	if m.Bias != nil {
		for c := 0; c < cols; c++ {
			for r := start; r < end; r++ {
				ys[c][r] += m.Bias[r]
			}
		}
	}
}

// GemmF32NT computes out[r*outStride+c] = Σ_k w[r*wStride+k]·x[c*xStride+k]
// over k < n, for r < rows and c < cols, on the caller's thread: every entry
// is the dot product of a row of w with a row of x, which is the shape both
// halves of an attention have once the values are transposed.
//
// The strides let a caller read heads out of a fused projection in place. The
// slices have to reach the last row they are asked for; that is the caller's to
// guarantee, as it is for every kernel here.
func GemmF32NT(w []float32, wStride int, x []float32, xStride, n, rows, cols int, out []float32, outStride int) {
	if n <= 0 || n%8 != 0 || !tilesAvailable() {
		for r := 0; r < rows; r++ {
			for c := 0; c < cols; c++ {
				out[r*outStride+c] = DotF32(w[r*wStride:r*wStride+n], x[c*xStride:c*xStride+n])
			}
		}
		return
	}
	var tile [12]float32
	for r := 0; r < rows; r += 4 {
		rs := min(4, rows-r)
		for c := 0; c < cols; c += 3 {
			cs := min(3, cols-c)
			switch {
			case rs == 4 && cs == 3:
				matF32Tile(w[r*wStride:], wStride, x[c*xStride:], xStride, n, &tile)
				for i := 0; i < 4; i++ {
					for j := 0; j < 3; j++ {
						out[(r+i)*outStride+c+j] = tile[i*3+j]
					}
				}
			case rs == 4:
				for j := 0; j < cs; j++ {
					matF32Tile(w[r*wStride:], wStride, x[(c+j)*xStride:], 0, n, &tile)
					for i := 0; i < 4; i++ {
						out[(r+i)*outStride+c+j] = tile[i*3]
					}
				}
			case cs == 3:
				for i := 0; i < rs; i++ {
					matF32Tile(w[(r+i)*wStride:], 0, x[c*xStride:], xStride, n, &tile)
					for j := 0; j < 3; j++ {
						out[(r+i)*outStride+c+j] = tile[j]
					}
				}
			default:
				for i := 0; i < rs; i++ {
					for j := 0; j < cs; j++ {
						matF32Tile(w[(r+i)*wStride:], 0, x[(c+j)*xStride:], 0, n, &tile)
						out[(r+i)*outStride+c+j] = tile[0]
					}
				}
			}
		}
	}
}
