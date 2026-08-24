//go:build arm64

package nn

// Q4_0 weights, laid out four rows at a time.
//
// The x86 layout beside this one interleaves eight rows, because eight int32
// sums fill one AVX2 register and because its lowRows/highRows split undoes the
// lane crossing of a single VPHADDD. Neither reason survives on NEON: a
// register holds four int32, and SDOT reduces horizontally inside its own four
// lanes, so there is no crossing to undo and the split would have to be undone
// per group instead.
//
// Four rows is also what llama.cpp reaches for here, and it did the measuring
// this port cannot. ggml-cpu/repack.cpp picks the eight-row form for AVX2 or
// for 256-bit SVE with i8mm, the four-row form for NEON — 4x8 where i8mm is
// available and 4x4 where only the dot product is. Both of those call the same
// repack and differ in the activation block width, so the layout below is the
// one an i8mm kernel would want too: moving to SMMLA later is a kernel change,
// not a change to the weights.
//
// The shape of one group, for a block of thirty-two inputs:
//
//	8 bytes   the four rows' fp16 scales, row 0 first
//	64 bytes  four chunks of sixteen bytes, each covering eight inputs
//
// Chunk c covers inputs 8c..8c+7. Its byte r*4+j carries two weights: the low
// nibble is row r at input 8c+j, the high nibble is row r at input 8c+4+j.
//
// That is exactly what one SDOT wants. The low nibbles of a chunk read as
// [row0: inputs 0..3][row1: 0..3][row2][row3], so against the four activations
// 8c..8c+3 duplicated across the vector, each of SDOT's four lanes accumulates
// one row — and the high nibbles do the same for 8c+4..8c+7. No shuffle, no
// horizontal add, and the rows come out in order.
//
// A group is exactly as large as the four rows it replaces: 4 x 18 = 72.

import "encoding/binary"

const (
	// PackedRows is how many rows one packed group interleaves.
	PackedRows = 4
	// packedBlockBytes is one block of a group: four scales, then the nibbles
	// of four rows.
	packedBlockBytes = 8 + 64
)

// packedRowOrder is which lane a row lands in. SDOT puts row r in lane r by
// construction here, so it is the identity, and the tests hold it to that.
func packedRowOrder(row int) int { return row }

// PackedQ4_0Bytes is what the packed form of a matrix occupies. Rows past the
// last whole group are not packed: they keep the file's layout and the kernels
// that read it.
func PackedQ4_0Bytes(rows, cols int) int {
	return rows / PackedRows * (cols / QuantBlock) * packedBlockBytes
}

// PackQ4_0 writes the packed form of the first whole groups of rows into out,
// which must hold PackedQ4_0Bytes(rows, cols).
//
// A source block holds input j in the low nibble of byte j and input j+16 in
// the high one, so each row is spread into its thirty-two weights first and the
// chunks are assembled from that: the alternative reads every byte four times
// and decides each time which half it wants, and this runs over every weight in
// the model.
func PackQ4_0(w []byte, rows, cols int, out []byte) {
	if cols%QuantBlock != 0 {
		panic("nn: Q4_0 rows must be a multiple of the block size")
	}
	blocks := cols / QuantBlock
	rowBytes := blocks * q4_0BlockBytes

	for group := 0; group < rows/PackedRows; group++ {
		base := group * PackedRows
		into := out[group*blocks*packedBlockBytes:]
		var spread [PackedRows][QuantBlock]byte
		for b := 0; b < blocks; b++ {
			dst := into[b*packedBlockBytes : (b+1)*packedBlockBytes]
			for r := 0; r < PackedRows; r++ {
				src := w[(base+r)*rowBytes+b*q4_0BlockBytes:]
				copy(dst[r*2:], src[:2]) // the row's scale, in its lane
				for j := 0; j < QuantBlock/2; j++ {
					spread[r][j] = src[2+j] & 0x0F
					spread[r][j+16] = src[2+j] >> 4
				}
			}
			nibbles := dst[8:]
			for c := 0; c < 4; c++ {
				chunk := nibbles[c*16 : (c+1)*16]
				for r := 0; r < PackedRows; r++ {
					for j := 0; j < 4; j++ {
						chunk[r*4+j] = spread[r][c*8+j] | spread[r][c*8+4+j]<<4
					}
				}
			}
		}
	}
}

// dotPackedQ4_0Go is the portable form of the packed product: one group of four
// rows against one column, over the n inputs beginning at the given block,
// accumulated into four lanes — one per row.
//
// The integers are summed exactly, so the order the kernel adds them in does
// not show; the floats are not, which is why this and the assembly agree to
// rounding rather than to the bit.
func dotPackedQ4_0Go(w []byte, b *Batch, block, column, n int, state []float32) {
	for step := 0; step < n/QuantBlock; step++ {
		group := w[step*packedBlockBytes : (step+1)*packedBlockBytes]
		index := (block+step)*b.Stride + column
		q := b.Q[index*QuantBlock : (index+1)*QuantBlock]
		activation, correction := b.Scales[index], b.Corr[index]

		var sums [PackedRows]int32
		nibbles := group[8:]
		for c := 0; c < 4; c++ {
			chunk := nibbles[c*16 : (c+1)*16]
			for r := 0; r < PackedRows; r++ {
				for j := 0; j < 4; j++ {
					byteValue := chunk[r*4+j]
					sums[r] += int32(byteValue&0x0F) * int32(q[c*8+j])
					sums[r] += int32(byteValue>>4) * int32(q[c*8+4+j])
				}
			}
		}
		for r := 0; r < PackedRows; r++ {
			scale := halfToFloat(binary.LittleEndian.Uint16(group[r*2:]))
			state[r] += float32(sums[r]) * (scale * activation)
			state[r] -= scale * correction
		}
	}
}
