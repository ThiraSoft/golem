package nn

// Q4_0 weights, laid out eight rows at a time.
//
// The file's own layout is a row at a time, and a kernel reading it pays for
// the row once per column of the batch: the nibbles are unpacked, the fp16
// scale is converted, and the block's integer sum is turned into a float and
// scaled — three of the four instructions a column of a block costs. Widening
// the group of columns amortized the first two. The third cannot be amortized
// that way, because the scale belongs to the row and the sum belongs to the
// pair: one conversion, one multiply and one add per row, per column, per
// block, whatever else the loop does.
//
// Unless the rows arrive in the lanes of one vector. Eight rows interleaved so
// that a block's eight sums come out of the kernel as eight lanes of a single
// register turn that third cost into one conversion and one fused multiply for
// the eight of them. That is what this layout is for, and it is what ggml
// repacks Q4_0 into before it reads a prompt.
//
// The price is a second copy of the weights: the mapped file keeps the layout
// the file has, and the packed form is built beside it at load time. Nothing
// else about the arithmetic changes — the same nibbles meet the same
// activations, and the blocks are read in the same order.
//
// The shape of one group, for a block of thirty-two inputs:
//
//	16 bytes  the eight rows' fp16 scales, row 0 first
//	128 bytes four chunks of eight inputs, each holding eight bytes per slot
//
// A chunk covers inputs 8g..8g+7. Its byte s*8+j carries two weights: the low
// nibble is row lowRows[s] at input 8g+j, the high nibble is row highRows[s].
// The split is what puts the rows in ascending lanes after the kernel's one
// horizontal add — see packedRowOrder.
//
// A group is exactly as large as the eight rows it replaces: 8 x 18 = 144.

import "encoding/binary"

const (
	// PackedRows is how many rows one packed group interleaves.
	PackedRows = 8
	// packedBlockBytes is one block of a group: eight scales, then the
	// nibbles of eight rows.
	packedBlockBytes = 16 + 128
)

// The rows a chunk's low and high nibbles carry, in slot order. The kernel
// folds its two accumulators with one VPHADDD, which interleaves them as
// [low0, low1, high0, high1, low2, low3, high2, high3]; laying the rows out
// this way is what makes that come out as rows zero to seven, in order, so
// that the scales can be read as they are stored and the results written as
// they are folded.
var (
	lowRows  = [4]int{0, 1, 4, 5}
	highRows = [4]int{2, 3, 6, 7}
)

// packedRowOrder is the inverse: which lane a row lands in. It is the identity,
// by construction, and the tests hold it to that.
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
			nibbles := dst[16:]
			for g := 0; g < 4; g++ {
				chunk := nibbles[g*32 : (g+1)*32]
				for s := 0; s < 4; s++ {
					low, high := &spread[lowRows[s]], &spread[highRows[s]]
					for j := 0; j < 8; j++ {
						chunk[s*8+j] = low[g*8+j] | high[g*8+j]<<4
					}
				}
			}
		}
	}
}

// dotPackedQ4_0Go is the portable form of the packed product: one group of
// eight rows against one column, over the n inputs beginning at the given
// block, accumulated into eight lanes — one per row.
//
// The integers are summed exactly, so the order the kernel adds them in does
// not show; the floats are not, which is why this and the assembly agree to
// rounding rather than to the bit. Everything the engines run goes through the
// assembly.
func dotPackedQ4_0Go(w []byte, b *Batch, block, column, n int, state []float32) {
	for step := 0; step < n/QuantBlock; step++ {
		group := w[step*packedBlockBytes : (step+1)*packedBlockBytes]
		index := (block+step)*b.Stride + column
		q := b.Q[index*QuantBlock : (index+1)*QuantBlock]
		activation, correction := b.Scales[index], b.Corr[index]

		var sums [PackedRows]int32
		nibbles := group[16:]
		for g := 0; g < 4; g++ {
			chunk := nibbles[g*32 : (g+1)*32]
			for s := 0; s < 4; s++ {
				for j := 0; j < 8; j++ {
					value := int32(q[g*8+j])
					byteValue := chunk[s*8+j]
					sums[packedRowOrder(lowRows[s])] += int32(byteValue&0x0F) * value
					sums[packedRowOrder(highRows[s])] += int32(byteValue>>4) * value
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

// maxGroupTile is the tile of the outer loop, in groups of eight rows. Its
// shape is the one matVecQ4_0Rows uses, for the same reason: a stretch of the
// columns' activations and the slice of weights the tile covers both have to
// stay in the first-level cache while the loop runs.
const maxGroupTile = 8

// matVecPackedQ4_0Rows computes rows [start, end) from the packed weights, on
// the caller's thread. A group whose rows are not all wanted is computed whole
// and written in part: the row range comes from however the section was split,
// and it costs less to recompute eight rows than to make every caller split on
// them.
func matVecPackedQ4_0Rows(w []byte, b *Batch, inputs int, ys [][]float32, start, end int) {
	groupBytes := inputs / QuantBlock * packedBlockBytes
	first, last := start/PackedRows, (end+PackedRows-1)/PackedRows

	steps := min(kTile, inputs)
	stepBytes := steps / QuantBlock * packedBlockBytes
	tile := min(max(rowTileBytes/stepBytes, 1), maxGroupTile)

	var four [maxGroupTile][4 * PackedRows]float32
	var one [maxGroupTile][PackedRows]float32

	write := func(g, column int, values []float32) {
		row := g * PackedRows
		for r := 0; r < PackedRows; r++ {
			if row+r >= start && row+r < end {
				ys[column][row+r] = values[r]
			}
		}
	}

	for from := first; from < last; from += tile {
		to := min(from+tile, last)
		c := 0
		for ; c+4 <= b.Size; c += 4 {
			for k := 0; k < inputs; k += steps {
				block, n := k/QuantBlock, min(steps, inputs-k)
				mode := Mode(0)
				if k == 0 {
					mode |= Begin
				}
				if k+n >= inputs {
					mode |= Finish
				}
				for g := from; g < to; g++ {
					dotPackedQ4_0x4(w[g*groupBytes+block*packedBlockBytes:], b, block, c, n, four[g-from][:], mode)
				}
			}
			for g := from; g < to; g++ {
				for j := 0; j < 4; j++ {
					write(g, c+j, four[g-from][j*PackedRows:])
				}
			}
		}
		for ; c < b.Size; c++ {
			for k := 0; k < inputs; k += steps {
				block, n := k/QuantBlock, min(steps, inputs-k)
				mode := Mode(0)
				if k == 0 {
					mode |= Begin
				}
				if k+n >= inputs {
					mode |= Finish
				}
				for g := from; g < to; g++ {
					dotPackedQ4_0(w[g*groupBytes+block*packedBlockBytes:], b, block, c, n, one[g-from][:], mode)
				}
			}
			for g := from; g < to; g++ {
				write(g, c, one[g-from][:])
			}
		}
	}
}
