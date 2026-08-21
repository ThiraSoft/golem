package nn

// The Q4_0 product, in integers.
//
// A Q4_0 block is 18 bytes: one fp16 scale, then 32 weights packed two per
// byte. The low nibble of byte j holds weight j, the high nibble holds weight
// j+16 — not j*2 and j*2+1. The stored nibble is the weight plus eight, so it
// is unsigned in 0..15 and recentred on subtraction.
//
// Nothing is dequantized. The nibbles multiply the Q8_0 activation as small
// integers, the sums accumulate exactly, and the two scales are applied once
// per block. That is bit-for-bit what ggml does, which is the point: it makes
// the parity test able to distinguish a bug from a rounding.

import (
	"encoding/binary"
	"math"
)

// halfToFloat converts an IEEE binary16 to a float32.
func halfToFloat(h uint16) float32 {
	sign := uint32(h&0x8000) << 16
	exponent := (h >> 10) & 0x1F
	mantissa := uint32(h & 0x03FF)

	switch exponent {
	case 0:
		if mantissa == 0 {
			return math.Float32frombits(sign)
		}
		// Subnormal: normalize it by hand.
		e := uint32(127 - 15 + 1)
		for mantissa&0x0400 == 0 {
			mantissa <<= 1
			e--
		}
		mantissa &= 0x03FF
		return math.Float32frombits(sign | e<<23 | mantissa<<13)
	case 0x1F:
		return math.Float32frombits(sign | 0xFF<<23 | mantissa<<13)
	}
	return math.Float32frombits(sign | (uint32(exponent)+127-15)<<23 | mantissa<<13)
}

// dotQ4_0Go is the portable form, accumulating into the same state the kernels
// use: eight lanes and one correction. Only the first lane is written — the
// fold that follows adds them all, and this way the answer does not depend on
// which of the two forms computed it.
func dotQ4_0Go(w []byte, b *Batch, first, column, n int, state []float32) {
	var sum, correction float32
	for step := 0; step < n/QuantBlock; step++ {
		bytes := w[step*q4_0BlockBytes : (step+1)*q4_0BlockBytes]
		scale := halfToFloat(binary.LittleEndian.Uint16(bytes))
		nibbles := bytes[2:]
		index := (first+step)*b.Stride + column
		q := b.Q[index*QuantBlock : (index+1)*QuantBlock]

		var accumulator int32
		for j := 0; j < QuantBlock/2; j++ {
			byteValue := nibbles[j]
			low := int32(byteValue & 0x0F)
			high := int32(byteValue >> 4)
			accumulator += low*int32(q[j]) + high*int32(q[j+16])
		}
		sum += float32(accumulator) * scale * b.Scales[index]
		correction += scale * b.Corr[index]
	}
	state[0] += sum
	state[8] += correction
}

// dotQ4_0x4Go is the same for four columns, in the state the kernel keeps: the
// four eight-lane sums, then the four corrections.
func dotQ4_0x4Go(w []byte, b *Batch, first, column, n int, state []float32) {
	var one [9]float32
	for c := 0; c < 4; c++ {
		one[0], one[8] = 0, 0
		dotQ4_0Go(w, b, first, column+c, n, one[:])
		state[c*8] += one[0]
		state[32+c] += one[8]
	}
}

// dotQ4_0x8Go is the same for eight columns, in the state the kernel keeps: the
// eight eight-lane sums, then the eight corrections.
func dotQ4_0x8Go(w []byte, b *Batch, first, column, n int, state []float32) {
	var one [9]float32
	for c := 0; c < 8; c++ {
		one[0], one[8] = 0, 0
		dotQ4_0Go(w, b, first, column+c, n, one[:])
		state[c*8] += one[0]
		state[64+c] += one[8]
	}
}

// Mode says what a kernel does with the sums it is given and the sums it
// leaves: a row cut into stretches of input begins at the first, ends at the
// last, and carries the lanes across the ones between.
type Mode int

const (
	Begin  Mode = 1 // start the sums at zero rather than at the state
	Finish Mode = 2 // fold the lanes into products rather than storing them
)

// fold turns eight accumulated lanes and a correction into the product, in the
// order the kernel folded them when it still folded them itself.
func fold(lanes []float32, correction float32) float32 {
	h0 := lanes[0] + lanes[4]
	h1 := lanes[1] + lanes[5]
	h2 := lanes[2] + lanes[6]
	h3 := lanes[3] + lanes[7]
	return (h0 + h2) + (h1 + h3) - correction
}

// MatVecQ4_0 computes y = W*x for a batch of activations already quantized.
func MatVecQ4_0(w []byte, b *Batch, outputs, inputs int, ys [][]float32) {
	if inputs%QuantBlock != 0 {
		panic("nn: Q4_0 rows must be a multiple of the block size")
	}
	InParallel(outputs, outputs*inputs*b.Size, func(start, end int) {
		matVecQ4_0Rows(w, b, inputs, ys, start, end)
	})
}

// The tiles.
//
// Two working sets meet in this loop: a row of weights, and the quantized
// activations of the columns it is being multiplied by. Whichever of them falls
// out of the first-level cache is fetched again for every step of the other, so
// both are cut down until they fit.
//
// The last stretch of a row is whatever is left, which need not be a whole
// tile: the 12B computes over 3840 and 15360 inputs, neither a multiple of the
// tile. A kernel told to read a whole tile there would read past the row.
//
// kTile cuts the input: a feed-forward output is twelve kilobytes per column,
// so a group of them is far past the cache, and the down projection of this
// model is exactly that shape. Two thousand inputs at a time brings a group
// back under it. rowTile then cuts the output, so that the slice of weights the columns
// sweep over stays put while they sweep.
//
// The cut is decided by the width alone, never by the size of the batch, so
// that a batch of sixty-four sums its products in the same order a batch of one
// does. That is what keeps a prompt read in one pass identical, to the last
// bit, to the same prompt read a token at a time.
const (
	kTile        = 2048
	kTile8       = 2048
	rowTileBytes = 24 * 1024
	maxRowTile   = 64
)

// matVecQ4_0Rows computes rows [start, end) on the caller's thread, eight
// columns of the batch at a time, then four, then what is left.
//
// Eight is where the row stops being what the loop is paying for: unpacking the
// nibbles, converting the fp16 scale and folding the correction happen once per
// block whatever the width, and at eight columns they are a seventh of the
// instructions rather than a third. A batch that is not a multiple of eight
// finishes on the narrower kernels, which sum in the same order.
//
// The order of the loops is the rest of it: a tile of rows on the outside, a
// stretch of inputs next, then the columns, then the rows of the tile. What
// that leaves in the cache is one stretch of a group's activations and the
// slice of weights the tile covers, and both are read from there by everything
// inside.
func matVecQ4_0Rows(w []byte, b *Batch, inputs int, ys [][]float32, start, end int) {
	rowBytes := inputs / QuantBlock * q4_0BlockBytes
	if inputs <= kTile {
		// One stretch: a row begins and ends in the same call, and the kernel
		// folds its own lanes.
		var wide [72]float32
		var state [36]float32
		c := 0
		for ; c+8 <= b.Size; c += 8 {
			for r := start; r < end; r++ {
				dotQ4_0x8(w[r*rowBytes:], b, 0, c, inputs, wide[:], Begin|Finish)
				for j := 0; j < 8; j++ {
					ys[c+j][r] = wide[j]
				}
			}
		}
		for ; c+4 <= b.Size; c += 4 {
			for r := start; r < end; r++ {
				dotQ4_0x4(w[r*rowBytes:], b, 0, c, inputs, state[:], Begin|Finish)
				ys[c][r], ys[c+1][r] = state[0], state[1]
				ys[c+2][r], ys[c+3][r] = state[2], state[3]
			}
		}
		for ; c < b.Size; c++ {
			for r := start; r < end; r++ {
				dotQ4_0(w[r*rowBytes:], b, 0, c, inputs, state[:], Begin|Finish)
				ys[c][r] = state[0]
			}
		}
		return
	}

	steps := kTile
	stepBytes := steps / QuantBlock * q4_0BlockBytes
	rows := min(max(rowTileBytes/stepBytes, 1), maxRowTile)

	var eight [maxRowTile][72]float32
	var four [maxRowTile][36]float32
	var one [maxRowTile][9]float32
	for from := start; from < end; from += rows {
		to := min(from+rows, end)
		c := 0
		for ; c+8 <= b.Size; c += 8 {
			for k := 0; k < inputs; k += steps {
				block, n := k/QuantBlock, min(steps, inputs-k)
				mode := Mode(0)
				if k == 0 {
					mode |= Begin
				}
				if k+n >= inputs {
					mode |= Finish
				}
				for r := from; r < to; r++ {
					dotQ4_0x8(w[r*rowBytes+block*q4_0BlockBytes:], b, block, c, n, eight[r-from][:], mode)
				}
			}
			for r := from; r < to; r++ {
				state := eight[r-from]
				for j := 0; j < 8; j++ {
					ys[c+j][r] = state[j]
				}
			}
		}
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
				for r := from; r < to; r++ {
					dotQ4_0x4(w[r*rowBytes+block*q4_0BlockBytes:], b, block, c, n, four[r-from][:], mode)
				}
			}
			for r := from; r < to; r++ {
				state := four[r-from]
				ys[c][r], ys[c+1][r] = state[0], state[1]
				ys[c+2][r], ys[c+3][r] = state[2], state[3]
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
				for r := from; r < to; r++ {
					dotQ4_0(w[r*rowBytes+block*q4_0BlockBytes:], b, block, c, n, one[r-from][:], mode)
				}
			}
			for r := from; r < to; r++ {
				ys[c][r] = one[r-from][0]
			}
		}
	}
}
