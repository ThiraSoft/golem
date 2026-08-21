package nn

// A weight matrix, whatever format it is stored in.
//
// The switch on the format happens once per matrix rather than once per
// element, so it costs nothing measurable and each format keeps a kernel
// written for it alone. The bytes are the ones in the mapped file: nothing is
// dequantized at load time, because re-reading the whole matrix on every token
// is what caps generation speed, and a wider format would mean more bandwidth.
//
// Repack is the one thing built at load time, and it is not a wider format: it
// is the same Q4_0 weights, byte for byte as many, laid out eight rows at a
// time so that the product can put a row in a lane. nn/pack_q4_0.go says what
// it buys and what it costs.

import (
	"encoding/binary"
	"fmt"
	"math"
	"unsafe"
)

type Matrix struct {
	Data       []byte
	Quant      Quant
	Rows, Cols int       // Rows outputs, each reading Cols inputs
	Bias       []float32 // nil when the projection has none

	// Packed is the same weights, eight rows interleaved, built by Repack.
	// It is what the products read when it is there; the rows past the last
	// whole group of eight are not in it and keep reading Data.
	Packed []byte
}

// Repack builds the interleaved form of a Q4_0 matrix, which nn/pack_q4_0.go
// describes. It costs a second copy of the weights in memory and gives back
// about a third of the time a prompt spends in the product. A matrix in any
// other format, or too narrow to hold one group of eight rows, is left as it
// is.
func (m *Matrix) Repack() {
	if m.Quant != Q4_0 || m.Packed != nil || m.Rows < PackedRows || m.Cols%QuantBlock != 0 {
		return
	}
	m.Packed = make([]byte, PackedQ4_0Bytes(m.Rows, m.Cols))
	PackQ4_0(m.Data, m.Rows, m.Cols, m.Packed)
}

// RowBytes is what one row occupies on disk.
func (m Matrix) RowBytes() int {
	switch m.Quant {
	case F32:
		return m.Cols * 4
	case BF16:
		return m.Cols * 2
	case Q4_0:
		return m.Cols / QuantBlock * q4_0BlockBytes
	case Q6_K:
		return m.Cols / SuperBlock * q6_kBlockBytes
	}
	panic(fmt.Sprintf("nn: no row size for %s", m.Quant))
}

// MatVec computes y = W*x + bias. y holds Rows elements, and x must already
// carry the quantized form the weights ask for.
func (m Matrix) MatVec(x *Batch, y []float32) {
	m.MatVecBatch(x, [][]float32{y})
}

// MatVecBatch computes one product per activation, reading each row of weights
// once for the whole batch.
//
// This is the difference between reading a prompt and generating an answer. A
// token being generated reads a gigabyte of weights to produce one column;
// sixty-four tokens of a prompt need the same gigabyte, and reading it
// sixty-four times is sixty-four times the memory traffic for the same
// arithmetic. So the row is the outer loop and the batch the inner one: the row
// stays in the first-level cache while every column of the batch meets it.
func (m Matrix) MatVecBatch(b *Batch, ys [][]float32) {
	InParallel(m.Rows, m.Rows*m.Cols*b.Size, func(start, end int) {
		m.rows(b, ys, start, end)
	})
}

// MatVecRows computes rows [start, end) of the product on the caller's thread,
// for a caller that is already inside a parallel section and wants to finish
// what it produced before the section ends.
func (m Matrix) MatVecRows(b *Batch, ys [][]float32, start, end int) {
	m.rows(b, ys, start, end)
}

// rows computes rows [start, end) of the product for every activation in xs, on
// the caller's thread. The format is decided once for the whole call, and the
// batch is the innermost loop so that a row is read from memory once.
func (m Matrix) rows(b *Batch, ys [][]float32, start, end int) {
	switch m.Quant {
	case Q4_0:
		if packed := m.Rows / PackedRows * PackedRows; m.Packed != nil && start < packed {
			matVecPackedQ4_0Rows(m.Packed, b, m.Cols, ys, start, min(end, packed))
			if end > packed {
				matVecQ4_0Rows(m.Data, b, m.Cols, ys, packed, end)
			}
		} else {
			matVecQ4_0Rows(m.Data, b, m.Cols, ys, start, end)
		}
	case BF16:
		weights := unsafe.Slice((*uint16)(unsafe.Pointer(&m.Data[0])), len(m.Data)/2)
		for r := start; r < end; r++ {
			row := weights[r*m.Cols : (r+1)*m.Cols]
			for c := 0; c < b.Size; c++ {
				ys[c][r] = dotBF16(row, b.F[c])
			}
		}
	case F32:
		weights := unsafe.Slice((*float32)(unsafe.Pointer(&m.Data[0])), len(m.Data)/4)
		for r := start; r < end; r++ {
			row := weights[r*m.Cols : (r+1)*m.Cols]
			for c := 0; c < b.Size; c++ {
				ys[c][r] = DotF32(row, b.F[c])
			}
		}
	case Q6_K:
		matVecQ6_KRows(m.Data, b, m.Cols, ys, start, end)
	default:
		panic(fmt.Sprintf("nn: %s is not a matrix format", m.Quant))
	}
	for _, y := range ys {
		for i := start; i < end && i < len(m.Bias); i++ {
			y[i] += m.Bias[i]
		}
	}
}

// Row expands one row into out, which holds Cols floats. This is how embedding
// tables are read: one row per token, never a product.
func (m Matrix) Row(index int, out []float32) {
	if index < 0 || index >= m.Rows {
		panic(fmt.Sprintf("nn: row %d out of %d", index, m.Rows))
	}
	stride := m.RowBytes()
	row := m.Data[index*stride : (index+1)*stride]

	switch m.Quant {
	case Q6_K:
		DequantizeQ6_K(row, m.Cols, out)
	case F32:
		for i := 0; i < m.Cols; i++ {
			out[i] = float32FromBytes(row[i*4:])
		}
	case BF16:
		for i := 0; i < m.Cols; i++ {
			out[i] = bf16ToFloat(row[i*2:])
		}
	case Q4_0:
		dequantizeQ4_0Row(row, m.Cols, out)
	}
}

// The small readers the Row path uses. None of them is on a hot path: an
// embedding table is read one row per token, and a float32 matrix is rare
// enough in a quantized model that a plain loop is the right amount of effort.

// float32FromBytes reads one little-endian float32.
func float32FromBytes(b []byte) float32 {
	return math.Float32frombits(binary.LittleEndian.Uint32(b))
}

// bf16ToFloat reads one bfloat16, which is a float32 with its low mantissa
// dropped — the conversion is a shift, as nn/kernel.go explains.
func bf16ToFloat(b []byte) float32 {
	return math.Float32frombits(uint32(binary.LittleEndian.Uint16(b)) << 16)
}

// dequantizeQ4_0Row expands one Q4_0 row: the block loop of dotQ4_0Go, without
// the activation.
func dequantizeQ4_0Row(w []byte, n int, out []float32) {
	if n%QuantBlock != 0 {
		panic("nn: Q4_0 rows must be a multiple of the block size")
	}
	for b := 0; b < n/QuantBlock; b++ {
		block := w[b*q4_0BlockBytes : (b+1)*q4_0BlockBytes]
		scale := halfToFloat(binary.LittleEndian.Uint16(block))
		nibbles := block[2:]
		dst := out[b*QuantBlock : (b+1)*QuantBlock]
		for j := 0; j < QuantBlock/2; j++ {
			byteValue := nibbles[j]
			dst[j] = float32(int32(byteValue&0x0F)-8) * scale
			dst[j+16] = float32(int32(byteValue>>4)-8) * scale
		}
	}
}

// matVecF32Rows computes rows [start, end) with W stored row-major in
// little-endian float32.
func matVecF32Rows(w []byte, x []float32, inputs int, y []float32, start, end int) {
	weights := unsafe.Slice((*float32)(unsafe.Pointer(&w[0])), len(w)/4)
	for o := start; o < end; o++ {
		y[o] = DotF32(weights[o*inputs:(o+1)*inputs], x)
	}
}
