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

	// Pre is the per-column vector this matrix's activation has to go through
	// before the product, and HadGroup the width of the rotation that follows
	// it. Both belong to a D4G matrix and are nil and zero for every other
	// format — see nn/d4g_prepare.go for what they are and why the weights do
	// not carry them.
	//
	// A product finds them here and does the transform itself, on a copy. A
	// caller reading one activation with several matrices of the same site can
	// leave them unset and do the transform once, which is what qwen does.
	Pre      []float32
	HadGroup int
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
	case BF16, F16:
		return m.Cols * 2
	case Q4_0:
		return m.Cols / QuantBlock * q4_0BlockBytes
	case Q4_1:
		return m.Cols / QuantBlock * q4_1BlockBytes
	case Q4_K:
		return m.Cols / SuperBlock * q4_kBlockBytes
	case Q5_K:
		return m.Cols / SuperBlock * q5_kBlockBytes
	case Q6_K:
		return m.Cols / SuperBlock * q6_kBlockBytes
	case Q8_0:
		return m.Cols / QuantBlock * q8_0BlockBytes
	case L8G:
		return m.Cols / L8Block * l8BlockBytes
	case D4G, D4G16:
		return m.Cols / D4Block * D4BlockBytes(m.Quant.D4Width())
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
	if m.WantsQ8K() {
		b.QuantizeK()
	}
	InParallel(m.Rows, m.Rows*m.Cols*b.Size, func(start, end int) {
		m.rows(b, ys, start, end)
	})
}

// WantsQ8K says the product reads the activation in its Q8_K form rather than
// its Q8_0 one. Only Q6_K does, and building that form is the caller's job when
// it calls MatVecRows: QuantizeK writes three slices into the batch, so a
// worker that started it while another worker read it would hand out one that
// is allocated and two that are not.
func (m Matrix) WantsQ8K() bool { return m.Quant == Q6_K }

// MatVecRows computes rows [start, end) of the product on the caller's thread,
// for a caller that is already inside a parallel section and wants to finish
// what it produced before the section ends. A Q6_K matrix wants QuantizeK on
// the batch first, outside the section — see WantsQ8K.
func (m Matrix) MatVecRows(b *Batch, ys [][]float32, start, end int) {
	m.rows(b, ys, start, end)
}

// rows computes rows [start, end) of the product for every activation in xs, on
// the caller's thread. The format is decided once for the whole call, and the
// batch is the innermost loop so that a row is read from memory once.
func (m Matrix) rows(b *Batch, ys [][]float32, start, end int) {
	if m.Pre != nil {
		b = m.prepare(b)
	}
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
	case Q4_1:
		matVecQ4_1Rows(m.Data, b, m.Cols, ys, start, end)
	case BF16:
		weights := unsafe.Slice((*uint16)(unsafe.Pointer(&m.Data[0])), len(m.Data)/2)
		for r := start; r < end; r++ {
			row := weights[r*m.Cols : (r+1)*m.Cols]
			for c := 0; c < b.Size; c++ {
				ys[c][r] = dotBF16(row, b.F[c])
			}
		}
	case F16:
		// IEEE binary16, which is what a clip projector stores its matrices
		// as. DotF32Half is the kernel nn already had for the key-value cache;
		// this is the first thing to multiply by a whole matrix of them.
		weights := unsafe.Slice((*uint16)(unsafe.Pointer(&m.Data[0])), len(m.Data)/2)
		for r := start; r < end; r++ {
			row := weights[r*m.Cols : (r+1)*m.Cols]
			for c := 0; c < b.Size; c++ {
				ys[c][r] = DotF32Half(b.F[c], row)
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
	case Q4_K:
		matVecQ4_KRows(m.Data, b, m.Cols, ys, start, end)
	case Q5_K:
		matVecQ5_KRows(m.Data, b, m.Cols, ys, start, end)
	case Q6_K:
		if len(b.QK) == 0 {
			panic("nn: a Q6_K product wants QuantizeK on the batch before the section")
		}
		matVecQ6_KRows(m.Data, b, m.Cols, ys, start, end)
	case Q8_0:
		matVecQ8_0Rows(m.Data, b, m.Cols, ys, start, end)
	case L8G:
		matVecL8GRows(m.Data, b, m.Cols, ys, start, end)
	case D4G, D4G16:
		matVecD4GRows(m.Data, b, m.Cols, m.Quant.D4Width(), ys, start, end)
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
// prepare is the activation this matrix's kernel should read: the caller's
// batch put through the site's vector and rotation, in a copy of its own.
//
// A copy because the batch is one and the matrices reading it are several —
// the three attention projections read the stream, and undoing the scheme in
// place for the first would hand the second something already transformed. It
// costs a pass over the activation where the product costs a pass over the
// matrix, which is a thousandth of it and buys a caller that needs to know
// nothing about the format.
//
// A caller that reads one activation with several matrices of the same site
// can do better by transforming it once itself, which is what qwen/block.go
// does. This is for the callers that would rather not.
func (m Matrix) prepare(b *Batch) *Batch {
	out := &Batch{Size: b.Size, Width: b.Width, F: make([][]float32, b.Size)}
	for c := 0; c < b.Size; c++ {
		out.F[c] = make([]float32, b.Width)
		copy(out.F[c], b.F[c])
		PrepareD4G(out.F[c], m.Pre, m.HadGroup)
	}
	return out
}

// unrotate recovers a weight row from the form the file holds it in.
//
// A D4G matrix is stored as A·(q ⊙ W), and a product never undoes that: it
// puts the activation through the reciprocal instead, which is the whole
// bargain of the scheme. But a row read on its own is not a product. The
// embedding table is the case that matters — every engine here reads a token's
// row out of it — and a row handed back rotated is a token entering the model
// as somebody else's vector.
//
// It is done here rather than at each of the eleven call sites because a
// matrix that carries its own transform should carry it: qwen/model.go undoes
// it by hand, qwen35 did not, and the model answered fluently and wrongly for
// a day. Nothing in this repository reads a row wanting the stored form; the
// products read Data.
func (m Matrix) unrotate(out []float32) {
	if m.Pre != nil && m.HadGroup > 0 {
		UnprepareD4G(out, m.Pre, m.HadGroup)
	}
}

func (m Matrix) Row(index int, out []float32) {
	if index < 0 || index >= m.Rows {
		panic(fmt.Sprintf("nn: row %d out of %d", index, m.Rows))
	}
	stride := m.RowBytes()
	row := m.Data[index*stride : (index+1)*stride]

	switch m.Quant {
	case Q6_K:
		DequantizeQ6_K(row, m.Cols, out)
	case Q5_K:
		DequantizeQ5_K(row, m.Cols, out)
	case Q4_K:
		DequantizeQ4_K(row, m.Cols, out)
	case L8G:
		DequantizeL8G(row, m.Cols, out)
		m.unrotate(out)
	case D4G, D4G16:
		DequantizeD4GN(row, m.Cols, m.Quant.D4Width(), out)
		m.unrotate(out)
	case F32:
		for i := 0; i < m.Cols; i++ {
			out[i] = float32FromBytes(row[i*4:])
		}
	case BF16:
		for i := 0; i < m.Cols; i++ {
			out[i] = bf16ToFloat(row[i*2:])
		}
	case F16:
		for i := 0; i < m.Cols; i++ {
			out[i] = halfToFloat(uint16(row[i*2]) | uint16(row[i*2+1])<<8)
		}
	case Q4_0:
		dequantizeQ4_0Row(row, m.Cols, out)
	case Q4_1:
		dequantizeQ4_1Row(row, m.Cols, out)
	case Q8_0:
		dequantizeQ8_0Row(row, m.Cols, out)
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
