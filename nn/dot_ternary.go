package nn

// The ternary products, in integers.
//
// Both formats were multiplied by expanding each block of 128 into floats and
// taking a float dot against the float activation: a table lookup and a
// conversion a weight, on one core's scalar unit. On the processor that is
// what made Bonsai 2 run at a fifth of a token a second where Prism's own fork
// of llama.cpp runs at eight, on a machine with no card to hand it to.
//
// The fork multiplies PQ2_0 against a Q8_K activation, one scale for every 256
// inputs, and this does the same: a block of 128 is half a superblock, so its
// whole integer dot is scaled once. PTQ1_0 reads the same Q8_K form; the fork
// takes Q8_0 there, through a scalar loop, which is the one place this departs
// from it, and by less than the float activation it replaces did.
//
// A code is the weight plus one, so the product is sum(c*a) - sum(a), and the
// second term is the activation's alone: it is computed once a column, not
// once a row.
//
// PQ2_0 packs four weights into a byte, weight 4b+j at bit 2j. Shifting the
// whole block right by 2j and masking gives one plane of thirty-two codes, and
// those are weights j, j+4, j+8 and so on — so the activation is put in the
// same order once, when it is quantized, and the kernel never shuffles a byte.

import "encoding/binary"

// ternaryColumn is one column of activation in the form a ternary product
// reads: the Q8_K values, the sum of each block of 128, and the superblock
// scales.
type ternaryColumn struct {
	q      []int8
	sums   []int32
	scales []float32
}

// ternaryActivation builds that form from the batch's Q8_K one. planes puts
// each block of 128 in PQ2_0's plane order; PTQ1_0 wants it as it is.
func ternaryActivation(b *Batch, planes bool) []ternaryColumn {
	cols := newTernaryColumns(b, planes)
	for c := range cols {
		fillTernary(&cols[c], b.QK[c*b.Width:(c+1)*b.Width], planes, 0, b.Width/TernaryBlock)
	}
	return cols
}

// newTernaryColumns allocates the form over a batch whose Q8_K form is
// allocated; fillTernary writes it.
func newTernaryColumns(b *Batch, planes bool) []ternaryColumn {
	cols := make([]ternaryColumn, b.Size)
	for c := range cols {
		cols[c] = ternaryColumn{
			q:      b.QK[c*b.Width : (c+1)*b.Width],
			sums:   make([]int32, b.Width/TernaryBlock),
			scales: b.KScales[c*b.Width/SuperBlock : (c+1)*b.Width/SuperBlock],
		}
		if planes {
			cols[c].q = make([]int8, b.Width)
		}
	}
	return cols
}

// fillTernary writes blocks [first, last) of one column from its Q8_K values.
func fillTernary(col *ternaryColumn, src []int8, planes bool, first, last int) {
	for blk := first; blk < last; blk++ {
		from := src[blk*TernaryBlock : (blk+1)*TernaryBlock]
		if planes {
			to := col.q[blk*TernaryBlock:]
			for j := 0; j < 4; j++ {
				for p := 0; p < 32; p++ {
					k, byteAt := p/8, p%8
					to[32*j+p] = from[32*k+4*byteAt+j]
				}
			}
		}
		var s int32
		for _, v := range from {
			s += int32(v)
		}
		col.sums[blk] = s
	}
}

// ternaryInput is the activation a ternary matrix reads, made from the
// caller's batch: rotated if the matrix is, quantized, and in plane order for
// PQ2_0. Each of those works within a Hadamard group or a superblock, so the
// groups go to the cores: every product does this, and done on one core it
// is time the memory spends idle.
func (m Matrix) ternaryInput(b *Batch) []ternaryColumn {
	planes := m.Quant == PQ2_0
	width := b.Width
	f := &Batch{Size: b.Size, Stride: b.Size, Width: width, F: make([][]float32, b.Size)}
	for c := range f.F {
		f.F[c] = make([]float32, width)
	}
	f.allocK()
	cols := newTernaryColumns(f, planes)

	span := SuperBlock
	if m.Pre != nil && m.HadGroup > span {
		span = m.HadGroup
	}
	if width%span != 0 {
		span = width
	}
	// The work is small, but it is on the path of every product, so it is
	// spread whatever its size.
	InParallel(width/span, parallelThreshold, func(first, last int) {
		lo, hi := first*span, last*span
		for c := range cols {
			x := f.F[c][lo:hi]
			copy(x, b.F[c][lo:hi])
			if m.Pre != nil {
				PrepareGolem(x, m.Pre[lo:hi], m.HadGroup)
			}
			for blk := lo / SuperBlock; blk < hi/SuperBlock; blk++ {
				f.quantizeKBlock(c, blk)
			}
			fillTernary(&cols[c], f.QK[c*width:(c+1)*width], planes, lo/TernaryBlock, hi/TernaryBlock)
		}
	})
	return cols
}

// dotPQ2_0Go is the portable form: one row of n weights against a column in
// plane order.
func dotPQ2_0Go(w []byte, a *ternaryColumn, n int) float32 {
	var total float32
	for blk := 0; blk < n/TernaryBlock; blk++ {
		block := w[blk*pq2_0BlockBytes : (blk+1)*pq2_0BlockBytes]
		d := halfToFloat(binary.LittleEndian.Uint16(block))
		qs := block[2:34]
		act := a.q[blk*TernaryBlock:]
		var dot int32
		for j := 0; j < 4; j++ {
			for p := 0; p < 32; p++ {
				dot += int32(qs[p]>>(2*j)&3) * int32(act[32*j+p])
			}
		}
		total += d * a.scales[blk/2] * float32(dot-a.sums[blk])
	}
	return total
}

// dotPTQ1_0Go is the portable form: one row of n weights against a column in
// natural order. The trits come out in the order DequantizePTQ1_0 writes them.
func dotPTQ1_0Go(w []byte, a *ternaryColumn, n int) float32 {
	var total float32
	var codes [TernaryBlock]uint8
	for blk := 0; blk < n/TernaryBlock; blk++ {
		block := w[blk*ptq1_0BlockBytes : (blk+1)*ptq1_0BlockBytes]
		d := halfToFloat(binary.LittleEndian.Uint16(block[26:28]))
		at := 0
		for _, stage := range [2][2]int{{0, 16}, {16, 8}} {
			for t := 0; t < 5; t++ {
				for m := 0; m < stage[1]; m++ {
					codes[at] = trit(block[stage[0]+m], t)
					at++
				}
			}
		}
		for t := 0; t < 4; t++ {
			for h := 0; h < 2; h++ {
				codes[at] = trit(block[24+h], t)
				at++
			}
		}
		act := a.q[blk*TernaryBlock:]
		var dot int32
		for i, c := range codes {
			dot += int32(c) * int32(act[i])
		}
		total += d * a.scales[blk/2] * float32(dot-a.sums[blk])
	}
	return total
}

// trit is the t-th base-3 digit of a PTQ1_0 byte, counting from the top: the
// byte holds the digits as a fraction of 243, so multiplying by 3^t brings the
// one wanted to the front and the top byte of three times that is the digit.
func trit(q byte, t int) uint8 {
	return uint8((uint16(q*pow3[t]) * 3) >> 8)
}

// ternaryRows computes rows [start, end) against every column.
func ternaryRows(quant Quant, w []byte, cols []ternaryColumn, n int, ys [][]float32, start, end int) {
	stride := n / TernaryBlock * pq2_0BlockBytes
	dot := dotPQ2_0
	if quant == PTQ1_0 {
		stride = n / TernaryBlock * ptq1_0BlockBytes
		dot = dotPTQ1_0
	}
	for r := start; r < end; r++ {
		row := w[r*stride : (r+1)*stride]
		for c := range cols {
			ys[c][r] = dot(row, &cols[c], n)
		}
	}
}
