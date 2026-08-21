package nn

// A batch of activations, quantized together.
//
// Several columns meeting the same matrix want to meet it one row at a time:
// the row is read from memory once and multiplies every column while it is
// still in the first-level cache. For that the kernel has to reach the same
// block of every column without walking away from the cache line it is on, so
// the quantized form is interleaved — block by block, and within a block,
// column by column.
//
// A batch of one is laid out exactly as a lone vector is, which is why
// generation loses nothing by going through the same path.

import "math"

type Batch struct {
	Size, Width int
	// Stride is how many columns the interleaving was laid out for. It is the
	// distance from one block of a column to the next, and it is what a kernel
	// is told so that it can walk one column of the interleaving.
	Stride int
	F      [][]float32 // the activations, one row per column

	Q      []int8    // [block][column][QuantBlock]
	Scales []float32 // [block][column]
	Corr   []float32 // [block][column], the recentring dot_q4_0 no longer does

	// The Q8_K form, built only for the batches a K-quantized matrix consumes.
	QK      []int8    // [column][width]
	KScales []float32 // [column][superblock]
	BSums   []int16   // [column][group of sixteen]

	// BF16 says the floats are to be rounded to bfloat16 as they are
	// quantized. ggml converts an activation to whatever the weight's dot
	// wants: Q8_0 for a Q4_0 weight, which is what Q and Scales above hold,
	// and bfloat16 for a bfloat16 one. Keeping float32 there would be more
	// precision than the reference has, which is a divergence and not an
	// improvement — it moves the queries of a Qwen block by a part in a
	// thousand, four thousand times the gap that remains once it is done.
	//
	// It is set on the batches an engine feeds to bfloat16 weights. Rounding
	// is idempotent, so a caller that has already rounded loses nothing.
	BF16 bool
}

// NewBatch allocates for `size` activations of `width` elements each.
func NewBatch(width, size int) *Batch {
	if width%QuantBlock != 0 {
		panic("nn: activation length must be a multiple of the block size")
	}
	blocks := width / QuantBlock
	b := &Batch{
		Size:   size,
		Stride: size,
		Width:  width,
		F:      make([][]float32, size),
		Q:      make([]int8, blocks*size*QuantBlock),
		Scales: make([]float32, blocks*size),
		Corr:   make([]float32, blocks*size),
	}
	for i := range b.F {
		b.F[i] = make([]float32, width)
	}
	return b
}

// Quantize refreshes the quantized form of every column from F.
func (b *Batch) Quantize() { b.QuantizeRange(0, b.Width) }

// QuantizeRange refreshes one stretch of every column, which must begin and end
// on a block. It is what lets a core quantize the part of a feed forward it
// just computed, inside the section that computed it.
func (b *Batch) QuantizeRange(first, last int) {
	if first%QuantBlock != 0 || last%QuantBlock != 0 {
		panic("nn: a quantized range must begin and end on a block")
	}
	for block := first / QuantBlock; block < last/QuantBlock; block++ {
		for c := 0; c < b.Size; c++ {
			b.quantizeBlock(block, c)
		}
	}
}

// QuantizeColumnRange is the same for one column alone, for the sections that
// are split by column rather than by row.
//
// When BF16 is set it also rounds the floats of that range, because a bfloat16
// weight is multiplied against a bfloat16 activation and not a float32 one.
// See the field's comment.
func (b *Batch) QuantizeColumnRange(column, first, last int) {
	if b.BF16 {
		f := b.F[column]
		for i := first; i < last && i < len(f); i++ {
			f[i] = RoundBF16(f[i])
		}
	}
	for block := first / QuantBlock; block < last/QuantBlock; block++ {
		b.quantizeBlock(block, column)
	}
}

// quantizeBlock is ggml's quantizer, block by block: the scale divides 127 by
// the peak rather than inverting itself, halves round to even, and the scale
// the product uses is the fp16 the reference would have stored. Each of those
// is one bit away from the obvious thing to write, and each decides which side
// of an integer an activation lands on often enough to be seen thirty blocks
// later.
func (b *Batch) quantizeBlock(block, column int) {
	values := b.F[column][block*QuantBlock : (block+1)*QuantBlock]
	index := block*b.Stride + column
	q := b.Q[index*QuantBlock : (index+1)*QuantBlock]

	var peak float32
	for _, f := range values {
		if a := float32(math.Abs(float64(f))); a > peak {
			peak = a
		}
	}
	if peak == 0 {
		b.Scales[index], b.Corr[index] = 0, 0
		clear(q)
		return
	}
	scale := peak / 127
	inverse := 127 / peak
	b.Scales[index] = halfToFloat(floatToHalf(scale))

	var sum int32
	for i, f := range values {
		n := nearestInt(f * inverse)
		if n > 127 {
			n = 127
		} else if n < -128 {
			n = -128
		}
		q[i] = int8(n)
		sum += n
	}
	// A Q4_0 weight is stored shifted by eight, and 8 x scale x sum(q) is the
	// same for every row of every matrix this column meets.
	b.Corr[index] = 8 * b.Scales[index] * float32(sum)
}

// The Q8_K form, which is what a Q6_K matrix wants against it.
//
// Q8_0 carries a scale for every 32 values; the K-quants work in superblocks of
// 256, and their product needs the activation cut the same way — one scale per
// superblock, plus the sum of each group of sixteen. Those sums are what pays
// for the recentring of the weights: a Q6_K weight is stored unsigned and
// shifted by 32, and subtracting 32 x sum(q) once per group is cheaper than
// subtracting 32 from every weight.
//
// It is not interleaved: the only K-quantized product in either engine is the
// logit head, which meets one column at a time.

// QuantizeK refreshes the Q8_K form of every column, allocating it on first
// use. The width must divide into whole superblocks.
func (b *Batch) QuantizeK() {
	if b.Width%SuperBlock != 0 {
		panic("nn: the Q8_K form needs a multiple of the superblock size")
	}
	if b.QK == nil {
		b.QK = make([]int8, b.Stride*b.Width)
		b.KScales = make([]float32, b.Stride*b.Width/SuperBlock)
		b.BSums = make([]int16, b.Stride*b.Width/16)
	}
	for c := 0; c < b.Size; c++ {
		q := b.QK[c*b.Width : (c+1)*b.Width]
		scales := b.KScales[c*b.Width/SuperBlock : (c+1)*b.Width/SuperBlock]
		sums := b.BSums[c*b.Width/16 : (c+1)*b.Width/16]

		for block := range scales {
			values := b.F[c][block*SuperBlock : (block+1)*SuperBlock]
			into := q[block*SuperBlock : (block+1)*SuperBlock]

			// ggml takes its scale from the signed extremum rather than the
			// absolute one, so the extreme value lands on -128 instead of 127.
			var peak, extreme float32
			for _, f := range values {
				a := f
				if a < 0 {
					a = -a
				}
				if a > peak {
					peak, extreme = a, f
				}
			}
			if peak == 0 {
				scales[block] = 0
				clear(into)
				clear(sums[block*(SuperBlock/16) : (block+1)*(SuperBlock/16)])
				continue
			}
			inverse := -128 / extreme
			for i, f := range values {
				n := nearestInt(f * inverse)
				if n > 127 {
					n = 127
				}
				into[i] = int8(n)
			}
			scales[block] = 1 / inverse
			for g := 0; g < SuperBlock/16; g++ {
				var sum int32
				for _, n := range into[g*16 : (g+1)*16] {
					sum += int32(n)
				}
				sums[block*(SuperBlock/16)+g] = int16(sum)
			}
		}
	}
}

// nearestInt rounds halves to even, the way ggml's quantizers do: adding
// 1.5 x 2^23 pushes the fractional bits off the end of a float32 mantissa, and
// the rounding the hardware applies to that addition is the rounding wanted.
// It is worth the trick — the quantizer runs over every activation of every
// block, and a call to math.RoundToEven there was showing up in the profile.
// The argument must stay well inside 2^22, which a value bound by 128 does.
func nearestInt(f float32) int32 {
	const magic = 12582912.0 // 1.5 * 2^23
	return int32(math.Float32bits(f+magic)&0x007fffff) - 0x00400000
}

// Set copies x into one column and quantizes it.
func (b *Batch) Set(column int, x []float32) {
	if len(x) != b.Width {
		panic("nn: activation length mismatch")
	}
	copy(b.F[column], x)
	b.QuantizeColumnRange(column, 0, b.Width)
}
