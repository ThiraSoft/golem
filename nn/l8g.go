package nn

// L8G: the same container as D4G with a different codebook. 64 weights in 26
// bytes, 3.25 bits each.
//
// Where D4G spends twelve bits on a point of a four-dimensional lattice, this
// spends three bits a weight on one of eight numbers — the Lloyd-Max levels of
// a unit Gaussian, which is what the weights are once the rotation has finished
// with them. Sixteen twelve-bit codes and sixty-four three-bit ones are the
// same twenty-four bytes, so the two formats have the same row, the same two
// step codes a block, and the same size to the byte.
//
// What changes is what the kernel reads. D4G expands a code by looking it up in
// a table of 4096 points, sixteen kibibytes that cost eleven to seventeen
// percent of a dispatch. This expands one by indexing eight constants, which a
// shader keeps in registers and reads nothing for.
//
// What that costs in weight error is 0.29 dB — measured on the real model, and
// close to the textbook space-filling gain of D4 over the scalar line, which is
// 0.37. What it costs the model is four times that: on Qwen3-0.6B the lattice
// reads 39.80 and this reads 41.30, where 0.29 dB should have been worth half a
// point. A uniform codebook of the same eight levels reads 43.22 at 15.34 dB,
// so the three sit in the order their weight error puts them in — perplexity
// simply grows faster than that error does, about as its 1.3rd power over this
// stretch.
//
// Three explanations for the gap were tried and each was wrong. It is not the
// tail being clipped: the lattice fails to reach more weights than this does,
// 23.9% against 13.2%, and shrinks the row by the same 2.4% of its energy. It
// is not the errors of a subvector leaning against each other to cancel in a
// dot product: the cross terms are +0.7% for the lattice and -0.3% here, which
// is nothing and is the wrong way round. And it is not that the model prefers a
// uniform step to a companded one: uniform is worse at both.
//
// So this format loses, and the reason is not understood. It is kept because
// the question comes back — the table is the only memory the kernel touches
// that is not weights — and because it answers in one flag instead of a week.
// It also encodes nine times faster, which is not nothing for a checkpoint that
// takes an hour.
//
// Everything else about the scheme is unchanged and is where its value is: the
// per-column vector, the Hadamard rotation, the step per thirty-two weights and
// the search that picks it. See nn/d4g_prepare.go.

// l8Levels are Lloyd's eight levels for a unit Gaussian, found by alternating
// centroids and boundaries over four million samples and symmetrised. They
// agree with Max's published values to four decimals.
var l8Levels = [8]float32{
	-2.152793, -1.344896, -0.756392, -0.245248,
	0.245248, 0.756392, 1.344896, 2.152793,
}

// l8Bounds are the midpoints between consecutive levels: the seven comparisons
// that name a code without a search.
var l8Bounds [7]float32

func init() {
	for i := 0; i < 7; i++ {
		l8Bounds[i] = (l8Levels[i] + l8Levels[i+1]) / 2
	}
}

const (
	// L8Bits is what one weight costs in codes.
	L8Bits = 3
	// l8BlockBytes is two step codes then sixty-four three-bit codes, which is
	// D4G's twenty-six to the byte.
	l8BlockBytes = 26
)

// L8Levels is the codebook, for a kernel that wants to hold it in registers.
func L8Levels() []float32 { return l8Levels[:] }

// L8Code names the level nearest a value, in units of the block's step.
func L8Code(v float32) byte {
	// A linear walk over seven bounds rather than a binary search: it is seven
	// compares either way on a modern core, and it is the same order the
	// levels are written in.
	c := byte(0)
	for i := 0; i < 7; i++ {
		if v > l8Bounds[i] {
			c = byte(i + 1)
		}
	}
	return c
}

// L8Level expands a code.
func L8Level(c byte) float32 { return l8Levels[c&7] }

// PutL8Codes packs sixty-four three-bit codes into twenty-four bytes. A code
// straddles two bytes whenever its bit offset passes five, which is five of
// every eight, so the write is two.
func PutL8Codes(dst []byte, codes []byte) {
	for i := range dst[:L8Block/8*3] {
		dst[i] = 0
	}
	for i, c := range codes {
		bit := i * L8Bits
		at, off := bit>>3, uint(bit&7)
		dst[at] |= c << off
		if off > 5 {
			dst[at+1] |= c >> (8 - off)
		}
	}
}

// L8Block is how many weights share one run of codes, the same sixty-four D4G
// uses.
const L8Block = 64

// l8CodeAt reads one of the sixty-four codes of a packed run.
func l8CodeAt(src []byte, i int) byte {
	bit := i * L8Bits
	at, off := bit>>3, uint(bit&7)
	v := src[at] >> off
	if off > 5 {
		v |= src[at+1] << (8 - off)
	}
	return v & 7
}

// L8Planes splits a row into its steps and its codes, the same split D4G takes
// and for the same reason: a block of twenty-six bytes would start on a
// two-byte boundary and a shader reads words.
func L8Planes(row []byte, n int) (steps, codes []byte) {
	nb := n / L8Block
	return row[:nb*2], row[nb*2 : nb*l8BlockBytes]
}

// DequantizeL8G expands one row of n weights. out must hold n floats.
func DequantizeL8G(w []byte, n int, out []float32) {
	if n%L8Block != 0 {
		panic("nn: L8G rows must be a multiple of 64")
	}
	steps, allCodes := L8Planes(w, n)
	for b := 0; b*L8Block < n; b++ {
		lo, hi := d4Steps[steps[b*2]], d4Steps[steps[b*2+1]]
		codes := allCodes[b*24 : (b+1)*24]
		dst := out[b*L8Block : (b+1)*L8Block]
		for i := 0; i < L8Block; i++ {
			d := lo
			if i >= D4SubBlock {
				d = hi
			}
			dst[i] = l8Levels[l8CodeAt(codes, i)] * d
		}
	}
}

// matVecL8GRows computes y = W x for L8G weights against float32 activations
// that Prepare has already scaled and rotated.
func matVecL8GRows(w []byte, b *Batch, cols int, ys [][]float32, start, end int) {
	stride := cols / L8Block * l8BlockBytes
	row := make([]float32, cols)
	for r := start; r < end; r++ {
		DequantizeL8G(w[r*stride:(r+1)*stride], cols, row)
		for c := 0; c < b.Size; c++ {
			ys[c][r] = DotF32(row, b.F[c])
		}
	}
}
