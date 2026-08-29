package nn

// D4G: golem's own weight format. 64 weights in 26 bytes, 3.25 bits each.
//
// A row is two planes: every block's two step codes, then every block's sixteen
// twelve-bit codes. Interleaving them would put a block at byte 26b, which is
// two-byte aligned, and a shader reads words — so the planes are split for the
// same reason vk/mixture.go splits Q5_K, and the row costs exactly what the
// interleaved form would.
//
// The two bytes a block spends on steps used to be one fp16 for all sixty-four
// weights. They are now two eight-bit codes, one per thirty-two, naming powers
// of two a sixteenth apart. The block pays the same and the step follows the
// weights twice as closely, which is worth half a decibel — measured, not
// assumed, by encoding the same model at both granularities.
//
// What the eight bits give up is range and precision, and the balance between
// them is not free to choose badly. A grid an eighth apart — nine percent, and
// a range of 2^32 nobody needs — costs a whole point of perplexity against an
// fp16 step at the same granularity, which is more than the finer granularity
// wins back. The step sits at the bottom of a curve, and nine percent is far
// enough up its sides to matter. A sixteenth apart is four percent over a range
// of 2^16, from 7.6e-6 to 0.48, which is wider than any weight of any model has
// asked for and near enough the bottom to cost a tenth of what the coarse grid
// did.
//
// A code names a point of the D4 lattice — the four-dimensional checkerboard,
// integer vectors of even coordinate sum — and there are exactly as many codes
// as twelve bits can name. The points are taken in canonical order, by squared
// norm and then lexicographically: every point of norm 40 or less, which is
// 3961 of them, and then the first 135 of norm 42. Stopping at 40 would leave
// 135 codes unspent, and a code that names nothing is a code the weights paid
// for.
//
// Why a lattice and not a trained codebook: the nearest point of D4 is found by
// rounding, in time that does not depend on how many points there are, which is
// what makes quantizing four billion weights an afternoon rather than a week.
// Why D4 and not E8, which packs better: the decoder is a table lookup, so the
// table is the codebook, and E8 at this rate would need thirteen million points
// — two hundred megabytes. D4 needs 3961, thirty-one kibibytes, which fits in a
// workgroup's shared memory. And since the table is the lattice itself, one
// table serves every tensor of every model, where a trained codebook would need
// one per tensor.
//
// What the block does not carry is the rest of the scheme: the activations meet
// a per-column vector and a Hadamard rotation before they reach these weights,
// and both belong to the matrix rather than to the block. See Prepare.

import (
	"math"
	"sort"
)

const (
	// D4Block is how many weights share one scale and one run of codes.
	D4Block = 64
	// d4BlockBytes is two step codes then sixteen twelve-bit codes.
	d4BlockBytes = 26
	// D4SubBlock is how many weights share one step.
	D4SubBlock = 32
	// D4Radius is the largest squared norm every point of which is addressable
	// at twelve bits.
	D4Radius = 40
	// d4EdgeNorm is the next norm up, the one the twelve-bit shell is cut in
	// the middle of so that the codes come out exactly 4096.
	d4EdgeNorm = 42
	// d4MaxNorm is where the enumeration stops: far enough for the sixteen-bit
	// shell, which needs 65536 points and reaches into squared norm 162.
	d4MaxNorm = 162
	// D4Bits is the width of one code in the ordinary tier, and 1<<D4Bits is
	// how many points it may name: the shell is cut to fit the code, not the
	// other way round.
	D4Bits = 12
	// D4Bits16 is the wide tier, four bits a code more and a whole bit a
	// weight. Its table is 65536 points rather than 4096 — a quarter of a
	// mebibyte rather than sixteen kibibytes — which no longer fits a
	// workgroup's shared memory and has to live in a buffer.
	D4Bits16 = 16
)

// d4Points is the shell in canonical order — by squared norm, then
// lexicographically — so that an index means the same thing to every encoder
// and every decoder. Four int8 a point.
var d4Points [][4]int8

// d4Index maps a point back to its place in that order.
var d4Index map[[4]int8]uint16

// D4Table is the shell as a flat float32 array, four to a point, in the order
// the codes index. This is what a kernel uploads.
func D4Table() []float32 { return D4TableN(D4Bits) }

// D4TableN is the table a tier of the given width uploads.
func D4TableN(bits int) []float32 {
	pts := d4Points[:1<<bits]
	out := make([]float32, len(pts)*4)
	for i, p := range pts {
		for j := 0; j < 4; j++ {
			out[i*4+j] = float32(p[j])
		}
	}
	return out
}

// D4Points is how many points the ordinary tier's shell holds.
func D4Points() int { return 1 << D4Bits }

// d4Steps is what the eight bits of a step code name.
var d4Steps [256]float32

// D4Step expands a step code.
func D4Step(code byte) float32 { return d4Steps[code] }

// D4StepCode is the code nearest a step, in the ratio the codes are spaced by.
// A step below the smallest the code can name is that smallest one: it belongs
// to a block whose weights are all but zero, and the lattice will round them
// there whatever the step says.
func D4StepCode(v float32) byte {
	if !(v > 0) {
		return 0
	}
	c := math.Round(math.Log2(float64(v))*16 + 272)
	if c < 0 {
		c = 0
	}
	if c > 255 {
		c = 255
	}
	return byte(c)
}

func init() {
	for c := 0; c < 256; c++ {
		d4Steps[c] = float32(math.Exp2((float64(c) - 272) / 16))
	}
	lim := int(math.Sqrt(float64(d4MaxNorm)))
	for a := -lim; a <= lim; a++ {
		for b := -lim; b <= lim; b++ {
			for c := -lim; c <= lim; c++ {
				for d := -lim; d <= lim; d++ {
					if (a+b+c+d)&1 != 0 {
						continue
					}
					if a*a+b*b+c*c+d*d > d4MaxNorm {
						continue
					}
					d4Points = append(d4Points, [4]int8{int8(a), int8(b), int8(c), int8(d)})
				}
			}
		}
	}
	sort.Slice(d4Points, func(i, j int) bool {
		pi, pj := d4Points[i], d4Points[j]
		ni := int(pi[0])*int(pi[0]) + int(pi[1])*int(pi[1]) + int(pi[2])*int(pi[2]) + int(pi[3])*int(pi[3])
		nj := int(pj[0])*int(pj[0]) + int(pj[1])*int(pj[1]) + int(pj[2])*int(pj[2]) + int(pj[3])*int(pj[3])
		if ni != nj {
			return ni < nj
		}
		for k := 0; k < 4; k++ {
			if pi[k] != pj[k] {
				return pi[k] < pj[k]
			}
		}
		return false
	})
	d4Points = d4Points[:1<<D4Bits16]
	d4Index = make(map[[4]int8]uint16, len(d4Points))
	for i, p := range d4Points {
		d4Index[p] = uint16(i)
	}
}

// D4Code returns the code naming a lattice point in the twelve-bit tier.
func D4Code(p [4]int8) (uint16, bool) { return D4CodeN(p, D4Bits) }

// D4CodeN is the same for a tier of the given width: the canonical rank of the
// point, and whether that rank is one the width can name. A point past the tier
// has no code, and the quantizer is responsible for not producing one.
func D4CodeN(p [4]int8, bits int) (uint16, bool) {
	i, ok := d4Index[p]
	return i, ok && int(i) < 1<<bits
}

// D4Has is D4CodeN without the code, and answers without a map wherever it can:
// below the tier's full norm every point is in, above its edge none is. The
// quantizer asks this of every candidate point of every subvector of every
// weight, so the map is worth avoiding.
func D4Has(p [4]int8, bits int) bool {
	n := int(p[0])*int(p[0]) + int(p[1])*int(p[1]) + int(p[2])*int(p[2]) + int(p[3])*int(p[3])
	full, edge := D4TierNorms(bits)
	if n <= full {
		return true
	}
	if n > edge {
		return false
	}
	_, ok := D4CodeN(p, bits)
	return ok
}

// D4TierNorms is the largest norm a tier holds entirely, and the norm it is cut
// in the middle of.
func D4TierNorms(bits int) (full, edge int) {
	if bits >= D4Bits16 {
		return 160, 162
	}
	return D4Radius, d4EdgeNorm
}

// D4BlockBytes is what one block of sixty-four weights costs at a code width:
// two step codes and sixteen codes.
func D4BlockBytes(bits int) int { return 2 + D4Block/4*bits/8 }

// D4Point expands a code.
func D4Point(code uint16) [4]int8 { return d4Points[code] }

// PutD4Codes packs sixteen twelve-bit codes into twenty-four bytes, two codes
// to three.
func PutD4Codes(dst []byte, codes []uint16) { PutD4CodesN(dst, codes, D4Bits) }

// PutD4CodesN packs a block's codes at the given width. Sixteen bits are two
// plain little-endian bytes; twelve are two codes to three.
func PutD4CodesN(dst []byte, codes []uint16, bits int) {
	if bits == D4Bits16 {
		for i, c := range codes {
			dst[i*2] = byte(c)
			dst[i*2+1] = byte(c >> 8)
		}
		return
	}
	for i := 0; i+1 < len(codes); i += 2 {
		a, b := codes[i], codes[i+1]
		o := i / 2 * 3
		dst[o] = byte(a)
		dst[o+1] = byte(a>>8&0x0F) | byte(b&0x0F)<<4
		dst[o+2] = byte(b >> 4)
	}
}

// d4CodeAt reads one of the sixteen codes of a packed run.
func d4CodeAt(src []byte, i, bits int) uint16 {
	if bits == D4Bits16 {
		return uint16(src[i*2]) | uint16(src[i*2+1])<<8
	}
	o := i / 2 * 3
	if i&1 == 0 {
		return uint16(src[o]) | uint16(src[o+1]&0x0F)<<8
	}
	return uint16(src[o+1]>>4) | uint16(src[o+2])<<4
}

// D4Planes splits a row into its steps and its codes.
func D4Planes(row []byte, n int) (steps, codes []byte) { return D4PlanesN(row, n, D4Bits) }

// D4PlanesN is the same at a given code width.
func D4PlanesN(row []byte, n, bits int) (steps, codes []byte) {
	nb := n / D4Block
	return row[:nb*2], row[nb*2 : nb*D4BlockBytes(bits)]
}

// DequantizeD4G expands one row of n weights. out must hold n floats.
func DequantizeD4G(w []byte, n int, out []float32) { DequantizeD4GN(w, n, D4Bits, out) }

// DequantizeD4GN expands one row at a given code width.
func DequantizeD4GN(w []byte, n, bits int, out []float32) {
	if n%D4Block != 0 {
		panic("nn: D4G rows must be a multiple of 64")
	}
	run := D4Block / 4 * bits / 8
	scales, allCodes := D4PlanesN(w, n, bits)
	for b := 0; b*D4Block < n; b++ {
		lo, hi := d4Steps[scales[b*2]], d4Steps[scales[b*2+1]]
		codes := allCodes[b*run : (b+1)*run]
		dst := out[b*D4Block : (b+1)*D4Block]
		for i := 0; i < 16; i++ {
			d := lo
			if i*4 >= D4SubBlock {
				d = hi
			}
			p := d4Points[d4CodeAt(codes, i, bits)]
			dst[i*4+0] = float32(p[0]) * d
			dst[i*4+1] = float32(p[1]) * d
			dst[i*4+2] = float32(p[2]) * d
			dst[i*4+3] = float32(p[3]) * d
		}
	}
}

// matVecD4GRows computes y = W x for D4G weights against float32 activations
// that Prepare has already scaled and rotated.
func matVecD4GRows(w []byte, b *Batch, cols, bits int, ys [][]float32, start, end int) {
	stride := cols / D4Block * D4BlockBytes(bits)
	row := make([]float32, cols)
	for r := start; r < end; r++ {
		DequantizeD4GN(w[r*stride:(r+1)*stride], cols, bits, row)
		for c := 0; c < b.Size; c++ {
			ys[c][r] = DotF32(row, b.F[c])
		}
	}
}
