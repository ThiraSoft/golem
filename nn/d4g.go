package nn

// D4G: golem's own weight format. 64 weights in 26 bytes, 3.25 bits each.
//
// A row is two planes: every block's fp16 scale, then every block's sixteen
// twelve-bit codes. Interleaving them would put a block at byte 26b, which is
// two-byte aligned, and a shader reads words — so the planes are split for the
// same reason vk/mixture.go splits Q5_K, and the row costs exactly what the
// interleaved form would.
//
// A code names a point
// of the D4 lattice — the four-dimensional checkerboard, integer vectors of
// even coordinate sum — within a squared radius of 40. There are 3961 such
// points, so twelve bits address them, and four weights come back from one
// code.
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
	"encoding/binary"
	"math"
	"sort"
)

const (
	// D4Block is how many weights share one scale and one run of codes.
	D4Block = 64
	// d4BlockBytes is one fp16 scale then sixteen twelve-bit codes.
	d4BlockBytes = 26
	// D4Radius bounds the shell: only points of squared norm at most this are
	// addressable, and the count of them is what sets the code width.
	D4Radius = 40
	// D4Bits is the width of one code.
	D4Bits = 12
)

// d4Points is the shell in canonical order — by squared norm, then
// lexicographically — so that an index means the same thing to every encoder
// and every decoder. Four int8 a point.
var d4Points [][4]int8

// d4Index maps a point back to its place in that order.
var d4Index map[[4]int8]uint16

// D4Table is the shell as a flat float32 array, four to a point, in the order
// the codes index. This is what a kernel uploads.
func D4Table() []float32 {
	out := make([]float32, len(d4Points)*4)
	for i, p := range d4Points {
		for j := 0; j < 4; j++ {
			out[i*4+j] = float32(p[j])
		}
	}
	return out
}

// D4Points is how many points the shell holds.
func D4Points() int { return len(d4Points) }

func init() {
	lim := int(math.Sqrt(float64(D4Radius)))
	for a := -lim; a <= lim; a++ {
		for b := -lim; b <= lim; b++ {
			for c := -lim; c <= lim; c++ {
				for d := -lim; d <= lim; d++ {
					if (a+b+c+d)&1 != 0 {
						continue
					}
					if a*a+b*b+c*c+d*d > D4Radius {
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
	d4Index = make(map[[4]int8]uint16, len(d4Points))
	for i, p := range d4Points {
		d4Index[p] = uint16(i)
	}
}

// D4Code returns the code naming a lattice point, and whether the point is on
// the shell at all. A point outside it has no code: the quantizer is
// responsible for not producing one.
func D4Code(p [4]int8) (uint16, bool) {
	i, ok := d4Index[p]
	return i, ok
}

// D4Point expands a code.
func D4Point(code uint16) [4]int8 { return d4Points[code] }

// PutD4Codes packs sixteen codes into twenty-four bytes, two codes to three.
func PutD4Codes(dst []byte, codes []uint16) {
	for i := 0; i+1 < len(codes); i += 2 {
		a, b := codes[i], codes[i+1]
		o := i / 2 * 3
		dst[o] = byte(a)
		dst[o+1] = byte(a>>8&0x0F) | byte(b&0x0F)<<4
		dst[o+2] = byte(b >> 4)
	}
}

// d4CodeAt reads one of the sixteen codes of a packed run.
func d4CodeAt(src []byte, i int) uint16 {
	o := i / 2 * 3
	if i&1 == 0 {
		return uint16(src[o]) | uint16(src[o+1]&0x0F)<<8
	}
	return uint16(src[o+1]>>4) | uint16(src[o+2])<<4
}

// D4Planes splits a row into its scales and its codes.
func D4Planes(row []byte, n int) (scales, codes []byte) {
	nb := n / D4Block
	return row[:nb*2], row[nb*2 : nb*26]
}

// DequantizeD4G expands one row of n weights. out must hold n floats.
func DequantizeD4G(w []byte, n int, out []float32) {
	if n%D4Block != 0 {
		panic("nn: D4G rows must be a multiple of 64")
	}
	scales, allCodes := D4Planes(w, n)
	for b := 0; b*D4Block < n; b++ {
		d := halfToFloat(binary.LittleEndian.Uint16(scales[b*2:]))
		codes := allCodes[b*24 : (b+1)*24]
		dst := out[b*D4Block : (b+1)*D4Block]
		for i := 0; i < 16; i++ {
			p := d4Points[d4CodeAt(codes, i)]
			dst[i*4+0] = float32(p[0]) * d
			dst[i*4+1] = float32(p[1]) * d
			dst[i*4+2] = float32(p[2]) * d
			dst[i*4+3] = float32(p[3]) * d
		}
	}
}

// matVecD4GRows computes y = W x for D4G weights against float32 activations
// that Prepare has already scaled and rotated.
func matVecD4GRows(w []byte, b *Batch, cols int, ys [][]float32, start, end int) {
	stride := cols / D4Block * d4BlockBytes
	row := make([]float32, cols)
	for r := start; r < end; r++ {
		DequantizeD4G(w[r*stride:(r+1)*stride], cols, row)
		for c := 0; c < b.Size; c++ {
			ys[c][r] = DotF32(row, b.F[c])
		}
	}
}
