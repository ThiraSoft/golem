package vk

// Q2_K on the card.
//
// The tier below Q3_K, and here for the same reason: a client who will not
// compress a checkpoint downloads what llama.cpp writes. A Qwen3-4B-Q2_K is
// 144 tensors of this format, 72 of Q3_K, 36 of Q4_K and a Q6_K embedding — so
// with Q3_K already read, this one format is the whole of what is left.
//
// It is Q3_K's packing without the third bit and with a minimum instead of the
// recentring. A superblock is sixteen bytes each holding a four-bit scale and a
// four-bit minimum, sixty-four bytes of two-bit quants walked exactly as Q3_K
// walks its own, and the two fp16 those nibbles are measured in. A weight is
// d*scale*q less dmin*minimum, all four of those unsigned and none of them
// biased, which makes this the only K-quant here with no offset anywhere.

import (
	"encoding/binary"

	"github.com/ThiraSoft/golem/nn"
)

// rowBytesQ2_K is what one row of that many inputs occupies once splitQ2_K has
// had it, which is what it occupies in the file: eighty-four bytes a superblock
// either way.
//
// The reordering moves bytes and packs nothing differently, because there is
// nothing to unpack — the scales stay two nibbles to a byte and are split in
// the shader, where it is two instructions and saves four bytes a superblock on
// the bus. Eighty-four is twenty-one words, so unlike Q3_K's hundred and
// fourteen and Q6_K's two hundred and ten there is no row to round up.
func rowBytesQ2_K(cols int) int { return cols / nn.SuperBlock * 84 }

// q2kBlockBase is where a row's blocks begin: past its scales and its fp16
// pairs, both of which are already a whole number of words a superblock.
func q2kBlockBase(nsb int) int { return 20 * nsb }

// splitQ2_K rewrites a Q2_K matrix into blocks of thirty-two a thread can take.
//
// The file's qs is interleaved the way Q3_K's is: four blocks of thirty-two
// share one window of thirty-two bytes at four bit positions, and the two
// halves of a block are sixteen bytes apart inside it. Undone here, a block is
// eight contiguous bytes.
//
// A row becomes
//
//	[0,     16nsb)  the sixteen scale-and-minimum bytes, verbatim
//	[16nsb, 20nsb)  d and dmin, one fp16 pair a superblock
//	[q2kBlockBase, ..)  eight bytes a block of thirty-two: a word of quant
//	                pairs for the first sixteen weights and a word for the
//	                second sixteen
//
// The scales are left packed where Q3_K's are unpacked, and the difference is
// what each buys. Q3_K's six-bit scales cost four masked shifts of three
// different bytes to undo, which is worth doing once here; Q2_K's are a nibble
// each, which is one mask in the shader and would cost four bytes a superblock
// on the bus to precompute — the wrong trade on the side of the bus.
func splitQ2_K(src []byte, rows, cols int) []byte {
	const superBytes = 84
	nsb := cols / nn.SuperBlock
	stride := nsb * superBytes
	dst := make([]byte, rows*stride)
	splitRows(src, dst, rows, stride, stride, func(in, out []byte) {
		blocks := out[q2kBlockBase(nsb):]
		for sb := 0; sb < nsb; sb++ {
			block := in[sb*superBytes : (sb+1)*superBytes]
			copy(out[16*sb:], block[0:16])
			copy(out[16*nsb+4*sb:], block[80:84])

			qs := block[16:80]
			for b := 0; b < 8; b++ {
				pack := blocks[(sb*8+b)*8 : (sb*8+b)*8+8]
				qSub := qs[(b/4)*32:]
				shift := uint(2 * (b % 4))
				var w0, w1 uint32
				for l := 0; l < 16; l++ {
					// Bit 2l of the word is weight l's pair, which puts four
					// consecutive weights in a byte at bits 0, 2, 4 and 6 —
					// the shape shaders/matvec.comp spreads one to a byte.
					w0 |= uint32(qSub[l]>>shift&3) << uint(2*l)
					w1 |= uint32(qSub[16+l]>>shift&3) << uint(2*l)
				}
				binary.LittleEndian.PutUint32(pack[0:], w0)
				binary.LittleEndian.PutUint32(pack[4:], w1)
			}
		}
	})
	return dst
}
