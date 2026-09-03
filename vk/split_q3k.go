package vk

// Q3_K on the card.
//
// This format is here for a reason none of the others are: nothing in golem
// writes it, and its own compressed forms beat it — a .golem T3G carries more
// of the model at the same three bits. It is here because a client who will
// not compress a checkpoint, or cannot, downloads a Q3_K_M and expects it to
// run. Refusing it refuses the model over a packing.
//
// What was missing was only the packing. nn/dequant_q3_k.go has read the
// format on the processor for as long as golem has been measured against
// llama.cpp, and vk/quantproduct.go's kernels already read the two K-quants a
// Q3_K_M mixes in around it — so a Q3_K_S is 252 tensors of one form the card
// could not read and one it could, and a Q3_K_M is that plus Q4_K and Q5_K.
// This file is the form.

import (
	"encoding/binary"

	"github.com/ThiraSoft/golem/nn"
)

// rowBytesQ3_K is what one row of that many inputs occupies once splitQ3_K has
// had it: eighteen bytes of superblock header and ninety-six of magnitudes,
// rounded up to a whole number of words.
//
// A hundred and fourteen against the format's own hundred and ten. The four
// are the scales: the file packs sixteen of them six bits at a time into
// twelve bytes, and a shader that unpacked them would do it once per block for
// every row of the matrix. Written out as signed bytes they are read the way
// Q6_K's are, by the same three lines, and the row is still a quarter smaller
// than the same matrix in Q4_K.
//
// The rounding is rowBytesQ6_K's, for its reason and not by coincidence: a
// hundred and fourteen is not a multiple of four either, so a matrix with an
// odd number of superblocks to a row would start every other row off a word.
func rowBytesQ3_K(cols int) int { return (cols/nn.SuperBlock*114 + 3) &^ 3 }

// q3kBlockBase is where a row's blocks begin: past its scales and magnitudes,
// on a word. It is q6kBlockBase's arithmetic because the header is the same
// shape — sixteen signed scales and one fp16 to a superblock.
func q3kBlockBase(nsb int) int { return (18*nsb + 3) &^ 3 }

// splitQ3_K rewrites a Q3_K matrix into blocks of thirty-two a thread can take.
//
// The file's order is interleaved twice over. A superblock's sixty-four qs
// bytes are two windows of thirty-two, and each window carries *four* blocks
// of thirty-two weights at four bit positions — block b reads its low two bits
// at shift 2*(b%4) of window b/4. Its third bit comes from somewhere else
// again: hmask is thirty-two bytes shared by all eight blocks, one bit each,
// and block b is bit b of every one of them. So a lane that wants one block
// touches thirty-six bytes spread over ninety-six to collect twelve, and it
// does that for every row of the matrix.
//
// A row becomes
//
//	[0,      16nsb)  the sixteen group scales, six bits unpacked to a signed
//	                 byte and already less thirty-two
//	[16nsb,  18nsb)  the fp16 magnitudes, one a superblock
//	[q3kBlockBase, ..)  twelve bytes a block of thirty-two: a word of low bit
//	                 pairs for the first sixteen weights, a word for the second
//	                 sixteen, and a word of third bits — bit l for weight l,
//	                 bit 16+l for weight 16+l
//
// which the shader reads as three loads a block against the activation's eight.
//
// The magnitudes are left three bits wide rather than spread to nibbles, which
// they would fit in — nought to seven. Nibbles would put the row on Q4_K's
// path with no unpacking at all, and cost twenty-eight per cent more of the
// memory bus. At the widths a token takes, the bus is the whole cost and the
// unpacking is free, so the wider form would be slower *and* larger. The one
// case it would win is a card with ALU to spare and bandwidth to burn, which
// is not the card a Q3_K checkpoint is chosen for.
//
// The recentring is not folded in. A Q3_K magnitude is nought to seven and its
// weight is that less four, which would fit a signed byte the way Q6_K's does
// — but the four is subtracted from a *group* of sixteen and the two groups of
// a block carry different scales, so it cannot ride the block's own scale. It
// is the activation's own sum over each half instead, which is what the Q6_K
// branch already pays for its thirty-two.
func splitQ3_K(src []byte, rows, cols int) []byte {
	const superBytes = 110
	nsb := cols / nn.SuperBlock
	inStride := nsb * superBytes
	outStride := rowBytesQ3_K(cols)
	dst := make([]byte, rows*outStride)
	splitRows(src, dst, rows, inStride, outStride, func(in, out []byte) {
		blocks := out[q3kBlockBase(nsb):]
		for sb := 0; sb < nsb; sb++ {
			block := in[sb*superBytes : (sb+1)*superBytes]
			hmask, qs, sc := block[0:32], block[32:96], block[96:108]
			copy(out[16*nsb+2*sb:], block[108:110])

			// ggml's own six-bit packing, written out. nn/dequant_q3_k.go is
			// the same four lines and the fixture behind them.
			scales := out[16*sb : 16*sb+16]
			for i := 0; i < 4; i++ {
				scales[i] = byte(int8((sc[i]&0x0F)|((sc[8+i]>>0)&3)<<4) - 32)
				scales[4+i] = byte(int8((sc[4+i]&0x0F)|((sc[8+i]>>2)&3)<<4) - 32)
				scales[8+i] = byte(int8((sc[i]>>4)|((sc[8+i]>>4)&3)<<4) - 32)
				scales[12+i] = byte(int8((sc[4+i]>>4)|((sc[8+i]>>6)&3)<<4) - 32)
			}

			for b := 0; b < 8; b++ {
				pack := blocks[(sb*8+b)*12 : (sb*8+b)*12+12]
				qSub := qs[(b/4)*32:]
				shift := uint(2 * (b % 4))
				var w0, w1, w2 uint32
				for l := 0; l < 16; l++ {
					// Bit 2l of the word is weight l's low pair, which puts
					// four consecutive weights in a byte at bits 0, 2, 4 and 6
					// — the shape the shader spreads one to a byte.
					w0 |= uint32(qSub[l]>>shift&3) << uint(2*l)
					w1 |= uint32(qSub[16+l]>>shift&3) << uint(2*l)
					w2 |= uint32(hmask[l]>>uint(b)&1) << uint(l)
					w2 |= uint32(hmask[16+l]>>uint(b)&1) << uint(16+l)
				}
				binary.LittleEndian.PutUint32(pack[0:], w0)
				binary.LittleEndian.PutUint32(pack[4:], w1)
				binary.LittleEndian.PutUint32(pack[8:], w2)
			}
		}
	})
	return dst
}
