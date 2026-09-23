package vk

// PQ2_0 on the card.
//
// A Prism ternary format packed at group size 128: one fp16 scale followed by
// 32 bytes of two-bit codes (four 32-weight sub-blocks, 8 bytes each).
//
// In the file, the scale precedes its 32 bytes of codes. On the card, a row is
// rearranged so that every block access is word-aligned:
//
//	[0, scaleBytes)       the fp16 scales of its 128-blocks, packed two to a uint
//	                      and padded to a whole word (4 bytes per pair of blocks)
//	[scaleBytes, rowBytes) eight bytes per 32-weight sub-block (the file's qs, unchanged)
//
// Because the scales are separated from the quants, both the scales and the
// quants begin on 4-byte word boundaries. Within the quants region, each
// 32-weight sub-block occupies exactly 8 contiguous bytes (two words).

import "github.com/ThiraSoft/golem/nn"

// rowBytesPQ2_0 is what one row of that many inputs occupies once splitPQ2_0
// has had it: the scales packed two to a uint and padded to a whole word,
// followed by 32 bytes of quants per 128-block.
func rowBytesPQ2_0(cols int) int {
	ntb := cols / nn.TernaryBlock
	scaleBytes := ((ntb + 1) / 2) * 4
	return scaleBytes + ntb*32
}

// pq20BlockBase is where a row's quants begin: past the packed fp16 scales.
func pq20BlockBase(ntb int) int {
	return ((ntb + 1) / 2) * 4
}

// splitPQ2_0 rewrites a PQ2_0 matrix into the card layout: all fp16 scales
// first, followed by the quants in 8-byte sub-blocks of 32 weights.
func splitPQ2_0(src []byte, rows, cols int) []byte {
	const blockBytes = 34
	ntb := cols / nn.TernaryBlock
	inStride := ntb * blockBytes
	outStride := rowBytesPQ2_0(cols)
	dst := make([]byte, rows*outStride)
	scaleBytes := pq20BlockBase(ntb)

	splitRows(src, dst, rows, inStride, outStride, func(in, out []byte) {
		blocks := out[scaleBytes:]
		for tb := 0; tb < ntb; tb++ {
			block := in[tb*blockBytes : (tb+1)*blockBytes]
			copy(out[2*tb:], block[:2])
			copy(blocks[tb*32:], block[2:34])
		}
	})
	return dst
}
