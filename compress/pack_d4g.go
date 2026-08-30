package compress

// The last step of encoding a matrix, on its own because two encoders share it.
//
// EncodeD4G does it inline: it has the codes of a row in registers when it
// finishes them, and there is no reason to walk the row twice. A card does not
// — vk/encode_d4g.go returns one code per four weights and one step per
// thirty-two, wide, because a shader that packed twelve-bit fields would cost
// more to write than the packing costs to do here.

import "github.com/ThiraSoft/golem/nn"

// PackD4G writes the file's rows from the codes and steps of a whole matrix:
// one code per four weights, one step per thirty-two, both in the order the
// weights are in.
func PackD4G(codes []uint16, steps []byte, rows, cols, bits int) []byte {
	if cols%nn.D4Block != 0 {
		panic("compress: a D4G row must be a multiple of 64 wide")
	}
	if len(codes) != rows*cols/4 || len(steps) != rows*cols/nn.D4SubBlock {
		panic("compress: the codes and the steps do not describe that matrix")
	}
	run := nn.D4Block / 4 * bits / 8
	rowBytes := cols / nn.D4Block * nn.D4BlockBytes(bits)
	out := make([]byte, rows*rowBytes)
	Parallel(rows, func(lo, hi int) {
		for r := lo; r < hi; r++ {
			scales, codeRun := nn.D4PlanesN(out[r*rowBytes:(r+1)*rowBytes], cols, bits)
			copy(scales, steps[r*cols/nn.D4SubBlock:(r+1)*cols/nn.D4SubBlock])
			rowCodes := codes[r*cols/4 : (r+1)*cols/4]
			for b := 0; b*nn.D4Block < cols; b++ {
				nn.PutD4CodesN(codeRun[b*run:], rowCodes[b*16:(b+1)*16], bits)
			}
		}
	})
	return out
}
