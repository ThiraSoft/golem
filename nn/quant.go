package nn

// The weight formats golem reads, and the geometry of their blocks.
//
// A quantized format stores its weights in fixed-size blocks, each with its own
// scale. Nothing here dequantizes a whole matrix: the kernels read the blocks
// where they lie, which is what keeps the bandwidth down.

// Quant names a weight format.
type Quant uint8

const (
	F32 Quant = iota
	BF16
	F16
	Q4_0
	Q4_1
	Q4_K
	Q5_K
	Q6_K
	Q8_0
	// D4G is golem's own: 64 weights in 26 bytes, a twelve-bit lattice code
	// every four. nn/d4g.go describes it.
	D4G
	// D4G16 is the same format with sixteen-bit codes — 34 bytes a block, a
	// whole bit a weight more, and a table sixteen times larger.
	D4G16
	// L8G is D4G's container with a scalar codebook: three bits a weight over
	// Lloyd's eight levels, in the same twenty-six bytes and with no table at
	// all. nn/l8g.go says what that costs.
	L8G
	// T4G is the trellis: 128 weights in 67 bytes, 4.1875 bits each, and a
	// codebook that is computed rather than looked up. nn/t4g.go describes it.
	T4G
)

// Golem says the format is one of golem's own — a matrix stored as A·(q ⊙ W),
// whose activation therefore has to meet the site's vector and the rotation
// before the product, and whose row has to be brought back through both when it
// is read on its own.
//
// It is asked rather than inferred from the code width, which is how a fourth
// format that has no lattice code at all came to need a name for the question.
func (q Quant) Golem() bool {
	switch q {
	case D4G, D4G16, L8G, T4G:
		return true
	}
	return false
}

// D4Width is the code width of a D4G format, and zero for anything else.
func (q Quant) D4Width() int {
	switch q {
	case D4G:
		return D4Bits
	case D4G16:
		return D4Bits16
	case L8G:
		return L8Bits
	}
	return 0
}

func (q Quant) String() string {
	switch q {
	case F32:
		return "F32"
	case BF16:
		return "BF16"
	case F16:
		return "F16"
	case Q4_0:
		return "Q4_0"
	case Q4_1:
		return "Q4_1"
	case Q4_K:
		return "Q4_K"
	case Q5_K:
		return "Q5_K"
	case Q6_K:
		return "Q6_K"
	case Q8_0:
		return "Q8_0"
	case D4G:
		return "D4G"
	case D4G16:
		return "D4G16"
	case L8G:
		return "L8G"
	case T4G:
		return "T4G"
	}
	return "unknown"
}

// QuantOf maps a dtype name, as the tensor readers report it, onto a Quant.
func QuantOf(dtype string) (Quant, bool) {
	switch dtype {
	case "F32":
		return F32, true
	case "BF16":
		return BF16, true
	case "F16":
		return F16, true
	case "Q4_0":
		return Q4_0, true
	case "Q4_1":
		return Q4_1, true
	case "Q4_K":
		return Q4_K, true
	case "Q5_K":
		return Q5_K, true
	case "Q6_K":
		return Q6_K, true
	case "Q8_0":
		return Q8_0, true
	case "D4G":
		return D4G, true
	case "D4G16":
		return D4G16, true
	case "L8G":
		return L8G, true
	case "T4G":
		return T4G, true
	}
	return 0, false
}

const (
	// QuantBlock is the block size shared by Q4_0 and Q8_0.
	QuantBlock = 32
	// q4_0BlockBytes is one fp16 scale followed by 32 nibbles.
	q4_0BlockBytes = 18
	// q4_1BlockBytes is an fp16 scale and an fp16 minimum, then 32 nibbles.
	q4_1BlockBytes = 20
	// SuperBlock is the block size of the K-quants.
	SuperBlock = 256
	// q4_kBlockBytes is 2 fp16 scales (d, dmin), 12 scales bytes, and 128 nibbles.
	q4_kBlockBytes = 144
	// q5_kBlockBytes is 2 fp16 scales (d, dmin), 12 scales bytes, 32 high bits, and 128 nibbles.
	q5_kBlockBytes = 176
	// q6_kBlockBytes is 128 low nibbles, 64 high pairs, 16 scales, one fp16.
	q6_kBlockBytes = 210
	// q8_0BlockBytes is one fp16 scale followed by 32 int8 quants.
	q8_0BlockBytes = 34
)
