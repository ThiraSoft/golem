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
)

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
