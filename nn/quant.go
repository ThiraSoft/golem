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
	Q4_0
	Q6_K
)

func (q Quant) String() string {
	switch q {
	case F32:
		return "F32"
	case BF16:
		return "BF16"
	case Q4_0:
		return "Q4_0"
	case Q6_K:
		return "Q6_K"
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
	case "Q4_0":
		return Q4_0, true
	case "Q6_K":
		return Q6_K, true
	}
	return 0, false
}

const (
	// QuantBlock is the block size shared by Q4_0 and Q8_0.
	QuantBlock = 32
	// q4_0BlockBytes is one fp16 scale followed by 32 nibbles.
	q4_0BlockBytes = 18
	// SuperBlock is the block size of the K-quants.
	SuperBlock = 256
	// q6_kBlockBytes is 128 low nibbles, 64 high pairs, 16 scales, one fp16.
	q6_kBlockBytes = 210
)
