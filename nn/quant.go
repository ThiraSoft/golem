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
	Q3_K
	Q4_K
	Q5_K
	Q6_K
	Q8_0
	// T4G is the trellis: 128 weights in 67 bytes, 4.1875 bits each, and a
	// codebook that is computed rather than looked up. nn/t4g.go describes it.
	T4G
	// T5G is the same trellis a bit a weight wider — 128 weights in 83 bytes,
	// 5.1875 bits each. It exists for the logit head, which is not a hidden
	// layer whose error is absorbed downstream but the thing that makes the
	// logits, and which llama.cpp's K-quant mixes have always given more bits
	// than the rest of the model.
	T5G
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
	case T4G, T5G:
		return true
	}
	return false
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
	case Q3_K:
		return "Q3_K"
	case Q4_K:
		return "Q4_K"
	case Q5_K:
		return "Q5_K"
	case Q6_K:
		return "Q6_K"
	case Q8_0:
		return "Q8_0"
	case T4G:
		return "T4G"
	case T5G:
		return "T5G"
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
	case "Q3_K":
		return Q3_K, true
	case "Q4_K":
		return Q4_K, true
	case "Q5_K":
		return Q5_K, true
	case "Q6_K":
		return Q6_K, true
	case "Q8_0":
		return Q8_0, true
	case "T4G":
		return T4G, true
	case "T5G":
		return T5G, true
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
	// q3_kBlockBytes is 32 bytes of high-bit mask, 64 bytes of two-bit
	// quants, 12 packed six-bit scales, and one fp16 super-scale.
	q3_kBlockBytes = 110
	// q4_kBlockBytes is 2 fp16 scales (d, dmin), 12 scales bytes, and 128 nibbles.
	q4_kBlockBytes = 144
	// q5_kBlockBytes is 2 fp16 scales (d, dmin), 12 scales bytes, 32 high bits, and 128 nibbles.
	q5_kBlockBytes = 176
	// q6_kBlockBytes is 128 low nibbles, 64 high pairs, 16 scales, one fp16.
	q6_kBlockBytes = 210
	// q8_0BlockBytes is one fp16 scale followed by 32 int8 quants.
	q8_0BlockBytes = 34
)
