package stt

// Turning the trunk's weights into something the bus can carry.
//
// At one frame every eighty milliseconds and a batch of one, every weight in
// the trunk is read once to be multiplied once. Nothing is reused, so nothing
// stays in cache, and the speed of the transcriber is the speed at which its
// 1.64 GiB of bfloat16 crosses from memory — measured at 20.7 GB/s of the
// 38.5 GB/s this machine can stream, which is where the 0.8x of real time came
// from.
//
// Q4_0 is four and a half bits where bfloat16 is sixteen: a quarter of the
// traffic, and the format nn/dot_q4_0.go has written its widest kernels for.
// The conversion happens once, at load, because the file ships bfloat16 and
// there is no reason to ask anyone to keep a second copy of a checkpoint.

import (
	"encoding/binary"
	"math"

	"github.com/ThiraSoft/golem/nn"
)

// quantizeQ4_0 converts a row-major bfloat16 matrix into Q4_0, block by block
// of thirty-two: one half-precision scale, then sixteen bytes holding two
// four-bit values each, value j in the low nibble and value j+16 in the high
// one. It is llama.cpp's layout, which is what nn's kernels and the Vulkan
// shaders both read.
func quantizeQ4_0(bf16 []byte, rows, cols int) []byte {
	if cols%nn.QuantBlock != 0 {
		panic("stt: a Q4_0 row must be a whole number of blocks")
	}
	blocks := cols / nn.QuantBlock
	const blockBytes = 18
	out := make([]byte, rows*blocks*blockBytes)

	source := func(r, i int) float32 { return bf16At(bf16, r*cols+i) }

	// One row per unit of work: the rows are independent, and a matrix of this
	// size takes long enough that handing them out beats doing them in turn.
	nn.InParallel(rows, rows*cols, func(start, end int) {
		var values [nn.QuantBlock]float32
		for r := start; r < end; r++ {
			for b := 0; b < blocks; b++ {
				for i := range values {
					values[i] = source(r, b*nn.QuantBlock+i)
				}
				block := out[(r*blocks+b)*blockBytes:]
				quantizeBlockQ4_0(values[:], block[:blockBytes])
			}
		}
	})
	return out
}

// quantizeQ8_0 converts a row-major bfloat16 matrix into Q8_0: 34 bytes to a
// block of thirty-two, a half-precision scale then one signed byte a weight.
// Twice the bytes of Q4_0 and four more bits of mantissa a weight, for whoever
// would rather pay the bus than the precision.
func quantizeQ8_0(bf16 []byte, rows, cols int) []byte {
	if cols%nn.QuantBlock != 0 {
		panic("stt: a Q8_0 row must be a whole number of blocks")
	}
	blocks := cols / nn.QuantBlock
	const blockBytes = 34
	out := make([]byte, rows*blocks*blockBytes)

	nn.InParallel(rows, rows*cols, func(start, end int) {
		for r := start; r < end; r++ {
			for b := 0; b < blocks; b++ {
				var amax float32
				at := r*cols + b*nn.QuantBlock
				for i := 0; i < nn.QuantBlock; i++ {
					v := bf16At(bf16, at+i)
					if a := float32(math.Abs(float64(v))); a > amax {
						amax = a
					}
				}
				// The grid runs from -128 to 127 and is symmetric about zero;
				// dividing by 127 rather than by 128 keeps it that way, which
				// is what llama.cpp does and what its shaders expect back.
				scale := amax / 127
				var inverse float32
				if scale != 0 {
					inverse = 1 / scale
				}
				block := out[(r*blocks+b)*blockBytes:]
				binary.LittleEndian.PutUint16(block, nn.FloatToHalf(scale))
				for i := 0; i < nn.QuantBlock; i++ {
					q := int(math.Round(float64(bf16At(bf16, at+i) * inverse)))
					if q > 127 {
						q = 127
					}
					if q < -128 {
						q = -128
					}
					block[2+i] = byte(int8(q))
				}
			}
		}
	})
	return out
}

// bf16At reads one weight out of the mapped bytes.
func bf16At(bf16 []byte, i int) float32 {
	return math.Float32frombits(uint32(binary.LittleEndian.Uint16(bf16[i*2:])) << 16)
}

// quantizeBlockQ4_0 writes one block. The scale is taken from the value of
// largest magnitude and divided by -8, not by 8: the grid runs from -8 to 7,
// so the extreme value has to land on -8 to be represented at all. Getting the
// sign wrong here costs a factor of eight on the largest weight of every block
// and reads as a model that has forgotten how to speak.
func quantizeBlockQ4_0(values []float32, out []byte) {
	var amax, extreme float32
	for _, v := range values {
		if a := float32(math.Abs(float64(v))); a > amax {
			amax, extreme = a, v
		}
	}
	scale := extreme / -8
	var inverse float32
	if scale != 0 {
		inverse = 1 / scale
	}
	binary.LittleEndian.PutUint16(out, nn.FloatToHalf(scale))
	for j := 0; j < nn.QuantBlock/2; j++ {
		low := clampNibble(values[j]*inverse + 8.5)
		high := clampNibble(values[j+16]*inverse + 8.5)
		out[2+j] = low | high<<4
	}
}

func clampNibble(v float32) byte {
	q := int(v)
	if q < 0 {
		q = 0
	}
	if q > 15 {
		q = 15
	}
	return byte(q)
}
