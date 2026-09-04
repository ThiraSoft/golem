package nn

import (
	"unsafe"
)

// dequantizeQ8_0Row expands one Q8_0 row into out.
func dequantizeQ8_0Row(w []byte, n int, out []float32) {
	nb := n / QuantBlock
	for b := 0; b < nb; b++ {
		block := w[b*q8_0BlockBytes : (b+1)*q8_0BlockBytes]
		dBits := *(*uint16)(unsafe.Pointer(&block[0]))
		d := halfToFloat(dBits)
		qs := unsafe.Slice((*int8)(unsafe.Pointer(&block[2])), 32)
		dst := out[b*QuantBlock : (b+1)*QuantBlock]
		for i := 0; i < 32; i++ {
			dst[i] = float32(qs[i]) * d
		}
	}
}
