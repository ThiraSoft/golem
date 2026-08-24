//go:build linux && arm64

package nn

import (
	"encoding/binary"
	"os"
)

// The auxiliary vector the kernel leaves above the stack, which /proc exposes
// as a sequence of uint64 pairs: a type, then a value, ending at a type of
// zero. Type 16 is AT_HWCAP, the first word of capability bits, and bit 20 of
// it is HWCAP_ASIMDDP — the advanced SIMD dot product, which is what SDOT is.
//
// Reading it here rather than through golang.org/x/sys/cpu keeps the promise in
// README.md that this module needs Go and nothing else.
const (
	atHWCAP       = 16
	hwcapASIMDDP  = 1 << 20
	auxvEntrySize = 16
)

var dotprod = probeDotProd()

func probeDotProd() bool {
	auxv, err := os.ReadFile("/proc/self/auxv")
	if err != nil {
		// No /proc is not a reason to run wrong code. The portable path is
		// slower and always correct.
		return false
	}
	for i := 0; i+auxvEntrySize <= len(auxv); i += auxvEntrySize {
		key := binary.LittleEndian.Uint64(auxv[i:])
		val := binary.LittleEndian.Uint64(auxv[i+8:])
		if key == 0 {
			break
		}
		if key == atHWCAP {
			return val&hwcapASIMDDP != 0
		}
	}
	return false
}
