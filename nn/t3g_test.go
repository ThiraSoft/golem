package nn

import (
	"math/rand"
	"testing"
)

// TestT3GGeometry is the arithmetic the format is: 384 bits of path and two
// step codes for 128 weights, which is 50 bytes and 3.125 bits a weight. The
// tail-biting path is three bits a weight exactly and there is no padding to
// count — the twelve bits that primed a window are the last three weights'
// windows wrapping to bit zero.
func TestT3GGeometry(t *testing.T) {
	if T3GSeqBytes != 48 {
		t.Fatalf("a three-bit sequence is 48 bytes of path, not %d", T3GSeqBytes)
	}
	if got := T4GRowBytesN(T4GSeq, T3G); got != 50 {
		t.Fatalf("128 weights of T3G are 50 bytes, not %d", got)
	}
	if bpw := float64(T4GRowBytesN(1024, T3G)*8) / 1024; bpw != 3.125 {
		t.Fatalf("T3G is %v bits a weight, not 3.125", bpw)
	}
	if !T4GTailBiting(T3G) || T4GTailBiting(T4G) || T4GTailBiting(T5G) {
		t.Fatal("the narrow tier is the tail-biting one, and it is the only one")
	}
}

// t3gCycle builds a legal tail-biting path: 128 three-bit symbols read as a
// ring, weight t's state being the four symbols starting at t. Any state
// sequence was legal in the padded layout; here only cycles are.
func t3gCycle(rng *rand.Rand) []uint16 {
	sym := make([]byte, T4GSeq)
	for i := range sym {
		sym[i] = byte(rng.Intn(1 << T3GK))
	}
	states := make([]uint16, T4GSeq)
	for t := range states {
		var s uint16
		for j := 0; j < T4GL/T3GK; j++ {
			s |= uint16(sym[(t+j)%T4GSeq]) << uint(T4GL-(j+1)*T3GK)
		}
		states[t] = s
	}
	return states
}

// TestT3GPathIsACycle is the constraint tail-biting adds: the last state's
// successors include the first. A path that does not close cannot be written as
// 384 bits, and the encoder is what has to guarantee it.
func TestT3GPathIsACycle(t *testing.T) {
	states := t3gCycle(rand.New(rand.NewSource(3)))
	for i := 1; i < T4GSeq; i++ {
		if states[i]>>T3GK != states[i-1]&(1<<uint(T4GL-T3GK)-1) {
			t.Fatalf("state %d does not follow %d", i, i-1)
		}
	}
	if states[0]>>T3GK != states[T4GSeq-1]&(1<<uint(T4GL-T3GK)-1) {
		t.Fatal("the ring does not close")
	}
}

// TestT3GRoundTripIsExact is the contract the file rests on: a decoder is a
// pure function of the bits it reads, so what goes into the planes comes back
// out of them unchanged. A round trip that is not exact is a packing bug.
func TestT3GRoundTripIsExact(t *testing.T) {
	states := t3gCycle(rand.New(rand.NewSource(1)))
	dst := make([]byte, T3GSeqBytes)
	PutT4GStatesN(dst, states, T3G)
	for i := range states {
		if got := T4GStateAtN(dst, i, T3G); got != states[i] {
			t.Fatalf("weight %d read back as %#x, want %#x", i, got, states[i])
		}
	}
}

// TestT3GHalfBlockIsByteAligned is what lets the shader keep matvec_t4g.comp's
// shape: the second step block of a sequence starts at bit 3·64 = 192, which is
// byte 24, so a half is a whole number of bytes at three bits exactly as it is
// at four and five.
func TestT3GHalfBlockIsByteAligned(t *testing.T) {
	if T3GK*T4GBlock%8 != 0 {
		t.Fatalf("a half block is %d bits, which is not a whole number of bytes", T3GK*T4GBlock)
	}
	if T3GK*T4GBlock/8 != 24 {
		t.Fatalf("HALFB is %d, not 24", T3GK*T4GBlock/8)
	}
}
