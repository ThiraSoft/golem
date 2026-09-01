package nn

import (
	"math/rand"
	"testing"
)

// TestT3GGeometry is the arithmetic the format is: 400 bits of path and two
// step codes for 128 weights, which is 52 bytes and 3.25 bits a weight — the
// same 3.25 the D4G lattice cost, to the bit.
func TestT3GGeometry(t *testing.T) {
	if T3GSeqBytes != 50 {
		t.Fatalf("a three-bit sequence is 50 bytes of path, not %d", T3GSeqBytes)
	}
	if got := T4GRowBytesN(T4GSeq, T3G); got != 52 {
		t.Fatalf("128 weights of T3G are 52 bytes, not %d", got)
	}
	if bpw := float64(T4GRowBytesN(1024, T3G)*8) / 1024; bpw != 3.25 {
		t.Fatalf("T3G is %v bits a weight, not 3.25", bpw)
	}
}

// TestT3GRoundTripIsExact is the contract the file rests on: a decoder is a
// pure function of the bits it reads, so what goes into the planes comes back
// out of them unchanged. A round trip that is not exact is a packing bug.
func TestT3GRoundTripIsExact(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	states := make([]uint16, T4GSeq)
	states[0] = uint16(rng.Intn(1 << T4GL))
	for i := 1; i < T4GSeq; i++ {
		states[i] = (states[i-1]<<T3GK | uint16(rng.Intn(1<<T3GK))) & (1<<T4GL - 1)
	}
	dst := make([]byte, T3GSeqBytes)
	PutT4GStatesN(dst, states, T3G)
	for i := range states {
		if got := T4GStateAtN(dst, i, T3G); got != states[i] {
			t.Fatalf("weight %d read back as %#x, want %#x", i, got, states[i])
		}
	}
}

// TestT3GPaddingIsNotRead is the only thing this layout adds to the format and
// the only thing a reader can read by mistake. Bits 393 to 399 of a sequence
// belong to no weight: weight 127's window ends at bit 392.
func TestT3GPaddingIsNotRead(t *testing.T) {
	rng := rand.New(rand.NewSource(2))
	states := make([]uint16, T4GSeq)
	states[0] = uint16(rng.Intn(1 << T4GL))
	for i := 1; i < T4GSeq; i++ {
		states[i] = (states[i-1]<<T3GK | uint16(rng.Intn(1<<T3GK))) & (1<<T4GL - 1)
	}
	clean := make([]byte, T3GSeqBytes)
	PutT4GStatesN(clean, states, T3G)

	dirty := append([]byte(nil), clean...)
	dirty[T3GSeqBytes-1] |= 0x7F // the seven bits past weight 127's window

	for i := range states {
		if T4GStateAtN(dirty, i, T3G) != T4GStateAtN(clean, i, T3G) {
			t.Fatalf("weight %d moved when the padding was set", i)
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
