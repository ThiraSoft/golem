package yue2

// Philox4x32-10 PRNG as used by cuRAND and Random123.
// The YuE2 token sampler draws categorical tokens using the Gumbel-max trick
// driven by uniform floats generated from this Philox counter sequence.

const (
	philoxM0 = 0xD2511F53
	philoxM1 = 0xCD9E8D57
	philoxW0 = 0x9E3779B9
	philoxW1 = 0xBB67AE85
)

// Philox4x32_10 applies 10 rounds of the 4x32 Philox bijection to counter ctr
// with key (k0, k1), returning the 4 resulting 32-bit words.
func Philox4x32_10(ctr [4]uint32, k0, k1 uint32) [4]uint32 {
	for round := 0; round < 10; round++ {
		prod0 := uint64(philoxM0) * uint64(ctr[0])
		hi0 := uint32(prod0 >> 32)
		lo0 := uint32(prod0)

		prod1 := uint64(philoxM1) * uint64(ctr[2])
		hi1 := uint32(prod1 >> 32)
		lo1 := uint32(prod1)

		ctr[0] = hi1 ^ ctr[1] ^ k0
		ctr[1] = lo1
		ctr[2] = hi0 ^ ctr[3] ^ k1
		ctr[3] = lo0

		if round < 9 {
			k0 += philoxW0
			k1 += philoxW1
		}
	}
	return ctr
}
