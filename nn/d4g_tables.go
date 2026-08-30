package nn

// What an encoder needs of the lattice that a reader does not.
//
// The decoder's direction — a code to a point — is in nn/d4g.go beside the
// format it belongs to. An encoder needs the other one, and it needs the steps
// as the numbers D4Step returns rather than as an exponential it recomputes:
// a card that lands a bit away from the processor writes a different file, and
// the two would be two formats sharing a name.

import "math"

// D4InverseTable is the other direction, for a kernel that has to name the
// point it just rounded to. D4CodeN answers from a map, which a shader has
// none of; this is the same answer as a flat array indexed by the coordinates
// themselves, which is the one lookup a card can afford.
//
// lim is the largest coordinate a point of the tier can have — every point
// past it is past the shell too, since lim² alone exceeds the edge norm — and
// side is 2·lim+1. The entry for a point outside the tier is D4NoCode.
func D4InverseTable(bits int) (lim, side int, table []uint32) {
	_, edge := D4TierNorms(bits)
	lim = int(math.Sqrt(float64(edge)))
	side = 2*lim + 1
	table = make([]uint32, side*side*side*side)
	for i := range table {
		table[i] = D4NoCode
	}
	for i, pt := range d4Points[:1<<bits] {
		at := 0
		for j := 0; j < 4; j++ {
			at = at*side + int(pt[j]) + lim
		}
		table[at] = uint32(i)
	}
	return lim, side, table
}

// D4NoCode is what the inverse table holds where the tier has no point.
const D4NoCode = 0xFFFFFFFF

// D4StepTable is what the eight bits of a step code name, all 256 of them, so
// that a kernel reads the same numbers D4Step returns rather than recomputing
// an exponential and landing a bit away.
func D4StepTable() []float32 {
	out := make([]float32, len(d4Steps))
	copy(out, d4Steps[:])
	return out
}
