package nn

// H3G: the trellis at three bits a weight, two weights a state. 128 weights in
// 51 bytes, 3.1875 bits each.
//
// T3G spends most of its product on one thing: every weight cuts its own
// twelve-bit window out of the stream and looks its value up. This is QTIP's
// answer to that (Tseng et al., arXiv:2406.11235, the HYB code of
// lib/codebook/bitshift.py): a state reconstructs a *pair* of weights, so the
// stream moves six bits a state and a window is cut, hashed and looked up once
// for two weights. The pair is read out of a small table rather than computed,
// which is the "hybrid" of the name — a hash picks the entry, a table holds it.
//
// What that costs is the trellis's memory. A state that covers two weights
// shifts twice as far, so at the same twelve bits of state a path remembers
// half as many weights back, and the Gaussian bench loses 0.45 dB to T3G. The
// state is fourteen bits here for that reason: at fourteen it is level with
// T3G's twelve (17.40 dB against 17.39 on a unit Gaussian at the same rate),
// and at sixteen it is 0.06 dB ahead for a Viterbi four times the size.
//
// Per sequence of 128 weights:
//
//   - two step codes, one for each 64 weights, exactly as T3G's: eight bits
//     each on the same grid, fitted by least squares after the path.
//   - 392 bits of code: fourteen for the first pair, six for each of the 63
//     after it. Exactly 49 bytes, and no padding at all.
//
//     392/128 + 8/64 = 3.0625 + 0.125 = 3.1875 bits a weight.
//
// Decoding a pair is random access:
//
//	s     = the fourteen bits at 6t within the sequence
//	h     = s·(s+1) mod 2^32
//	pair t = step · H3GCodebook[bits 5..15 of h]
//
// The codebook is 2048 pairs of half-precision numbers, eight kibibytes, the
// same for every tensor of every model. It was trained once, by Lloyd's
// algorithm run through the trellis itself on a unit Gaussian — a path, then
// every entry becomes the mean of the pairs it coded — and it lives in
// nn/h3g_codebook.go; cmd/h3gcodebook is what wrote it. QTIP keeps a table of
// 512 pairs and a sign bit; a free table of 2048 measured the same and costs
// the decoder one instruction less.

const (
	// H3GL is the state: the width of the window a pair is decoded from.
	H3GL = 14
	// H3GK is the bits a weight adds to the stream, so a pair adds twice this.
	H3GK = 3
	// H3GSeqBytes is 14 + 63·6 = 392 bits, which is 49 bytes.
	H3GSeqBytes = (H3GL + (T4GSeq/2-1)*2*H3GK) / 8
	// H3GEntries is how many pairs the codebook holds.
	H3GEntries = 2048
)

// h3gCodebook is the codebook expanded to float32 once. Every value is a
// half-precision number exactly, because that is how a kernel holds it.
var h3gCodebook [2 * H3GEntries]float32

func init() {
	for i, h := range h3gCodebookHalf {
		h3gCodebook[i] = halfToFloat(h)
	}
}

// H3GCodebook is the codebook as float32 pairs, for a caller that uploads it.
func H3GCodebook() []float32 { return h3gCodebook[:] }

// H3GEntry is which codebook entry a state reads.
func H3GEntry(s uint16) int {
	h := uint32(s) * (uint32(s) + 1)
	return int(h >> 5 & (H3GEntries - 1))
}

// H3GValue is the pair a state reconstructs, before the block's step.
func H3GValue(s uint16) (float32, float32) {
	e := H3GEntry(s)
	return h3gCodebook[2*e], h3gCodebook[2*e+1]
}

// H3GTable is every state's pair, for the Viterbi: 2^14 pairs.
func H3GTable() []float32 {
	t := make([]float32, 2<<H3GL)
	for s := 0; s < 1<<H3GL; s++ {
		t[2*s], t[2*s+1] = H3GValue(uint16(s))
	}
	return t
}

// H3GRowBytes is what one row of n weights occupies: the steps, then the codes.
// A multiple of four whenever n is a multiple of 512.
func H3GRowBytes(n int) int { return n/T4GBlock + n/T4GSeq*H3GSeqBytes }

// H3GPlanes splits a row into its steps and its codes.
func H3GPlanes(row []byte, n int) (steps, codes []byte) {
	ns := n / T4GBlock
	return row[:ns], row[ns:H3GRowBytes(n)]
}

// PutH3GStates writes one sequence's path, sixty-four states, into 49 bytes.
// The order is T4G's, most significant bit first: the state is the last
// fourteen bits of the stream, so each is its predecessor shifted up by six
// with six new bits at the bottom, and a window read at offset 6t is exactly
// that.
func PutH3GStates(dst []byte, states []uint16) {
	if len(states) != T4GSeq/2 || len(dst) < H3GSeqBytes {
		panic("nn: an H3G sequence is 64 states")
	}
	for i := range dst[:H3GSeqBytes] {
		dst[i] = 0
	}
	put := func(at, width int, v uint32) {
		for i := width - 1; i >= 0; i-- {
			if v>>uint(i)&1 != 0 {
				dst[at>>3] |= 0x80 >> uint(at&7)
			}
			at++
		}
	}
	put(0, H3GL, uint32(states[0]))
	for t := 1; t < T4GSeq/2; t++ {
		put(H3GL+(t-1)*2*H3GK, 2*H3GK, uint32(states[t])&(1<<(2*H3GK)-1))
	}
}

// H3GStateAt reads the state of pair t out of a sequence's 49 bytes. A
// fourteen-bit window at an even offset spans three bytes at most; the last
// one starts at bit 378 and ends at 391, the sequence's last bit.
func H3GStateAt(codes []byte, t int) uint16 {
	at := t * 2 * H3GK
	v := uint32(codes[at>>3])<<16 | uint32(codes[at>>3+1])<<8
	if n := at>>3 + 2; n < len(codes) {
		v |= uint32(codes[n])
	}
	return uint16(v >> uint(10-(at&7)) & (1<<H3GL - 1))
}

// DequantizeH3G expands one row of n weights. out must hold n floats.
func DequantizeH3G(w []byte, n int, out []float32) {
	if n%T4GSeq != 0 {
		panic("nn: trellis rows must be a multiple of 128")
	}
	steps, codes := H3GPlanes(w, n)
	for s := 0; s*T4GSeq < n; s++ {
		seq := codes[s*H3GSeqBytes : (s+1)*H3GSeqBytes]
		dst := out[s*T4GSeq : (s+1)*T4GSeq]
		for t := 0; t < T4GSeq/2; t++ {
			d := t4gSteps[steps[s*t4gStepsPerSeq+2*t/T4GBlock]]
			a, b := H3GValue(H3GStateAt(seq, t))
			dst[2*t] = a * d
			dst[2*t+1] = b * d
		}
	}
}

// matVecH3GRows computes y = W x for H3G weights against float32 activations
// that Prepare has already scaled and rotated.
func matVecH3GRows(w []byte, b *Batch, cols int, ys [][]float32, start, end int) {
	stride := H3GRowBytes(cols)
	row := make([]float32, cols)
	for r := start; r < end; r++ {
		DequantizeH3G(w[r*stride:(r+1)*stride], cols, row)
		for c := 0; c < b.Size; c++ {
			ys[c][r] = DotF32(row, b.F[c])
		}
	}
}

// HalfToFloat widens a half-precision number, for the codebook's generator.
func HalfToFloat(h uint16) float32 { return halfToFloat(h) }
