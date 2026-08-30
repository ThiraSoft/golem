package nn

// T4G: golem's trellis format. 128 weights in 67 bytes, 4.1875 bits each.
//
// The codebook is a state machine and not a table. A sequence of 128 weights is
// one path through QTIP's bitshift trellis (Tseng et al., NeurIPS 2024): the
// state is the last twelve bits of the code stream, so weight t reads the
// twelve bits at offset 4t and hashes them. Nothing about the codebook is
// stored, and nothing has to be — which is the whole of why this format exists,
// and what D4G could not do at four bits, where its shell needs 493 KiB against
// the 32 a workgroup has.
//
// A row is two planes, the steps then the codes, for the reason D4G's is: a
// 67-byte block would start on an odd boundary and a shader reads words.
//
// Per sequence of 128 weights:
//
//   - two step codes, one for each 64 weights. Eight bits each, naming powers
//     of two a sixteenth apart — the same spacing D4G uses, and nn/d4g.go says
//     why a coarser one costs a whole point of perplexity. The window is not
//     D4G's: a lattice step is a fraction of its block's RMS, because the shell
//     the block is scaled into is several units across, and a trellis step is
//     the block's RMS itself. Two octaves up, and wider at the top, for that
//     reason and one measurement — a unit-variance source clipped at D4G's
//     ceiling of 0.478 and reconstructed at 5.5 dB instead of 22.6.
//   - 520 bits of code: twelve for the first weight, which needs a whole window
//     before there is any history to shift, and four for each of the 127 after
//     it. Exactly 65 bytes, which is what makes 128 the sequence length rather
//     than an arbitrary one — that, and vk/shaders/viterbi_tcq.comp, where the
//     backpointers of a longer sequence stop fitting beside the two cost planes
//     in a workgroup's sixty-four kibibytes.
//
//     520/128 + 8/64 = 4.0625 + 0.125 = 4.1875 bits a weight.
//
// The eight-bit step against an fp16 scale is measured and not assumed, because
// compress/README.md records a step grid an eighth apart costing a whole point
// of perplexity that 0.03 dB of weight error could not see. On Qwen3-0.6B, at
// k=4 and the same everything else: fp16 reads 29.8120 and KL 0.0659, the step
// code reads 29.8737 and 0.0672. Two hundredths of a point and two percent of
// the divergence, for three percent of the file.
//
// Decoding is random access and four instructions. Weight t depends on nothing
// but the bits at its own offset, so no product changes shape:
//
//	weight t = step[t/64] · value(the twelve bits at 4t within the sequence)
//
// What the block does not carry is the rest of the scheme — the per-column
// vector and the Hadamard rotation the activations meet — which belongs to the
// site and not to the block. See nn/d4g_prepare.go; T4G shares all of it.

import (
	"math"
	"sync/atomic"
)

const (
	// T4GSeq is how many weights are coded as one path. It is the effective
	// dimension of the codebook, and it is set by what the encoder's
	// backpointers fit in rather than by anything the decoder cares about.
	T4GSeq = 128
	// T4GBlock is how many weights share one step code.
	T4GBlock = 64
	// T4GK is the bits a weight adds to the code stream.
	T4GK = 4
	// T4GL is the width of the window a weight is decoded from — the state.
	T4GL = 12
	// T4GSeqBytes is (T4GSeq-1)·T4GK + T4GL bits, which is 520, which is 65.
	T4GSeqBytes = ((T4GSeq-1)*T4GK + T4GL) / 8
	// t4gStepsPerSeq is how many step codes a sequence carries.
	t4gStepsPerSeq = T4GSeq / T4GBlock
)

const (
	// The 1MAD code of compress/trellis.go, and the constants have to be these
	// ones: an encoder and a decoder that hash a state differently are two
	// formats sharing a name. compress calls this rather than keeping a second
	// copy, and vk/shaders/matvec_t4g.comp is the third — held to it by a test,
	// because a shader cannot call Go.
	t4gMulA = 34038481
	t4gAddB = 76625530
	// 1/147.8 written as the float32 it is. A division is not a multiply by the
	// reciprocal: twenty of 1MAD's 1021 values differ by one unit in the last
	// place between the two forms, and compress/trellis.go says what that cost
	// when the two sides disagreed about which.
	t4gScale = 0.00676589971
)

// T4GValue is what a twelve-bit state reconstructs, before the block's step.
// This is the decoder, and it is the whole codebook: no table, four
// instructions, and the same answer for every tensor of every model.
func T4GValue(s uint16) float32 {
	x := uint32(t4gMulA)*uint32(s) + uint32(t4gAddB)
	sum := (x & 0xff) + ((x >> 8) & 0xff) + ((x >> 16) & 0xff) + ((x >> 24) & 0xff)
	return (float32(sum) - 510) * t4gScale
}

// T4GTable is every state's value, for a caller that would rather look one up
// than compute it — the Viterbi, which reads 2^L of them per weight. A kernel
// does not want it: eight kibibytes to save four instructions is the trade this
// format exists to refuse.
func T4GTable() []float32 {
	t := make([]float32, 1<<T4GL)
	for s := range t {
		t[s] = T4GValue(uint16(s))
	}
	return t
}

// T4GRowBytes is what one row of n weights occupies: the steps, then the codes.
//
// It is a multiple of four whenever n is a multiple of 512, which every matrix
// of every language model here satisfies. A 1152-wide vision matrix is not, and
// the format takes it anyway — a shader reads its bytes out of words either
// way, and refusing a tensor to keep an offset even would cost more than the
// unaligned read does.
func T4GRowBytes(n int) int {
	return n/T4GBlock + n/T4GSeq*T4GSeqBytes
}

// T4GPlanes splits a row into its steps and its codes.
func T4GPlanes(row []byte, n int) (steps, codes []byte) {
	ns := n / T4GBlock
	return row[:ns], row[ns:T4GRowBytes(n)]
}

// PutT4GStates writes one sequence's path into 65 bytes.
//
// The stream is written most significant bit first, because that is the order
// the trellis shifts in: the state is the last twelve bits of the stream, so a
// state is its predecessor shifted up by four with four new bits at the bottom,
// and a window read MSB-first at offset 4t is exactly that. Written the other
// way round the recurrence would run backwards and the decoder would have to
// reverse every window.
//
// The first weight's whole twelve bits go in, then four bits each for the 127
// after it. states must be the path itself — each one its predecessor shifted
// up by four — which is what a Viterbi traceback produces and what
// TestT4GRoundTripIsExact holds the encoders to.
func PutT4GStates(dst []byte, states []uint16) {
	if len(states) != T4GSeq || len(dst) < T4GSeqBytes {
		panic("nn: a T4G sequence is 128 states in 65 bytes")
	}
	for i := range dst[:T4GSeqBytes] {
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
	put(0, T4GL, uint32(states[0]))
	for t := 1; t < T4GSeq; t++ {
		put(T4GL+(t-1)*T4GK, T4GK, uint32(states[t])&(1<<T4GK-1))
	}
}

// T4GStateAt reads the state of weight t out of a sequence's 65 bytes.
//
// Two bytes and a shift, and never three: every offset is a multiple of four,
// so a twelve-bit window starting at bit 4t spans two bytes whichever of the
// two alignments it has. The last weight starts at bit 508 and ends at 519,
// which is the sequence's last byte and not one past it.
func T4GStateAt(codes []byte, t int) uint16 {
	at := t * T4GK
	v := uint32(codes[at>>3])<<8 | uint32(codes[at>>3+1])
	return uint16(v >> uint(4-(at&7)) & 0xFFF)
}

// DequantizeT4G expands one row of n weights. out must hold n floats.
func DequantizeT4G(w []byte, n int, out []float32) {
	if n%T4GSeq != 0 {
		panic("nn: T4G rows must be a multiple of 128")
	}
	steps, codes := T4GPlanes(w, n)
	for q := 0; q*T4GSeq < n; q++ {
		seq := codes[q*T4GSeqBytes : (q+1)*T4GSeqBytes]
		dst := out[q*T4GSeq : (q+1)*T4GSeq]
		for t := 0; t < T4GSeq; t++ {
			d := t4gSteps[steps[q*t4gStepsPerSeq+t/T4GBlock]]
			dst[t] = T4GValue(T4GStateAt(seq, t)) * d
		}
	}
}

// matVecT4GRows computes y = W x for T4G weights against float32 activations
// that Prepare has already scaled and rotated.
func matVecT4GRows(w []byte, b *Batch, cols int, ys [][]float32, start, end int) {
	stride := T4GRowBytes(cols)
	row := make([]float32, cols)
	for r := start; r < end; r++ {
		DequantizeT4G(w[r*stride:(r+1)*stride], cols, row)
		for c := 0; c < b.Size; c++ {
			ys[c][r] = DotF32(row, b.F[c])
		}
	}
}

// t4gStepBias places the 256 codes on the exponent axis. The grid is
// 2^((c-224)/16), which runs from 6.1e-5 to 3.83 — sixteen octaves at four
// percent a step.
//
// Where it sits is measured, not chosen. The block RMS of every matrix of
// Qwen3-0.6B, rotated, spans 0.0011 to 0.27, eight octaves; the salience scales
// columns by at most 24 either way before the rotation mixes them, and the
// remaining headroom is for that. A block that lands on either end is counted
// and said out loud rather than clipped quietly — see T4GStepCode — because a
// step that saturates is exactly the failure this format has already produced
// once, at 5.5 dB on a source whose RMS was one.
const t4gStepBias = 224

var t4gSteps [256]float32

func init() {
	for c := range t4gSteps {
		t4gSteps[c] = float32(math.Exp2((float64(c) - t4gStepBias) / 16))
	}
}

// T4GStepClipped counts the blocks whose step landed on an end of the grid.
// Zero on every model this has been pointed at; anything else means the window
// is in the wrong place for that checkpoint, and the file is worse than the
// codec it was written with.
var T4GStepClipped atomic.Int64

// T4GStepCode is the code nearest a step. A step at or below zero is the
// smallest code: it belongs to a block that is all but zero, and no path
// through the trellis reaches further down than the grid does.
func T4GStepCode(v float32) byte {
	if !(v > 0) {
		return 0
	}
	c := math.Round(math.Log2(float64(v))*16 + t4gStepBias)
	if c < 0 {
		T4GStepClipped.Add(1)
		c = 0
	}
	if c > 255 {
		T4GStepClipped.Add(1)
		c = 255
	}
	return byte(c)
}

// T4GStep expands a step code.
func T4GStep(c byte) float32 { return t4gSteps[c] }

// T4GStepBias is where the grid sits, for a kernel that expands a code with an
// exp2 rather than a table.
const T4GStepBias = t4gStepBias
