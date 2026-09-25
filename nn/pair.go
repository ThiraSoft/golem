package nn

import "github.com/ThiraSoft/golem/internal/pairbook"

// H3G and H4G: the trellis with two weights a state, at three and four bits a
// weight. 128 weights in 51 and 67 bytes.
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
// H4G is the same at four bits: a fifteen-bit state moving eight bits a pair,
// so every window starts on a byte, and 4096 pairs picked by bits 4..15 of h.
// 15 + 63·8 = 519 bits, 65 bytes with one bit unused: T4G's 67 bytes a block.
// At four bits the pair code starts behind — a table of 2048 pairs and a
// fourteen-bit state read 0.45 dB under T4G untrained — and the wider state
// and table are what bring it level before training.
//
// H3G's codebook is 2048 pairs of half-precision numbers, eight kibibytes, the
// same for every tensor of every model. It was trained once, by Lloyd's
// algorithm run through the trellis itself on a unit Gaussian — a path, then
// every entry becomes the mean of the pairs it coded — and it lives in
// internal/pairbook; cmd/paircodebook is what wrote it. QTIP keeps a table of
// 512 pairs and a sign bit; a free table of 2048 measured the same and costs
// the decoder one instruction less.
//
// A file carries the table it was written with, under golem.<tier>.codebook,
// and tensors refuses one whose table is not this build's.

// PairTier is one tier of the pair trellis: everything a reader, a writer and
// an encoder need to agree on.
type PairTier struct {
	Quant    Quant
	L        int // state bits: the window a pair is decoded from
	K        int // bits a weight; a state adds 2K
	SeqBytes int // one sequence of 128 weights, 64 states
	Entries  int // pairs in the codebook
	Shift    int // the entry is bits Shift.. of s·(s+1)
	book     []float32
	half     []uint16
}

// CodebookHalves is the codebook as the half-precision bits a file carries.
func (p *PairTier) CodebookHalves() []uint16 { return p.half }

const (
	// H3GL, H3GK: H3G's state and rate.
	H3GL = 14
	H3GK = 3
	// H4GL, H4GK: H4G's.
	H4GL = 15
	H4GK = 4
)

var (
	h3gTier = &PairTier{Quant: H3G, L: H3GL, K: H3GK, SeqBytes: 49, Entries: 2048, Shift: 5}
	h4gTier = &PairTier{Quant: H4G, L: H4GL, K: H4GK, SeqBytes: 65, Entries: 4096, Shift: 4}
)

func init() {
	for _, t := range []struct {
		tier *PairTier
		half []uint16
	}{{h3gTier, pairbook.H3G[:]}, {h4gTier, pairbook.H4G[:]}} {
		t.tier.half = t.half
		t.tier.book = make([]float32, len(t.half))
		for i, h := range t.half {
			t.tier.book[i] = halfToFloat(h)
		}
	}
}

// PairTierOf is the tier of a pair-trellis format, or nil for any other.
func PairTierOf(q Quant) *PairTier {
	switch q {
	case H3G:
		return h3gTier
	case H4G:
		return h4gTier
	}
	return nil
}

// Codebook is the tier's codebook as float32 pairs, every value a
// half-precision number exactly, because that is how a kernel holds it.
func (p *PairTier) Codebook() []float32 { return p.book }

// Entry is which codebook entry a state reads.
func (p *PairTier) Entry(s uint16) int { return PairEntry(s, p.Shift, p.Entries) }

// PairEntry is Entry for a codebook that is not a tier's yet: the trainer's.
func PairEntry(s uint16, shift, entries int) int {
	h := uint32(s) * (uint32(s) + 1)
	return int(h >> uint(shift) & uint32(entries-1))
}

// Value is the pair a state reconstructs, before the block's step.
func (p *PairTier) Value(s uint16) (float32, float32) {
	e := p.Entry(s)
	return p.book[2*e], p.book[2*e+1]
}

// RowBytes is what one row of n weights occupies: the steps, then the codes.
func (p *PairTier) RowBytes(n int) int { return n/T4GBlock + n/T4GSeq*p.SeqBytes }

// Planes splits a row into its steps and its codes.
func (p *PairTier) Planes(row []byte, n int) (steps, codes []byte) {
	ns := n / T4GBlock
	return row[:ns], row[ns:p.RowBytes(n)]
}

// PutStates writes one sequence's path, sixty-four states. The order is T4G's,
// most significant bit first: the state is the last L bits of the stream, so
// each is its predecessor shifted up by 2K with 2K new bits at the bottom, and
// a window read at offset 2K·t is exactly that. Bits past the path are zero.
func (p *PairTier) PutStates(dst []byte, states []uint16) {
	if len(states) != T4GSeq/2 || len(dst) < p.SeqBytes {
		panic("nn: a pair sequence is 64 states")
	}
	for i := range dst[:p.SeqBytes] {
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
	put(0, p.L, uint32(states[0]))
	for t := 1; t < T4GSeq/2; t++ {
		put(p.L+(t-1)*2*p.K, 2*p.K, uint32(states[t])&(1<<(2*p.K)-1))
	}
}

// StateAt reads the state of pair t out of a sequence. A window of fourteen
// or fifteen bits at an even offset spans three bytes at most.
func (p *PairTier) StateAt(codes []byte, t int) uint16 {
	at := t * 2 * p.K
	v := uint32(codes[at>>3])<<16 | uint32(codes[at>>3+1])<<8
	if n := at>>3 + 2; n < len(codes) {
		v |= uint32(codes[n])
	}
	return uint16(v >> uint(24-p.L-(at&7)) & (1<<p.L - 1))
}

// Dequantize expands one row of n weights. out must hold n floats.
func (p *PairTier) Dequantize(w []byte, n int, out []float32) {
	if n%T4GSeq != 0 {
		panic("nn: trellis rows must be a multiple of 128")
	}
	steps, codes := p.Planes(w, n)
	for s := 0; s*T4GSeq < n; s++ {
		seq := codes[s*p.SeqBytes : (s+1)*p.SeqBytes]
		dst := out[s*T4GSeq : (s+1)*T4GSeq]
		for t := 0; t < T4GSeq/2; t++ {
			d := t4gSteps[steps[s*t4gStepsPerSeq+2*t/T4GBlock]]
			a, b := p.Value(p.StateAt(seq, t))
			dst[2*t] = a * d
			dst[2*t+1] = b * d
		}
	}
}

// matVecPairRows computes y = W x for a pair tier against float32 activations
// that Prepare has already scaled and rotated.
func (p *PairTier) matVecRows(w []byte, b *Batch, cols int, ys [][]float32, start, end int) {
	stride := p.RowBytes(cols)
	row := make([]float32, cols)
	for r := start; r < end; r++ {
		p.Dequantize(w[r*stride:(r+1)*stride], cols, row)
		for c := 0; c < b.Size; c++ {
			ys[c][r] = DotF32(row, b.F[c])
		}
	}
}

// HalfToFloat widens a half-precision number, for the codebook's generator.
func HalfToFloat(h uint16) float32 { return halfToFloat(h) }
