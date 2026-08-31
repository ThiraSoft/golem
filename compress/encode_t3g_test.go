package compress

import (
	"math"
	"math/rand"
	"testing"

	"github.com/ThiraSoft/golem/nn"
)

// TestEncodeT3GDecodesToWhatItReconstructed is the join between the codec and
// the file: what the Viterbi chose has to be what the decoder reads back. Not
// approximately — the states are integers and the planes hold them or they do
// not.
func TestEncodeT3GDecodesToWhatItReconstructed(t *testing.T) {
	const rows, cols = 3, 256
	rng := rand.New(rand.NewSource(7))
	w := make([]float32, rows*cols)
	for i := range w {
		w[i] = float32(rng.NormFloat64())
	}
	q := make([]float32, cols)
	for i := range q {
		q[i] = 1
	}
	p := D4Params{ScaleBlock: 64, HadGroup: 0}

	data := EncodeT4GAs(w, rows, cols, q, p, nn.T3G)
	if want := rows * nn.T4GRowBytesN(cols, nn.T3G); len(data) != want {
		t.Fatalf("%d bytes, want %d", len(data), want)
	}

	out := make([]float32, cols)
	stride := nn.T4GRowBytesN(cols, nn.T3G)
	var num, den float64
	for r := 0; r < rows; r++ {
		nn.DequantizeT4GN(data[r*stride:(r+1)*stride], cols, nn.T3G, out)
		for c := 0; c < cols; c++ {
			d := float64(out[c] - w[r*cols+c])
			num += d * d
			den += float64(w[r*cols+c]) * float64(w[r*cols+c])
		}
	}
	snr := 10 * math.Log10(den/num)
	// The bench reads 17.05 dB at three bits on a Gaussian. Tail-biting takes
	// the Viterbi's free choice of start away, so hold this well below that and
	// let the end-to-end measurement be the real check: what is being detected
	// here is a codec wired to the wrong rate, which reads far worse than 14.
	if snr < 14 {
		t.Fatalf("three-bit trellis reads %.2f dB, which is not a three-bit codec", snr)
	}
}

// TestT3GOptsAreTheThreeBitRate guards the one field a copy-paste gets wrong.
//
// BPW() is BitsPerSeq()/Seq, the path alone, with no step codes: three bits a
// weight exactly, because the tail-biting sequence is 128 symbols and nothing
// else. The other two tiers still pay L−k to prime a window, and this is the
// difference that pays for the second Viterbi pass.
func TestT3GOptsAreTheThreeBitRate(t *testing.T) {
	o := T4GOptsFor(nn.T3G)
	if o.K != nn.T3GK || o.Seq != nn.T4GSeq || o.L != nn.T4GL || !o.TailBiting {
		t.Fatalf("T3G opts are %+v", o)
	}
	if bpw := o.BPW(); math.Abs(bpw-3) > 1e-9 {
		t.Fatalf("the path is %v bits a weight, not 3", bpw)
	}
	if o := T4GOptsFor(nn.T4G); o.TailBiting {
		t.Fatal("the four-bit tier does not tail-bite: L is not a multiple of k")
	}
}

// TestT3GPathsClose is the encoder's half of the format's one new invariant.
// The decoder reads the last three weights of a sequence through a window that
// wraps, so the path the file records has to be a cycle whatever the search
// found — closeTail is what guarantees it, and this is what says the
// guarantee holds on real weights and not only in the arithmetic.
func TestT3GPathsClose(t *testing.T) {
	const seqs = 64
	rng := rand.New(rand.NewSource(11))
	z := make([]float32, seqs*nn.T4GSeq)
	for i := range z {
		z[i] = float32(rng.NormFloat64())
	}
	o := T4GOptsFor(nn.T3G)
	states := make([]uint16, len(z))
	before := TrellisTailOpen.Load()
	QuantizeTrellisPath(z, o, states)
	t.Logf("%d of %d sequences did not close on the first two passes",
		TrellisTailOpen.Load()-before, seqs)

	dst := make([]byte, nn.T3GSeqBytes)
	for s := 0; s < seqs; s++ {
		path := states[s*nn.T4GSeq : (s+1)*nn.T4GSeq]
		if path[0]>>nn.T3GK != path[nn.T4GSeq-1]&(1<<uint(nn.T4GL-nn.T3GK)-1) {
			t.Fatalf("sequence %d: the ring does not close", s)
		}
		nn.PutT4GStatesN(dst, path, nn.T3G)
		for i, want := range path {
			if got := nn.T4GStateAtN(dst, i, nn.T3G); got != want {
				t.Fatalf("sequence %d weight %d: read back %#x, want %#x", s, i, got, want)
			}
		}
	}
}
