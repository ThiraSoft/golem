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
	p := GolemParams{ScaleBlock: 64, HadGroup: 0}

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
	// The bench reads 17.05 dB at three bits on a Gaussian with the window
	// primed. A short row pays the priming on fewer weights, so hold this well
	// below that and let the end-to-end measurement be the real check: what is
	// being detected here is a codec wired to the wrong rate, which reads far
	// worse than 14.
	if snr < 14 {
		t.Fatalf("three-bit trellis reads %.2f dB, which is not a three-bit codec", snr)
	}
}

// TestT3GOptsAreTheThreeBitRate guards the one field a copy-paste gets wrong.
//
// The expected BPW is not 1/8 over three: BPW() is BitsPerSeq()/Seq, and
// BitsPerSeq() is the path only — (Seq-1)*K + L, with no byte padding and no
// step codes — because that is what the trellis itself costs, before the
// format rounds a sequence up to a whole number of bytes. For T3G that is
// (128-1)*3 + 12 = 393 bits over 128 weights, or 3.0703125 bpw; the padded
// 400 bits (50 bytes) that T3GSeqBytes actually writes is a format decision
// BPW does not see.
func TestT3GOptsAreTheThreeBitRate(t *testing.T) {
	o := T4GOptsFor(nn.T3G)
	if o.K != nn.T3GK || o.Seq != nn.T4GSeq || o.L != nn.T4GL {
		t.Fatalf("T3G opts are %+v", o)
	}
	const want = 393.0 / 128.0
	if bpw := o.BPW(); math.Abs(bpw-want) > 1e-9 {
		t.Fatalf("the path is %v bits a weight, not %v", bpw, want)
	}
}
