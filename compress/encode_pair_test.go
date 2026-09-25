package compress

import (
	"math"
	"math/rand"
	"testing"

	"github.com/ThiraSoft/golem/nn"
)

// TestEncodePairsDecodesToACodecOfItsRate is the join between the codec and
// the file: what the Viterbi chose has to be what the decoder reads back, and
// at the rate the tier claims. A codec wired to the wrong state, table or
// stride reads far under these floors.
func TestEncodePairsDecodesToACodecOfItsRate(t *testing.T) {
	for _, c := range []struct {
		q     nn.Quant
		floor float64 // dB; a unit Gaussian reads 17.6 at three bits, 23.4 at four
	}{{nn.H3G, 15}, {nn.H4G, 20}} {
		const rows, cols = 3, 256
		rng := rand.New(rand.NewSource(7))
		w := make([]float32, rows*cols)
		for i := range w {
			w[i] = float32(rng.NormFloat64())
		}
		data := EncodeGolem(w, rows, cols, nil, GolemParams{ScaleBlock: 64}, c.q)
		p := nn.PairTierOf(c.q)
		if want := rows * p.RowBytes(cols); len(data) != want {
			t.Fatalf("%s: %d bytes, want %d", c.q, len(data), want)
		}
		out := make([]float32, cols)
		stride := p.RowBytes(cols)
		var num, den float64
		for r := 0; r < rows; r++ {
			p.Dequantize(data[r*stride:(r+1)*stride], cols, out)
			for i := range out {
				d := float64(out[i] - w[r*cols+i])
				num += d * d
				den += float64(w[r*cols+i]) * float64(w[r*cols+i])
			}
		}
		if snr := 10 * math.Log10(den/num); snr < c.floor {
			t.Fatalf("%s reads %.2f dB, which is not a codec of its rate", c.q, snr)
		}
	}
}
