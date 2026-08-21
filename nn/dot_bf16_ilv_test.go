package nn

import (
	"math"
	"math/rand"
	"testing"
)

// The interleaved product against the plain one, on the shapes the tower has.
func TestMatMatBF16IlvAgreesWithThePlainOne(t *testing.T) {
	if !Interleaved(768) {
		t.Skip("no interleaved kernel on this machine")
	}
	r := rand.New(rand.NewSource(31))
	for _, c := range []struct{ outputs, inputs, batch int }{
		{16, 16, 1}, {16, 16, 7}, {32, 768, 13}, {768, 768, 40}, {17, 32, 5},
	} {
		w := make([]byte, c.outputs*c.inputs*2)
		for i := range w {
			w[i] = byte(r.Intn(256))
		}
		// A weight of a sane magnitude, so the comparison is about the layout
		// rather than about denormals.
		u := unsafeWords(w)
		for i := range u {
			u[i] = uint16(math.Float32bits(r.Float32()*2-1) >> 16)
		}

		plain := make([]float32, c.batch*c.inputs)
		for i := range plain {
			plain[i] = r.Float32()*2 - 1
		}
		ilv := make([]float32, len(plain))
		for l := 0; l < c.batch; l++ {
			for g := 0; g < c.inputs; g += 16 {
				for i, from := range InterleaveOrder {
					ilv[l*c.inputs+g+i] = plain[l*c.inputs+g+from]
				}
			}
		}

		want := make([]float32, c.batch*c.outputs)
		MatMatBF16(w, plain, c.outputs, c.inputs, c.batch, want)
		got := make([]float32, c.batch*c.outputs)
		MatMatBF16IlvRows(w, ilv, c.outputs, c.inputs, c.batch, got, 0, c.outputs)

		for i := range want {
			if diff := got[i] - want[i]; diff > 1e-3 || diff < -1e-3 {
				t.Fatalf("%dx%d batch %d: element %d is %g, expected %g",
					c.outputs, c.inputs, c.batch, i, got[i], want[i])
			}
		}
	}
}
