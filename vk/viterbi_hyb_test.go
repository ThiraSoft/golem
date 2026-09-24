package vk

import (
	"math"
	"math/rand"
	"testing"

	"github.com/ThiraSoft/golem/compress"
	"github.com/ThiraSoft/golem/nn"
)

// TestPairViterbiMatchesCPU holds the card's H3G encoder to the processor's on
// the contract TestViterbiMatchesCPU gives the one-weight trellis: a path of
// the same cost, sequence by sequence, not the same path. It also holds the
// card to its own path: the states it returns chain as the format says, and
// decode to the reconstruction it wrote.
func TestPairViterbiMatchesCPU(t *testing.T) {
	d := open(t)
	defer d.Close()

	const n = PairGPUSeq * 256
	rg := rand.New(rand.NewSource(9))
	src := make([]float32, n)
	for i := range src {
		src[i] = float32(rg.NormFloat64())
	}
	// Any codebook will do for the contract; a Gaussian one exercises the
	// same near-ties a trained one has.
	book := make([]float32, 2*nn.H3GEntries)
	for i := range book {
		book[i] = float32(rg.NormFloat64())
	}
	o := compress.PairOpts{K: PairGPUK, L: PairGPUL, Seq: PairGPUSeq}

	want := append([]float32(nil), src...)
	compress.QuantizePairsPath(want, o, book, make([]uint16, n/2))

	e, err := NewPairEncoder(d, n, book)
	if err != nil {
		t.Fatal(err)
	}
	defer e.Close()
	got := append([]float32(nil), src...)
	states := make([]uint16, n/2)
	if err := e.QuantizePath(got, states); err != nil {
		t.Fatal(err)
	}

	var same int
	var worst float64
	for q := 0; q < n; q += PairGPUSeq {
		var ec, eg float64
		for i := q; i < q+PairGPUSeq; i++ {
			if got[i] == want[i] {
				same++
			}
			dc := float64(src[i] - want[i])
			dg := float64(src[i] - got[i])
			ec += dc * dc
			eg += dg * dg
		}
		if r := math.Abs(eg-ec) / ec; r > worst {
			worst = r
		}
	}
	t.Logf("%.1f%% of weights identical; worst sequence cost differs by %.2g", 100*float64(same)/n, worst)
	if worst > 1e-5 {
		t.Fatalf("a sequence's cost differs by %.2g between card and processor", worst)
	}

	for i, s := range states {
		if i%(PairGPUSeq/2) != 0 {
			if prev := states[i-1]; s>>(2*PairGPUK) != prev&(1<<(PairGPUL-2*PairGPUK)-1) {
				t.Fatalf("pair %d: state %#x does not follow %#x", i, s, prev)
			}
		}
		e := nn.H3GEntry(s)
		if got[2*i] != book[2*e] || got[2*i+1] != book[2*e+1] {
			t.Fatalf("pair %d: the reconstruction is not what its state decodes to", i)
		}
	}
}
