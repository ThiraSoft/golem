package vk

import (
	"math"
	"math/rand"
	"testing"

	"github.com/ThiraSoft/golem/compress"
)

// What the card owes the processor here is a path of the same cost, not the
// same path — and that is a weaker contract than the lattice encoder's on
// purpose.
//
// TestEncodeD4GMatchesCPU demands byte equality because a D4 code is an index
// into a shared enumeration: two encoders that disagree write different files
// for the same weights. A trellis records the path it chose, and the decoder is
// a pure function of the bits it reads, so any minimum-cost path is an equally
// valid file. Two of them are not a disagreement.
//
// And they will differ, unavoidably. 1MAD has 1021 distinct values for 4096
// states, so exact and near ties are everywhere, and a Viterbi is a chain of
// comparisons: one unit in the last place anywhere moves whole paths. Writing
// the scale as a multiply rather than a division — see mad1Scale — recovered a
// point and a half of agreement; the rest is the driver contracting
// `best + d*d` into a fused multiply-add, which `precise` does not survive -O.
// None of it is visible in the answer: the two reconstructions land within a
// few parts in a hundred million of each other, which is what this asserts.
func TestViterbiMatchesCPU(t *testing.T) {
	d, err := Open()
	if err != nil {
		t.Skip(err)
	}
	defer d.Close()

	const n = TrellisGPUSeq * 512
	rg := rand.New(rand.NewSource(9))
	src := make([]float32, n)
	for i := range src {
		src[i] = float32(rg.NormFloat64())
	}
	const gain = 0.94

	want := append([]float32(nil), src...)
	compress.QuantizeTrellis(want, compress.TrellisOpts{
		K: TrellisGPUK, L: TrellisGPUL, Seq: TrellisGPUSeq,
		Gain: gain, Code: compress.Code1MAD})

	e, err2 := NewTrellisEncoder(d, n)
	if err2 != nil {
		t.Fatal(err2)
	}
	defer e.Close()
	got := append([]float32(nil), src...)
	if err := e.Quantize(got, gain); err != nil {
		t.Fatal(err)
	}

	var same int
	var errCPU, errGPU float64
	for i := range src {
		if got[i] == want[i] {
			same++
		}
		dc := float64(src[i] - want[i])
		dg := float64(src[i] - got[i])
		errCPU += dc * dc
		errGPU += dg * dg
	}
	// Per sequence, so a difference cannot hide in an aggregate.
	var worst float64
	var differ int
	for q := 0; q+TrellisGPUSeq <= n; q += TrellisGPUSeq {
		var ec, eg float64
		for i := q; i < q+TrellisGPUSeq; i++ {
			dc := float64(src[i] - want[i])
			dg := float64(src[i] - got[i])
			ec += dc * dc
			eg += dg * dg
		}
		if ec != eg {
			differ++
			if r := math.Abs(eg-ec) / ec; r > worst {
				worst = r
			}
		}
	}
	t.Logf("%d of %d sequences differ in cost at all; worst by %.3g relative",
		differ, n/TrellisGPUSeq, worst)
	t.Logf("errors: processor %.10g, card %.10g", errCPU, errGPU)

	agree := float64(same) / float64(n) * 100
	dbCPU := 10 * math.Log10(float64(n)/errCPU)
	dbGPU := 10 * math.Log10(float64(n)/errGPU)
	t.Logf("%.2f%% of weights identical; %.3f dB on the processor, %.3f dB on the card", agree, dbCPU, dbGPU)
	// Agreement is reported because a collapse in it would mean the kernel had
	// stopped running the algorithm at all, but it is not the contract.
	if agree < 80 {
		t.Errorf("only %.2f%% of weights agree: the card is not running the processor's algorithm", agree)
	}
	if rel := math.Abs(errGPU-errCPU) / errCPU; rel > 1e-5 {
		t.Errorf("the card's path costs %.9g against the processor's %.9g, %.3g apart",
			errGPU, errCPU, rel)
	}
	if math.Abs(dbCPU-dbGPU) > 0.005 {
		t.Errorf("the card reconstructs at %.3f dB against the processor's %.3f", dbGPU, dbCPU)
	}
}
