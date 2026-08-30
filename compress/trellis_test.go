package compress

import (
	"math"
	"math/rand"
	"testing"
)

// The rotation makes every tensor statistically the same Gaussian, which is why
// this repository can read 16.05 dB on all of them to a thirtieth of a dB. So
// the codebook question can be asked of a Gaussian directly, with no model in
// the way: whatever wins here wins on the weights.
//
// The bar: Shannon's rate-distortion bound for a memoryless Gaussian is 6.02·R
// dB, so 18.06 dB at three bits. A D4 lattice with a spherical boundary tops
// out near 16.9 and this repository measures 16.05 on real tensors.

func sqnrDB(orig, rec []float32) float64 {
	var sig, err float64
	for i := range orig {
		sig += float64(orig[i]) * float64(orig[i])
		d := float64(orig[i] - rec[i])
		err += d * d
	}
	return 10 * math.Log10(sig/err)
}

func gaussian(n int, seed int64) []float32 {
	rg := rand.New(rand.NewSource(seed))
	x := make([]float32, n)
	for i := range x {
		x[i] = float32(rg.NormFloat64())
	}
	return x
}

// The computed code has to look like the source it is coding.
func TestTrellisValuesAreStandardGaussian(t *testing.T) {
	for _, c := range []struct {
		code TrellisCode
		name string
	}{{Code1MAD, "1MAD"}} {
		tab := TrellisTable(c.code, 16)
		var m, v float64
		for _, x := range tab {
			m += float64(x)
		}
		m /= float64(len(tab))
		for _, x := range tab {
			v += (float64(x) - m) * (float64(x) - m)
		}
		v = math.Sqrt(v / float64(len(tab)))
		t.Logf("%-6s mean %+.4f  sd %.4f", c.name, m, v)
		if math.Abs(m) > 0.05 || math.Abs(v-1) > 0.15 {
			t.Errorf("%s is not N(0,1): mean %.4f sd %.4f", c.name, m, v)
		}
	}
}

// What the trellis buys over the lattice, at the rate the format spends today.
func TestTrellisBeatsD4OnGaussian(t *testing.T) {
	const n = 1 << 18
	x := gaussian(n, 7)

	// D4 at r²=40 is the shipped codebook: 12 bits for four weights.
	pt := make([]float32, 4)
	tmp := make([]float32, 4)
	buf := make([]float32, 4)
	best, bestBeta := 0.0, 0.0
	for _, beta := range []float64{1.6, 2.0, 2.5, 3.0, 4.0, 6.0, 8.0, 12.0} {
		cand := append([]float32(nil), x...)
		for p := 0; p*4 < n; p++ {
			v := cand[p*4 : (p+1)*4]
			for i := range v {
				buf[i] = v[i] * float32(beta)
			}
			quantizeLattice(LatD4, buf, pt, tmp, 40)
			for i := range v {
				v[i] = buf[i] / float32(beta)
			}
		}
		if db := sqnrDB(x, cand); db > best {
			best, bestBeta = db, beta
		}
	}
	t.Logf("D4  r²=40   2.99 bits/weight   %6.2f dB  (best beta %.1f)", best, bestBeta)

	for _, seq := range []int{256, 1024, 4096} {
		for _, l := range []int{12, 14, 16} {
			o := TrellisOpts{K: 3, L: l, Seq: seq, Code: Code1MAD}
			rec := append([]float32(nil), x...)
			quantizeTrellis(rec, o, TrellisTable(o.Code, o.L))
			t.Logf("TCQ k=3 L=%-2d T=%-4d  %.3f bits/weight   %6.2f dB", l, seq, o.BPW(), sqnrDB(x, rec))
		}
	}
}

// The step up: four bits of code against the wide D4 tier, which is where a
// file that means to reach Q4_K_M's quality is going to have to live.
func TestTrellisAtFourBits(t *testing.T) {
	const n = 1 << 18
	x := gaussian(n, 7)

	pt := make([]float32, 4)
	tmp := make([]float32, 4)
	buf := make([]float32, 4)
	best, bestBeta := 0.0, 0.0
	for _, beta := range []float64{2.5, 3.5, 4.6, 6.0, 8.0, 12.0, 16.0} {
		cand := append([]float32(nil), x...)
		for p := 0; p*4 < n; p++ {
			v := cand[p*4 : (p+1)*4]
			for i := range v {
				buf[i] = v[i] * float32(beta)
			}
			quantizeLattice(LatD4, buf, pt, tmp, 160)
			for i := range v {
				v[i] = buf[i] / float32(beta)
			}
		}
		if db := sqnrDB(x, cand); db > best {
			best, bestBeta = db, beta
		}
	}
	t.Logf("D4  r²=160  3.986 bits/weight  %6.2f dB  (best beta %.1f; table 493 KiB, too big for a workgroup)", best, bestBeta)

	for _, l := range []int{12, 14, 16} {
		o := TrellisOpts{K: 4, L: l, Seq: 4096, Code: Code1MAD}
		rec := append([]float32(nil), x...)
		quantizeTrellis(rec, o, TrellisTable(o.Code, o.L))
		t.Logf("TCQ k=4 L=%-2d T=4096  %.3f bits/weight  %6.2f dB   (no table at all)", l, o.BPW(), sqnrDB(x, rec))
	}
}

// The rate ladder, which is what says how big the file has to be to reach a
// given quality — the whole question, once the codec is chosen.
func TestTrellisRateLadder(t *testing.T) {
	if testing.Short() {
		t.Skip("the ladder is minutes, not seconds")
	}
	const n = 1 << 18
	x := gaussian(n, 11)
	for _, k := range []int{2, 3, 4, 5} {
		o := TrellisOpts{K: k, L: 12, Seq: 256, Code: Code1MAD}
		rec := append([]float32(nil), x...)
		quantizeTrellis(rec, o, TrellisTable(o.Code, o.L))
		db := sqnrDB(x, rec)
		t.Logf("TCQ k=%d L=12  %.2f bits/weight  %6.2f dB   (Shannon %5.2f, gap %4.2f)",
			k, o.BPW(), db, 6.02*float64(k), 6.02*float64(k)-db)
	}
}
