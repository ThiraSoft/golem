package nn

// GELU on the error function, in float32, eight at a time.
//
// GELURange computes it in float64 with math.Erf, which is exact and costs
// some twenty nanoseconds a value: an encoder's feed forward gates a few
// million of them a pass, and on laya's processor path that was a tenth of
// the time. GELUErf computes 0.5·x·erfc(-x/√2) with Numerical Recipes'
// erfc, whose relative error is under 1.2e-7 everywhere, and an exponential
// by range reduction and a polynomial of degree six. In float32 the whole is
// within 5e-7 of the exact GELU relative to the value above one and absolute
// below it (nn/gelu_erf_test.go holds it there): the negative tail's
// exponent is a sum of terms near ten, which float32 carries to a few parts
// in ten million. The kernel on amd64 takes eight lanes at a time; the tail
// and other machines take the same formula one value at a time.

import "math"

// GELUErf replaces every x[i] with GELU(x[i]).
func GELUErf(x []float32) {
	n := len(x) &^ 7
	if n > 0 && avx2 {
		geluErfAVX2(&x[0], n)
	} else {
		n = 0
	}
	for i := n; i < len(x); i++ {
		x[i] = geluErf1(x[i])
	}
}

// The constants of both paths.
const (
	erfLog2e  = 1.44269504088896341
	erfLn2Hi  = 0.693359375
	erfLn2Lo  = -2.12194440e-4
	erfExpMin = -87.0
)

// expF32 is e^v for v in [-87, 0], as the kernel computes it.
func expF32(v float32) float32 {
	v = max(v, erfExpMin)
	k := float32(math.Floor(float64(v*erfLog2e + 0.5)))
	r := v - k*erfLn2Hi - k*erfLn2Lo
	p := float32(1.0 / 720)
	p = p*r + 1.0/120
	p = p*r + 1.0/24
	p = p*r + 1.0/6
	p = p*r + 0.5
	p = p*r + 1
	p = p*r + 1
	return p * math.Float32frombits(uint32(int32(k)+127)<<23)
}

func geluErf1(x float32) float32 {
	u := -x * 0.70710678118654752
	z := float32(math.Abs(float64(u)))
	t := 1 / (1 + 0.5*z)
	e := -z*z - 1.26551223 + t*(1.00002368+t*(0.37409196+t*(0.09678418+
		t*(-0.18628806+t*(0.27886807+t*(-1.13520398+t*(1.48851587+
			t*(-0.82215223+t*0.17087277))))))))
	r := t * expF32(e)
	if u < 0 {
		r = 2 - r
	}
	return 0.5 * x * r
}

// SoftmaxF32 normalizes x into probabilities as SoftmaxInPlace does, with
// the exponential GELUErf uses: in float32, within 2e-7 of e^x, eight at a
// time. A score of minus infinity comes out zero.
func SoftmaxF32(x []float32) {
	if len(x) == 0 {
		return
	}
	if len(x)%8 == 0 && avx2 {
		softmaxF32AVX2(&x[0], len(x))
		return
	}
	peak := float32(math.Inf(-1))
	for _, v := range x {
		peak = max(peak, v)
	}
	n := len(x) &^ 7
	if n > 0 && avx2 {
		expSubAVX2(&x[0], n, peak)
	} else {
		n = 0
	}
	for i := n; i < len(x); i++ {
		if v := x[i] - peak; v >= erfExpMin {
			x[i] = expF32(v)
		} else {
			x[i] = 0
		}
	}
	var sum float32
	for _, v := range x {
		sum += v
	}
	for i := range x {
		x[i] /= sum
	}
}
