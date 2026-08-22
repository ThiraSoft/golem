package mel

import "math"

// FFT is a radix-2 transform of one fixed size, with its twiddle factors and
// its bit-reversal built once. A front end calls Real a few thousand times per
// second of audio, so the tables are worth keeping and the scratch is worth
// owning: an FFT is not safe for two goroutines at once, and a caller that
// wants two makes two.
type FFT struct {
	n       int
	cos     []float64 // cos(-2*pi*k/n) for k < n/2
	sin     []float64
	reverse []int
	re, im  []float64
}

// NewFFT builds a transform of size n, which must be a power of two: this is
// radix-2 and nothing else, because 512 is the only size the models here ask
// for.
func NewFFT(n int) *FFT {
	if n < 2 || n&(n-1) != 0 {
		panic("mel: an FFT of " + itoa(n) + " needs a power of two")
	}
	f := &FFT{
		n:       n,
		cos:     make([]float64, n/2),
		sin:     make([]float64, n/2),
		reverse: make([]int, n),
		re:      make([]float64, n),
		im:      make([]float64, n),
	}
	for k := 0; k < n/2; k++ {
		theta := -2 * math.Pi * float64(k) / float64(n)
		f.cos[k] = math.Cos(theta)
		f.sin[k] = math.Sin(theta)
	}
	bits := 0
	for 1<<bits < n {
		bits++
	}
	for i := range f.reverse {
		r := 0
		for b := 0; b < bits; b++ {
			r |= ((i >> b) & 1) << (bits - 1 - b)
		}
		f.reverse[i] = r
	}
	return f
}

// Real transforms n real samples and fills the n/2+1 bins from zero to
// Nyquist, which is all a spectrum of a real signal holds; the rest is their
// conjugate and no front end wants it.
//
// The transform itself is the complex one over the real input. Packing two
// real signals into one complex transform would halve the work, and at 512
// points that is a few microseconds a frame against a tower that takes
// seconds; the simpler code is worth more than the microseconds.
func (f *FFT) Real(in []float32, out []complex64) {
	if len(in) != f.n {
		panic("mel: an FFT of " + itoa(f.n) + " was given " + itoa(len(in)) + " samples")
	}
	if len(out) != f.n/2+1 {
		panic("mel: an FFT of " + itoa(f.n) + " fills " + itoa(f.n/2+1) + " bins")
	}
	for i, r := range f.reverse {
		f.re[i] = float64(in[r])
		f.im[i] = 0
	}
	for size := 2; size <= f.n; size <<= 1 {
		half := size / 2
		step := f.n / size
		for start := 0; start < f.n; start += size {
			for k := 0; k < half; k++ {
				c, s := f.cos[k*step], f.sin[k*step]
				i, j := start+k, start+k+half
				tr := f.re[j]*c - f.im[j]*s
				ti := f.re[j]*s + f.im[j]*c
				f.re[j] = f.re[i] - tr
				f.im[j] = f.im[i] - ti
				f.re[i] += tr
				f.im[i] += ti
			}
		}
	}
	for k := range out {
		out[k] = complex(float32(f.re[k]), float32(f.im[k]))
	}
}

// itoa keeps the panics above from pulling in a formatter.
func itoa(v int) string {
	if v == 0 {
		return "0"
	}
	neg := v < 0
	if neg {
		v = -v
	}
	var b [20]byte
	i := len(b)
	for v > 0 {
		i--
		b[i] = byte('0' + v%10)
		v /= 10
	}
	if neg {
		i--
		b[i] = '-'
	}
	return string(b[i:])
}
