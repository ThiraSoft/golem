// Package mel turns a waveform into the log-mel spectrogram a speech model
// expects.
//
// Nothing here is a fresh design. The filterbank, the window and the paddings
// are transcriptions of what llama.cpp's mtmd-audio.cpp computes for one
// projector or another, because a front end that is a tenth of a decibel off
// hands the tower an input it was never trained on and there is no way to see
// that from the output.
package mel

import "math"

// HzToMel is HTK's mel scale. Slaney's — librosa's default, linear below a
// kilohertz — is a different curve, and the two disagree by enough to matter;
// which one a model wants is a property of how it was trained.
func HzToMel(hz float64) float64 { return 2595 * math.Log10(1+hz/700) }

// MelToHz inverts HzToMel.
func MelToHz(m float64) float64 { return 700 * (math.Pow(10, m/2595) - 1) }

// Filterbank builds bins triangular filters over the fftSize/2+1 bins of a
// spectrum, laid out one filter per row. htk chooses the mel scale; the
// triangles are not area-normalized, so each peaks at one.
func Filterbank(bins, fftSize, rate int, htk bool) []float32 {
	if !htk {
		panic("mel: only the HTK scale is implemented")
	}
	nBins := fftSize/2 + 1
	fmin, fmax := 0.0, float64(rate)/2
	lo, hi := HzToMel(fmin), HzToMel(fmax)

	// bins+2 edges: each filter rises from one and falls to the next but one.
	hz := make([]float64, bins+2)
	for i := range hz {
		hz[i] = MelToHz(lo + (hi-lo)*float64(i)/float64(bins+1))
	}

	step := float64(rate) / float64(fftSize)
	out := make([]float32, bins*nBins)
	for m := 0; m < bins; m++ {
		left, centre, right := hz[m], hz[m+1], hz[m+2]
		dl := math.Max(1e-30, centre-left)
		dr := math.Max(1e-30, right-centre)
		for k := 0; k < nBins; k++ {
			f := float64(k) * step
			var w float64
			switch {
			case f >= left && f <= centre:
				w = (f - left) / dl
			case f > centre && f <= right:
				w = (right - f) / dr
			}
			out[m*nBins+k] = float32(w)
		}
	}
	return out
}

// HannPeriodic is the window the reference builds: the periodic form, divided
// by the length rather than by the length minus one, and zero-padded to the
// size of the transform when the two differ.
func HannPeriodic(length, padTo int) []float32 {
	w := make([]float32, padTo)
	for i := 0; i < length && i < padTo; i++ {
		w[i] = float32(0.5 - 0.5*math.Cos(2*math.Pi*float64(i)/float64(length)))
	}
	return w
}
