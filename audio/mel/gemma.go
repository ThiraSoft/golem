package mel

import "math"

// The front end of the Gemma 4 conformer, transcribed from
// mtmd_audio_preprocessor_gemma4a and log_mel_spectrogram rather than designed
// here. Every constant is one llama.cpp hardcodes in the PROJECTOR_TYPE_GEMMA4A
// arm of clip.cpp, because the file does not carry them.
const (
	Rate    = 16000 // samples a second
	FFTSize = 512   // points of the transform
	Window  = 320   // 20 ms of samples, not the 25 ms most front ends use
	Hop     = 160   // 10 ms between two frames
	Bins    = 128   // mel filters
	Floor   = 0.001 // what a filter's energy is raised to before the logarithm
	Chunk   = 30 * Rate
)

// Frames says how many mel frames Gemma4A will produce for a signal, without
// producing it. A tower sizes its scratch from this.
func Frames(samples int) int {
	total := 0
	for off := 0; off < samples; off += Chunk {
		n := samples - off
		if n > Chunk {
			n = Chunk
		}
		total += chunkFrames(n)
	}
	return total
}

// chunkFrames is PyTorch's count for one chunk: unfold(size=Window+1, step=Hop)
// over the semicausally padded waveform. The +1 is not a rounding choice, it is
// what the reference implementation wrote, and the weights were trained against
// the number it gives.
func chunkFrames(n int) int {
	withLeft := n + Window/2
	f := (withLeft-(Window+1))/Hop + 1
	if f < 0 {
		return 0
	}
	return f
}

// Gemma4A returns one entry per thirty-second chunk. Each is Bins*frames
// values laid out mel-major — value (m, t) at data[m*frames+t] — which is the
// layout the tower's first tensor loads from.
//
// The paddings are the whole difficulty and they are three:
//
//   - a left padding of half a window, which is what makes the front end
//     semicausal: frame t looks at samples from t*Hop - Window/2;
//   - a right padding out to whatever the frame count above needs, so the last
//     frames exist rather than being dropped;
//   - a trim back to that same count, because the spectrogram is computed over
//     the padded signal and comes out one or two frames longer.
//
// Getting any of them wrong shifts the whole spectrogram by a frame, which
// looks like nothing at all until the tower's answer is nonsense.
func Gemma4A(samples []float32) [][]float32 {
	if len(samples) == 0 {
		return nil
	}
	fft := NewFFT(FFTSize)
	window := HannPeriodic(Window, FFTSize)
	filters := Filterbank(Bins, FFTSize, Rate, true)
	nfb := FFTSize/2 + 1

	spectrum := make([]complex64, nfb)
	magnitude := make([]float32, nfb)
	frame := make([]float32, FFTSize)

	var out [][]float32
	for off := 0; off < len(samples); off += Chunk {
		end := off + Chunk
		if end > len(samples) {
			end = len(samples)
		}
		chunk := samples[off:end]

		frames := chunkFrames(len(chunk))
		if frames <= 0 {
			continue
		}
		padLeft := Window / 2
		needed := (frames-1)*Hop + FFTSize
		totalPad := needed - len(chunk)
		if totalPad < padLeft {
			totalPad = padLeft
		}
		padded := make([]float32, totalPad+len(chunk))
		copy(padded[padLeft:], chunk)

		mel := make([]float32, Bins*frames)
		for t := 0; t < frames; t++ {
			at := t * Hop
			for j := range frame {
				if at+j < len(padded) {
					frame[j] = window[j] * padded[at+j]
				} else {
					frame[j] = 0
				}
			}
			fft.Real(frame, spectrum)
			for k, c := range spectrum {
				re, im := real(c), imag(c)
				// The magnitude, not the power: params.use_magnitude is set
				// for this projector and for no other Gemma one.
				magnitude[k] = float32(math.Sqrt(float64(re*re + im*im)))
			}
			for m := 0; m < Bins; m++ {
				row := filters[m*nfb : (m+1)*nfb]
				var sum float64
				for k, w := range row {
					sum += float64(magnitude[k]) * float64(w)
				}
				if sum < Floor {
					sum = Floor
				}
				mel[m*frames+t] = float32(math.Log(sum))
			}
		}
		out = append(out, mel)
	}
	return out
}
