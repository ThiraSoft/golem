package yue2

// The output stage, yue2.cpp's src/audio-io.h: the percentile normalization
// and the three WAV writers, each byte as yue2.cpp writes it. golem's
// audio/wav writes 16-bit mono, so the stereo writers are here.

import (
	"encoding/binary"
	"fmt"
	"math"
)

// SampleRate is the decoder's rate.
const SampleRate = 48000

// Normalize is audio_normalize: the 1 - peakClip/1e6 percentile of the
// absolute signal, both channels together, is scaled to one and what passes
// it is clipped. peakClip is clamped to [0, 999]; a reference below 1e-6
// leaves the audio as it is.
func Normalize(audio []float32, peakClip int) {
	n := len(audio)
	if n == 0 {
		return
	}
	peakClip = min(max(peakClip, 0), 999)
	abs := make([]float32, n)
	for i, v := range audio {
		if v < 0 {
			v = -v
		}
		abs[i] = v
	}
	pct := 1 - float64(peakClip)/1000000
	ref := selectNth(abs, int(float64(n-1)*pct))
	if ref < 1e-6 {
		return
	}
	gain := 1 / ref
	for i, v := range audio {
		v *= gain
		if v > 1 {
			v = 1
		} else if v < -1 {
			v = -1
		}
		audio[i] = v
	}
}

// selectNth is the value std::nth_element puts at k: the k-th smallest of
// v, which it reorders.
func selectNth(v []float32, k int) float32 {
	lo, hi := 0, len(v)-1
	for lo < hi {
		// Median of three, then Hoare's partition.
		mid := lo + (hi-lo)/2
		if v[mid] < v[lo] {
			v[mid], v[lo] = v[lo], v[mid]
		}
		if v[hi] < v[lo] {
			v[hi], v[lo] = v[lo], v[hi]
		}
		if v[hi] < v[mid] {
			v[hi], v[mid] = v[mid], v[hi]
		}
		pivot := v[mid]
		i, j := lo, hi
		for i <= j {
			for v[i] < pivot {
				i++
			}
			for pivot < v[j] {
				j--
			}
			if i <= j {
				v[i], v[j] = v[j], v[i]
				i++
				j--
			}
		}
		switch {
		case k <= j:
			hi = j
		case k >= i:
			lo = i
		default:
			return v[k]
		}
	}
	return v[k]
}

// wavHeader is wav_write_header_basic: the classic 44-byte RIFF header of
// stereo samples of bits bits, tag 1 for integers and 3 for floats.
func wavHeader(samples, bits int, tag uint16) []byte {
	const channels = 2
	width := bits / 8
	data := samples * channels * width
	h := make([]byte, 0, 44+data)
	h = append(h, "RIFF"...)
	h = binary.LittleEndian.AppendUint32(h, uint32(36+data))
	h = append(h, "WAVEfmt "...)
	h = binary.LittleEndian.AppendUint32(h, 16)
	h = binary.LittleEndian.AppendUint16(h, tag)
	h = binary.LittleEndian.AppendUint16(h, channels)
	h = binary.LittleEndian.AppendUint32(h, SampleRate)
	h = binary.LittleEndian.AppendUint32(h, uint32(SampleRate*channels*width))
	h = binary.LittleEndian.AppendUint16(h, uint16(channels*width))
	h = binary.LittleEndian.AppendUint16(h, uint16(bits))
	h = append(h, "data"...)
	return binary.LittleEndian.AppendUint32(h, uint32(data))
}

// sanitize is wav_sanitize and wav_clamp1 together, the integer formats'
// guard: NaN and infinities to zero, then [-1, 1].
func sanitize(x float32) float32 {
	if math.IsNaN(float64(x)) || math.IsInf(float64(x), 0) {
		return 0
	}
	return min(max(x, -1), 1)
}

// EncodeWAV writes planar stereo, [2][samples], as format: wav16 and wav24
// truncate toward zero after scaling by 32767 and 8388607 as yue2.cpp's
// casts do, wav32 keeps the floats, NaN and infinities zeroed. It does not
// normalize.
func EncodeWAV(audio []float32, samples int, format string) ([]byte, error) {
	if len(audio) < 2*samples {
		return nil, fmt.Errorf("yue2: %d floats for %d stereo samples", len(audio), samples)
	}
	left, right := audio[:samples], audio[samples:2*samples]
	switch format {
	case FormatWAV16:
		b := wavHeader(samples, 16, 1)
		for t := range samples {
			b = binary.LittleEndian.AppendUint16(b, uint16(int16(sanitize(left[t])*32767)))
			b = binary.LittleEndian.AppendUint16(b, uint16(int16(sanitize(right[t])*32767)))
		}
		return b, nil
	case FormatWAV24:
		b := wavHeader(samples, 24, 1)
		for t := range samples {
			for _, x := range [2]float32{left[t], right[t]} {
				u := uint32(int32(sanitize(x) * 8388607))
				b = append(b, byte(u), byte(u>>8), byte(u>>16))
			}
		}
		return b, nil
	case FormatWAV32:
		b := wavHeader(samples, 32, 3)
		for t := range samples {
			for _, x := range [2]float32{left[t], right[t]} {
				if math.IsNaN(float64(x)) || math.IsInf(float64(x), 0) {
					x = 0
				}
				b = binary.LittleEndian.AppendUint32(b, math.Float32bits(x))
			}
		}
		return b, nil
	case FormatMP3:
		return nil, fmt.Errorf("yue2: golem writes no MP3; ask for wav16, wav24 or wav32")
	}
	return nil, fmt.Errorf("yue2: unknown output format %q", format)
}

// Encode is audio_write's encoding of the song: normalized at peakClip
// except in wav32, which keeps the decoder's range, then written as format.
// The song's audio is left as it is.
func (s *Song) Encode(format string, peakClip int) ([]byte, error) {
	audio := s.Audio[:2*s.Samples]
	if format != FormatWAV32 {
		audio = append([]float32(nil), audio...)
		Normalize(audio, peakClip)
	}
	return EncodeWAV(audio, s.Samples, format)
}
