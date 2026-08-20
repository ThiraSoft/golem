package wav

// Reading a WAV file. The format is a sequence of chunks; only two of them
// matter, and the rest — the metadata some editors leave behind — is skipped by
// its declared length.

import (
	"encoding/binary"
	"fmt"
	"io"
	"math"
)

// Read returns the samples of a WAV, in [-1, 1], and its sampling rate.
//
// Several channels are mixed down to one by averaging: the models here listen
// in mono, and a caller that wanted a particular channel would have had to say
// which.
//
// Integer PCM of 8, 16, 24 and 32 bits is understood, and 32-bit floating
// point. Anything else — compressed formats above all — is refused by name
// rather than decoded wrongly.
func Read(r io.Reader) ([]float32, int, error) {
	var riff [12]byte
	if _, err := io.ReadFull(r, riff[:]); err != nil {
		return nil, 0, fmt.Errorf("reading the RIFF header: %w", err)
	}
	if string(riff[0:4]) != "RIFF" || string(riff[8:12]) != "WAVE" {
		return nil, 0, fmt.Errorf("not a WAV file")
	}

	var (
		format     uint16
		channels   int
		sampleRate int
		bits       int
		haveFormat bool
	)
	for {
		var head [8]byte
		if _, err := io.ReadFull(r, head[:]); err != nil {
			if err == io.EOF || err == io.ErrUnexpectedEOF {
				return nil, 0, fmt.Errorf("no data chunk in the file")
			}
			return nil, 0, err
		}
		id := string(head[0:4])
		size := int(binary.LittleEndian.Uint32(head[4:8]))

		switch id {
		case "fmt ":
			chunk := make([]byte, size)
			if _, err := io.ReadFull(r, chunk); err != nil {
				return nil, 0, fmt.Errorf("reading the format chunk: %w", err)
			}
			if len(chunk) < 16 {
				return nil, 0, fmt.Errorf("format chunk of %d bytes, want at least 16", len(chunk))
			}
			format = binary.LittleEndian.Uint16(chunk[0:2])
			channels = int(binary.LittleEndian.Uint16(chunk[2:4]))
			sampleRate = int(binary.LittleEndian.Uint32(chunk[4:8]))
			bits = int(binary.LittleEndian.Uint16(chunk[14:16]))
			// An extensible header states the real format in a subformat GUID,
			// whose first two bytes carry the tag this code understands.
			if format == 0xFFFE && len(chunk) >= 26 {
				format = binary.LittleEndian.Uint16(chunk[24:26])
			}
			haveFormat = true

		case "data":
			if !haveFormat {
				return nil, 0, fmt.Errorf("data chunk before the format chunk")
			}
			data := make([]byte, size)
			if _, err := io.ReadFull(r, data); err != nil {
				return nil, 0, fmt.Errorf("reading %d bytes of samples: %w", size, err)
			}
			samples, err := decode(data, format, bits, channels)
			if err != nil {
				return nil, 0, err
			}
			return samples, sampleRate, nil

		default:
			// Chunks are padded to an even length, and the padding byte is not
			// counted in the size.
			skip := int64(size + size%2)
			if _, err := io.CopyN(io.Discard, r, skip); err != nil {
				return nil, 0, fmt.Errorf("skipping chunk %q: %w", id, err)
			}
		}
	}
}

// decode turns the raw bytes of the data chunk into mono samples.
func decode(data []byte, format uint16, bits, channels int) ([]float32, error) {
	if channels < 1 {
		return nil, fmt.Errorf("%d channels", channels)
	}
	const (
		pcm       = 1
		ieeeFloat = 3
	)
	if format != pcm && format != ieeeFloat {
		return nil, fmt.Errorf("format %d is not uncompressed PCM: decode it first", format)
	}

	width := bits / 8
	if width == 0 || len(data)%(width*channels) != 0 {
		return nil, fmt.Errorf("%d bytes for %d channels of %d bits", len(data), channels, bits)
	}
	frames := len(data) / (width * channels)
	out := make([]float32, frames)

	for f := 0; f < frames; f++ {
		var sum float32
		for c := 0; c < channels; c++ {
			at := (f*channels + c) * width
			var v float32
			switch {
			case format == ieeeFloat && bits == 32:
				v = math.Float32frombits(binary.LittleEndian.Uint32(data[at:]))
			case bits == 8:
				// Eight-bit PCM is the one unsigned case, centred on 128.
				v = (float32(data[at]) - 128) / 128
			case bits == 16:
				v = float32(int16(binary.LittleEndian.Uint16(data[at:]))) / 32768
			case bits == 24:
				u := uint32(data[at]) | uint32(data[at+1])<<8 | uint32(data[at+2])<<16
				if u&0x800000 != 0 {
					u |= 0xFF000000
				}
				v = float32(int32(u)) / 8388608
			case bits == 32:
				v = float32(int32(binary.LittleEndian.Uint32(data[at:]))) / 2147483648
			default:
				return nil, fmt.Errorf("%d bits per sample", bits)
			}
			sum += v
		}
		out[f] = sum / float32(channels)
	}
	return out, nil
}
