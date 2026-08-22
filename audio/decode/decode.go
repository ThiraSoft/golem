// Package decode turns an encoded sound file into samples.
//
// Three formats, chosen by what the first bytes say rather than by a file
// name: a WAV that golem already knows how to read, an MP3, and a FLAC. What
// comes back is interleaved and at whatever rate the file was written; making
// it mono and 16 kHz is audio/resample's business.
package decode

import (
	"bufio"
	"bytes"
	"fmt"
	"io"

	"github.com/ThiraSoft/golem/audio/wav"
	"github.com/hajimehoshi/go-mp3"
	"github.com/mewkiz/flac"
)

// Format names what a file's first bytes declare, or "" for anything else.
// MP3 is the awkward one: it has no magic, only an ID3 tag that is optional
// and a frame header whose first eleven bits are set.
func Format(head []byte) string {
	switch {
	case len(head) >= 12 && bytes.Equal(head[0:4], []byte("RIFF")) && bytes.Equal(head[8:12], []byte("WAVE")):
		return "wav"
	case len(head) >= 4 && bytes.Equal(head[0:4], []byte("fLaC")):
		return "flac"
	case len(head) >= 3 && bytes.Equal(head[0:3], []byte("ID3")):
		return "mp3"
	case len(head) >= 2 && head[0] == 0xff && head[1]&0xe0 == 0xe0:
		return "mp3"
	}
	return ""
}

// Decode reads a whole file. The samples are interleaved and in [-1, 1].
func Decode(r io.Reader) ([]float32, int, int, error) {
	br := bufio.NewReader(r)
	head, err := br.Peek(12)
	if err != nil && len(head) == 0 {
		return nil, 0, 0, fmt.Errorf("audio: the file is empty")
	}
	switch Format(head) {
	case "wav":
		return wav.ReadChannels(br)
	case "mp3":
		return decodeMP3(br)
	case "flac":
		return decodeFLAC(br)
	}
	return nil, 0, 0, fmt.Errorf("audio: these bytes are not a WAV, an MP3 or a FLAC")
}

// decodeMP3 reads the decoder's stream, which is always 16-bit little-endian
// stereo whatever the file held: a mono recording comes back as two identical
// channels, and saying so is the caller's business, not this function's.
func decodeMP3(r io.Reader) ([]float32, int, int, error) {
	d, err := mp3.NewDecoder(r)
	if err != nil {
		return nil, 0, 0, fmt.Errorf("audio: reading the MP3: %w", err)
	}
	raw, err := io.ReadAll(d)
	if err != nil {
		return nil, 0, 0, fmt.Errorf("audio: decoding the MP3: %w", err)
	}
	samples := make([]float32, len(raw)/2)
	for i := range samples {
		samples[i] = float32(int16(uint16(raw[2*i])|uint16(raw[2*i+1])<<8)) / 32768
	}
	return samples, d.SampleRate(), 2, nil
}

// decodeFLAC walks the stream's frames. Each subframe is one channel, in the
// file's own bit depth, which is what the division brings back into [-1, 1].
func decodeFLAC(r io.Reader) ([]float32, int, int, error) {
	s, err := flac.Parse(r)
	if err != nil {
		return nil, 0, 0, fmt.Errorf("audio: reading the FLAC: %w", err)
	}
	defer s.Close()
	channels := int(s.Info.NChannels)
	if channels < 1 {
		return nil, 0, 0, fmt.Errorf("audio: the FLAC declares %d channels", channels)
	}
	scale := float32(int32(1) << (s.Info.BitsPerSample - 1))
	out := make([]float32, 0, int(s.Info.NSamples)*channels)
	for {
		frame, err := s.ParseNext()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, 0, 0, fmt.Errorf("audio: decoding the FLAC: %w", err)
		}
		n := len(frame.Subframes[0].Samples)
		for i := 0; i < n; i++ {
			for c := 0; c < channels; c++ {
				out = append(out, float32(frame.Subframes[c].Samples[i])/scale)
			}
		}
	}
	return out, int(s.Info.SampleRate), channels, nil
}
