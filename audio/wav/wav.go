package wav

// Writing a WAV file: a forty-four byte RIFF header, then the samples. There is
// nothing subtle about the format, it just needs to be exact.

import (
	"encoding/binary"
	"io"
)

// Write serializes floating-point samples as monophonic 16-bit PCM.
func Write(w io.Writer, samples []float32, sampleRate int) error {
	nbytes := len(samples) * 2

	header := make([]byte, 0, 44)
	header = append(header, "RIFF"...)
	header = binary.LittleEndian.AppendUint32(header, uint32(36+nbytes))
	header = append(header, "WAVEfmt "...)
	header = binary.LittleEndian.AppendUint32(header, 16) // size of the format chunk
	header = binary.LittleEndian.AppendUint16(header, 1)  // integer PCM
	header = binary.LittleEndian.AppendUint16(header, 1)  // one channel
	header = binary.LittleEndian.AppendUint32(header, uint32(sampleRate))
	header = binary.LittleEndian.AppendUint32(header, uint32(sampleRate*2)) // bytes per second
	header = binary.LittleEndian.AppendUint16(header, 2)                    // bytes per frame
	header = binary.LittleEndian.AppendUint16(header, 16)                   // bits per sample
	header = append(header, "data"...)
	header = binary.LittleEndian.AppendUint32(header, uint32(nbytes))
	if _, err := w.Write(header); err != nil {
		return err
	}

	data := make([]byte, 0, nbytes)
	for _, v := range samples {
		// The clipping is there on principle: the decoder stays well within
		// bounds, but a silent overflow would be audible as a click.
		if v > 1 {
			v = 1
		} else if v < -1 {
			v = -1
		}
		data = binary.LittleEndian.AppendUint16(data, uint16(int16(v*32767)))
	}
	_, err := w.Write(data)
	return err
}
