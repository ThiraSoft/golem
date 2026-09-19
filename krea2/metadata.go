package krea2

// What a picture says about how it was drawn, in the PNG itself: the prompt,
// the settings and the seed, enough to draw it again, and which golem drew
// it. The parameters are written the way Automatic1111 writes them, which is
// what most tools that read a picture's settings look for, a LoRA included:
// <lora:name:strength> after the prompt.

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"hash/crc32"
	"image"
	"image/png"
	"io"
	"path/filepath"
	"strings"

	"github.com/ThiraSoft/golem/internal/version"
)

// Parameters is the request as a picture's "parameters" text.
func (r Request) Parameters(model string) string {
	var b strings.Builder
	b.WriteString(r.Prompt)
	if r.Lora != "" && r.LoraStrength != 0 {
		fmt.Fprintf(&b, " <lora:%s:%g>", strings.TrimSuffix(r.Lora, ".safetensors"), r.LoraStrength)
	}
	if r.Negative != "" {
		fmt.Fprintf(&b, "\nNegative prompt: %s", r.Negative)
	}
	fmt.Fprintf(&b, "\nSteps: %d, Sampler: er_sde, Schedule type: simple, CFG scale: %g, Seed: %d, Size: %dx%d, Model: %s, Version: golem %s",
		r.Steps, r.CFG, r.Seed, r.Width, r.Height, model, version.String())
	return b.String()
}

// WritePNG writes the picture as a PNG carrying the request that drew it.
func (p *Pipeline) WritePNG(w io.Writer, img image.Image, r Request) error {
	return WritePNG(w, img, r, strings.TrimSuffix(filepath.Base(p.o.DiT), ".safetensors"))
}

// WritePNG writes img as a PNG with three text chunks after its header:
// "parameters", "Software" and "golem", the request as JSON.
func WritePNG(w io.Writer, img image.Image, r Request, model string) error {
	var buf bytes.Buffer
	if err := png.Encode(&buf, img); err != nil {
		return err
	}
	raw := buf.Bytes()
	// The signature, then IHDR: four bytes of length, four of type, the
	// data, four of CRC. The text goes right after it.
	const sig = 8
	end := sig + 12 + int(binary.BigEndian.Uint32(raw[sig:]))
	req, err := json.Marshal(r)
	if err != nil {
		return err
	}
	if _, err := w.Write(raw[:end]); err != nil {
		return err
	}
	for _, kv := range [][2]string{
		{"parameters", r.Parameters(model)},
		{"Software", "golem " + version.String()},
		{"golem", string(req)},
	} {
		if _, err := w.Write(itxt(kv[0], kv[1])); err != nil {
			return err
		}
	}
	_, err = w.Write(raw[end:])
	return err
}

// itxt is an uncompressed iTXt chunk, which holds UTF-8 where tEXt holds
// Latin-1 only: a prompt may be in any language.
func itxt(key, text string) []byte {
	data := append([]byte(key), 0, 0, 0, 0, 0) // no compression, no language, no translated keyword
	data = append(data, text...)
	chunk := make([]byte, 8, 12+len(data))
	binary.BigEndian.PutUint32(chunk, uint32(len(data)))
	copy(chunk[4:], "iTXt")
	chunk = append(chunk, data...)
	return binary.BigEndian.AppendUint32(chunk, crc32.ChecksumIEEE(chunk[4:]))
}
