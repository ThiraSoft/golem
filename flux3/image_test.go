package flux3

import (
	"bytes"
	"image"
	"image/color"
	"image/jpeg"
	"testing"
)

// makeJPEGWithOrientation creates a minimal valid JPEG with an Exif APP1 segment
// carrying the specified orientation tag.
func makeJPEGWithOrientation(t *testing.T, w, h int, orientation uint16) []byte {
	t.Helper()
	img := image.NewRGBA(image.Rect(0, 0, w, h))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			img.Set(x, y, color.RGBA{R: uint8(x * 255 / w), G: uint8(y * 255 / h), B: 128, A: 255})
		}
	}
	var rawBuf bytes.Buffer
	if err := jpeg.Encode(&rawBuf, img, nil); err != nil {
		t.Fatal(err)
	}
	raw := rawBuf.Bytes()

	// Build TIFF payload with Orientation tag (big-endian)
	// Header: MM (0x4D4D), 42 (0x002A), IFD offset = 8 (0x00000008)
	// IFD0: 1 entry (0x0001)
	// Entry: tag=0x0112, type=3 (SHORT), count=1, value=orientation (in upper 16 bits of 4-byte value)
	// Next IFD: 0x00000000
	var tiff bytes.Buffer
	tiff.Write([]byte{0x4D, 0x4D, 0x00, 0x2A, 0x00, 0x00, 0x00, 0x08})
	tiff.Write([]byte{0x00, 0x01})
	tiff.Write([]byte{0x01, 0x12, 0x00, 0x03, 0x00, 0x00, 0x00, 0x01})
	tiff.Write([]byte{byte(orientation >> 8), byte(orientation & 0xFF), 0x00, 0x00})
	tiff.Write([]byte{0x00, 0x00, 0x00, 0x00})

	// Build APP1 segment: 0xFF 0xE1 [2-byte length] "Exif\x00\x00" [tiff]
	app1Payload := append([]byte("Exif\x00\x00"), tiff.Bytes()...)
	app1Len := len(app1Payload) + 2
	var app1 bytes.Buffer
	app1.Write([]byte{0xFF, 0xE1, byte(app1Len >> 8), byte(app1Len & 0xFF)})
	app1.Write(app1Payload)

	// Insert APP1 right after SOI (raw[:2])
	var result bytes.Buffer
	result.Write(raw[:2])
	result.Write(app1.Bytes())
	result.Write(raw[2:])
	return result.Bytes()
}

func TestJPEGWithOrientation6DecodesRotated(t *testing.T) {
	// Create a 30x20 JPEG with orientation 6 (rotate 90 CW)
	data := makeJPEGWithOrientation(t, 30, 20, 6)

	// DecodeImage must apply orientation 6, swapping 30x20 into 20x30
	img, err := DecodeImage(bytes.NewReader(data))
	if err != nil {
		t.Fatalf("DecodeImage failed: %v", err)
	}
	b := img.Bounds()
	if b.Dx() != 20 || b.Dy() != 30 {
		t.Fatalf("decoded dimensions = %dx%d, want 20x30 (rotated)", b.Dx(), b.Dy())
	}
}

func TestJPEGOrientations(t *testing.T) {
	tests := []struct {
		orientation  uint16
		wantW, wantH int
	}{
		{1, 30, 20},
		{2, 30, 20},
		{3, 30, 20},
		{4, 30, 20},
		{5, 20, 30},
		{6, 20, 30},
		{7, 20, 30},
		{8, 20, 30},
	}
	for _, tc := range tests {
		data := makeJPEGWithOrientation(t, 30, 20, tc.orientation)
		img, err := DecodeImage(bytes.NewReader(data))
		if err != nil {
			t.Fatalf("orientation %d: DecodeImage failed: %v", tc.orientation, err)
		}
		b := img.Bounds()
		if b.Dx() != tc.wantW || b.Dy() != tc.wantH {
			t.Errorf("orientation %d: got %dx%d, want %dx%d", tc.orientation, b.Dx(), b.Dy(), tc.wantW, tc.wantH)
		}
	}
}
