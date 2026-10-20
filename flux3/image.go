package flux3

import (
	"bytes"
	"encoding/binary"
	"image"
	"image/color"
	_ "image/jpeg"
	_ "image/png"
	"io"

	_ "golang.org/x/image/webp"
)

// ReadJPEGExifOrientation parses JPEG segments looking for an APP1 Exif marker
// and extracts the TIFF Orientation tag (0x0112). Returns 1 if not found or invalid.
func ReadJPEGExifOrientation(data []byte) int {
	if len(data) < 4 || data[0] != 0xFF || data[1] != 0xD8 {
		return 1
	}
	idx := 2
	for idx+4 <= len(data) {
		if data[idx] != 0xFF {
			break
		}
		marker := data[idx+1]
		if marker == 0xDA || marker == 0xD9 {
			break
		}
		if marker == 0xFF {
			idx++
			continue
		}
		length := int(binary.BigEndian.Uint16(data[idx+2 : idx+4]))
		if length < 2 || idx+2+length > len(data) {
			break
		}
		if marker == 0xE1 { // APP1
			payload := data[idx+4 : idx+2+length]
			if len(payload) >= 14 && string(payload[:6]) == "Exif\x00\x00" {
				tiff := payload[6:]
				if len(tiff) >= 8 {
					var bo binary.ByteOrder
					if tiff[0] == 'I' && tiff[1] == 'I' {
						bo = binary.LittleEndian
					} else if tiff[0] == 'M' && tiff[1] == 'M' {
						bo = binary.BigEndian
					}
					if bo != nil && bo.Uint16(tiff[2:4]) == 42 {
						ifdOffset := int(bo.Uint32(tiff[4:8]))
						if ifdOffset >= 0 && ifdOffset+2 <= len(tiff) {
							numEntries := int(bo.Uint16(tiff[ifdOffset : ifdOffset+2]))
							entriesStart := ifdOffset + 2
							for i := 0; i < numEntries && entriesStart+(i+1)*12 <= len(tiff); i++ {
								entry := tiff[entriesStart+i*12 : entriesStart+(i+1)*12]
								tag := bo.Uint16(entry[0:2])
								if tag == 0x0112 {
									val := int(bo.Uint16(entry[8:10]))
									if val >= 1 && val <= 8 {
										return val
									}
								}
							}
						}
					}
				}
			}
		}
		idx += 2 + length
	}
	return 1
}

// ApplyOrientation transposes an image according to the EXIF Orientation tag (1-8),
// matching PIL ImageOps.exif_transpose.
func ApplyOrientation(img image.Image, orientation int) image.Image {
	if orientation <= 1 || orientation > 8 {
		return img
	}
	bounds := img.Bounds()
	w, h := bounds.Dx(), bounds.Dy()
	var dst *image.NRGBA
	switch orientation {
	case 2, 3, 4:
		dst = image.NewNRGBA(image.Rect(0, 0, w, h))
	case 5, 6, 7, 8:
		dst = image.NewNRGBA(image.Rect(0, 0, h, w))
	}
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			c := color.NRGBAModel.Convert(img.At(bounds.Min.X+x, bounds.Min.Y+y)).(color.NRGBA)
			var dx, dy int
			switch orientation {
			case 2: // flip horizontal
				dx, dy = w-1-x, y
			case 3: // rotate 180
				dx, dy = w-1-x, h-1-y
			case 4: // flip vertical
				dx, dy = x, h-1-y
			case 5: // transpose
				dx, dy = y, x
			case 6: // rotate 90 CW (PIL ROTATE_270)
				dx, dy = h-1-y, x
			case 7: // transverse
				dx, dy = h-1-y, w-1-x
			case 8: // rotate 270 CW (PIL ROTATE_90)
				dx, dy = y, w-1-x
			}
			dst.SetNRGBA(dx, dy, c)
		}
	}
	return dst
}

// DecodeImage decodes an image from r and applies any JPEG EXIF orientation tag (1-8).
func DecodeImage(r io.Reader) (image.Image, error) {
	data, err := io.ReadAll(r)
	if err != nil {
		return nil, err
	}
	orientation := ReadJPEGExifOrientation(data)
	img, _, err := image.Decode(bytes.NewReader(data))
	if err != nil {
		return nil, err
	}
	return ApplyOrientation(img, orientation), nil
}
