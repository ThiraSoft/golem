package tha3

import (
	"fmt"
	"image"
	"image/draw"
	"image/png"
	"math"
	"os"
)

// Size is the side of the picture the networks take, and of the frame they
// return.
const Size = 512

// LoadImage reads a 512×512 PNG with an alpha channel into the model's range.
func LoadImage(path string) (Tensor, error) {
	f, err := os.Open(path)
	if err != nil {
		return Tensor{}, err
	}
	defer f.Close()
	img, err := png.Decode(f)
	if err != nil {
		return Tensor{}, fmt.Errorf("tha3: %s: %w", path, err)
	}
	b := img.Bounds()
	if b.Dx() != Size || b.Dy() != Size {
		return Tensor{}, fmt.Errorf("tha3: %s is %dx%d, want %dx%d", path, b.Dx(), b.Dy(), Size, Size)
	}
	nrgba, ok := img.(*image.NRGBA)
	if !ok {
		nrgba = image.NewNRGBA(b)
		draw.Draw(nrgba, b, img, b.Min, draw.Src)
	}
	return FromNRGBA(nrgba), nil
}

// FromNRGBA prepares a picture as the demo's
// extract_numpy_image_from_PIL_image_with_pytorch_layout does: a pixel with
// alpha 0 becomes transparent black, so its colour cannot leak into the
// networks; colour goes from sRGB to linear; every channel is scaled from
// [0, 1] to [-1, 1]. The arithmetic is float64 and rounded once, as numpy's.
func FromNRGBA(img *image.NRGBA) Tensor {
	b := img.Bounds()
	w, h := b.Dx(), b.Dy()
	out := NewTensor(4, h, w)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			i := img.PixOffset(b.Min.X+x, b.Min.Y+y)
			p := y*w + x
			if img.Pix[i+3] == 0 {
				for c := 0; c < 4; c++ {
					out.Plane(c)[p] = -1
				}
				continue
			}
			for c := 0; c < 3; c++ {
				out.Plane(c)[p] = float32(srgbToLinear(float64(img.Pix[i+c])/255)*2 - 1)
			}
			out.Plane(3)[p] = float32(float64(img.Pix[i+3])/255*2 - 1)
		}
	}
	return out
}

// ToNRGBA is the way back: [-1, 1] to [0, 1], colour from linear to sRGB,
// everything clipped, as the demo's rgba_to_numpy_image.
func ToNRGBA(t Tensor) *image.NRGBA {
	img := image.NewNRGBA(image.Rect(0, 0, t.W, t.H))
	for y := 0; y < t.H; y++ {
		for x := 0; x < t.W; x++ {
			p := y*t.W + x
			i := img.PixOffset(x, y)
			for c := 0; c < 3; c++ {
				img.Pix[i+c] = toByte(linearToSRGB(unit(t.Plane(c)[p])))
			}
			img.Pix[i+3] = toByte(unit(t.Plane(3)[p]))
		}
	}
	return img
}

func unit(v float32) float64 { return math.Min(math.Max((float64(v)+1)/2, 0), 1) }

func toByte(v float64) uint8 { return uint8(math.Round(v * 255)) }

func srgbToLinear(x float64) float64 {
	if x <= 0.04045 {
		return x / 12.92
	}
	return math.Pow((x+0.055)/1.055, 2.4)
}

func linearToSRGB(x float64) float64 {
	if x <= 0.003130804953560372 {
		return x * 12.92
	}
	return 1.055*math.Pow(x, 1/2.4) - 0.055
}
