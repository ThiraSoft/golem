package tha3

import (
	"fmt"
	"math"
	"os"
	"path/filepath"
)

// What the face morpher paints over a shutting eye, it decides at 192 px, and
// on the larger picture a closed eye arrives as a smudge however the fields are
// resized: the lash line is a band of grey a few pixels wide. An anime upscaler
// draws it again as a line. The face the morpher made at 192 goes through it,
// four times larger, and the result is laid over the larger picture's face
// where the eyes were repainted, fading out around them.
//
// The upscaler is Real-ESRGAN's realesr-animevideov3 (BSD-3-Clause), converted
// by ref/tha3/upscaler.py into the weights directory. It is optional: a poser
// without it never reads the file.

// upscalerFile is the upscaler's weights, beside the five networks'.
const upscalerFile = "eye_upscaler"

// upscalerConvs is how many 3×3 convolutions the upscaler chains, and
// upscalerFeatures their width. Every one but the last is followed by a PReLU;
// the last makes 3·4·4 channels, which a pixel shuffle lays out four times
// larger.
const (
	upscalerConvs    = 18
	upscalerFeatures = 64
	upscalerFactor   = 4
)

// upscaler is SRVGGNetCompact with realesr-animevideov3's shape. It works in
// sRGB in [0, 1], as it was trained.
type upscaler struct {
	convs  []Conv2d
	slopes [][]float32 // the PReLUs', one per channel
}

func newUpscaler(w *weights) *upscaler {
	u := &upscaler{}
	in := 3
	for i := range upscalerConvs {
		out := upscalerFeatures
		if i == upscalerConvs-1 {
			out = 3 * upscalerFactor * upscalerFactor
		}
		u.convs = append(u.convs, w.conv(fmt.Sprintf("body.%d", 2*i), in, out, 3, 1, 1, 1, true))
		if i < upscalerConvs-1 {
			u.slopes = append(u.slopes, w.get(fmt.Sprintf("body.%d.weight", 2*i+1), out))
		}
		in = out
	}
	return u
}

// features is the chain of convolutions: 48 channels at x's size.
func (u *upscaler) features(x Tensor) Tensor {
	for i, c := range u.convs {
		x = c.Apply(x)
		if i < len(u.slopes) {
			prelu(x, u.slopes[i])
		}
	}
	return x
}

func prelu(x Tensor, slopes []float32) {
	for c := range x.C {
		a := slopes[c]
		pl := x.Plane(c)
		for i, v := range pl {
			if v < 0 {
				pl[i] = a * v
			}
		}
	}
}

// forward is the whole network: x four times larger, still in sRGB and not
// clamped.
func (u *upscaler) forward(x Tensor) Tensor { return shuffle(u.features(x), x) }

// shuffle is the pixel shuffle of the features, plus x enlarged by nearest
// neighbour: output channel c at (y, x) is feature c·16 + (y%4)·4 + x%4 at
// (y/4, x/4).
func shuffle(f, x Tensor) Tensor {
	const r = upscalerFactor
	out := NewTensor(3, x.H*r, x.W*r)
	for c := range 3 {
		dst, src := out.Plane(c), x.Plane(c)
		for y := range out.H {
			for xx := range out.W {
				ch := c*r*r + (y%r)*r + xx%r
				p := (y/r)*x.W + xx/r
				dst[y*out.W+xx] = f.Plane(ch)[p] + src[p]
			}
		}
	}
	return out
}

// The mask the upscaled face is laid through: the morpher's eye alpha,
// blurred, spread past its edge, and blurred again, all at 192. The first blur
// and the gain after it widen it until it covers the whole place the open eye
// had: the morpher's own alpha thins out at the corners, and the picture's
// open eye shows through it there, a ghost the upscaled face must cover. The
// last blur is the fade into the face around.
const (
	eyeMaskBlur = 8 // the first blur: down to 192/8 and back
	eyeMaskGain = 4 // then multiplied and clamped to one
	eyeMaskFade = 4 // the second: down to 192/4 and back
)

// eyeMask is the mask at the face morpher's size.
func eyeMask(eyeAlpha Tensor) Tensor {
	blurred := blurBy(eyeAlpha, eyeMaskBlur)
	for i, v := range blurred.Data {
		blurred.Data[i] = min(v*eyeMaskGain, 1)
	}
	return blurBy(blurred, eyeMaskFade)
}

// blurBy is x shrunk by ratio and brought back, bilinear both ways.
func blurBy(x Tensor, ratio int) Tensor {
	return ResizeBilinear(ResizeBilinear(x, max(x.H/ratio, 1), max(x.W/ratio, 1)), x.H, x.W)
}

// toSRGB is the colour of a picture tensor, linear in [-1, 1], as the upscaler
// reads it: sRGB in [0, 1], three channels.
func toSRGB(x Tensor) Tensor {
	out := NewTensor(3, x.H, x.W)
	for c := range 3 {
		for i, v := range x.Plane(c) {
			out.Plane(c)[i] = float32(linearToSRGB(unit(v)))
		}
	}
	return out
}

// upscaleEyes lays face, the morpher's 192 output, upscaled over hiFace, the
// same face on the larger picture, through the mask eyeAlpha makes, by
// amount. The alpha channel is hiFace's.
func (u *upscaler) upscaleEyes(face, eyeAlpha, hiFace Tensor, amount float32) Tensor {
	big := ResizeBilinear(u.forward(toSRGB(face)), hiFace.H, hiFace.W)
	mask := resize(eyeMask(eyeAlpha), hiFace.H, hiFace.W)
	out := hiFace.Clone()
	m := mask.Plane(0)
	for c := range 3 {
		src, dst := big.Plane(c), out.Plane(c)
		for i, v := range src {
			lin := float32(srgbToLinear(math.Min(math.Max(float64(v), 0), 1)))*2 - 1
			a := m[i] * amount
			dst[i] = dst[i]*(1-a) + lin*a
		}
	}
	return out
}

// SetEyeUpscale lays the eyes the face morpher paints, upscaled, over the
// larger picture, by amount: zero, the default, leaves them as the fields
// make them, one lays the upscaled face whole where the eyes were repainted.
// The first call above zero reads the upscaler's weights from the directory
// Open was given, and fails if they are not there. It does nothing at a scale
// of one, where nothing was enlarged.
//
// On the card it rebuilds the passes, so it is set once, early.
func (p *Poser) SetEyeUpscale(amount float32) error {
	if p.closed {
		return errClosed
	}
	if amount < 0 || amount > 1 {
		return fmt.Errorf("tha3: an eye upscale of %v is not between 0 and 1", amount)
	}
	if amount == p.eyeUpscale {
		return nil
	}
	if amount > 0 && p.upscaler == nil {
		w, err := openWeights(p.dir, upscalerFile)
		if err != nil {
			return fmt.Errorf("tha3: the eye upscaler: %w (see ref/tha3/README.md)", err)
		}
		u := newUpscaler(w)
		if err := w.err(); err != nil {
			w.close()
			return err
		}
		p.files = append(p.files, w)
		p.upscaler = u
	}
	old := p.eyeUpscale
	p.eyeUpscale = amount
	p.haveLast = false
	if p.gpu == nil {
		return nil
	}
	p.gpu.close()
	p.gpu = nil
	if err := p.useVulkan(p.gpuTrace); err != nil {
		p.eyeUpscale = old
		return fmt.Errorf("tha3: the card for an eye upscale of %v: %w", amount, err)
	}
	return nil
}

// HasEyeUpscaler reports whether dir holds the upscaler's weights.
func HasEyeUpscaler(dir string) bool {
	_, err := os.Stat(filepath.Join(dir, upscalerFile+".safetensors"))
	return err == nil
}
