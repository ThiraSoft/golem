package tha3

// applyColorChange is change*alpha + image*(1-alpha), channel by channel, as
// image_processing_util.apply_color_change computes it. An alpha of one
// channel applies to every channel, the way PyTorch broadcasts it; an alpha
// with as many channels as the image applies channel to channel.
func applyColorChange(alpha, change, image Tensor) Tensor {
	out := NewTensor(image.C, image.H, image.W)
	for c := 0; c < image.C; c++ {
		a := alpha.Plane(0)
		if alpha.C > 1 {
			a = alpha.Plane(c)
		}
		ch, im, o := change.Plane(c), image.Plane(c), out.Plane(c)
		for i := range o {
			o[i] = ch[i]*a[i] + im[i]*(1-a[i])
		}
	}
	return out
}

// lumaWeights are Rec. 709's, for the linear light the tensors hold.
var lumaWeights = [3]float32{0.2126, 0.7152, 0.0722}

// sharpenGain bounds the brightening and the darkening a pixel can take, so
// that a pixel whose brightness is near zero, where the gain is a ratio of
// two small numbers, cannot fly off.
const sharpenGain = 3

// applySharpen is an unsharp mask weighed pixel by pixel by mask, on the
// brightness alone: every pixel is multiplied by how much brighter it is than
// the blur around it, so the three colour channels keep their ratios and only
// the light moves. Nothing can saturate that was not saturated, by
// construction: a colour's chromaticity is those ratios, and multiplying all
// three by one number leaves it exactly where it was. Adding the same step to
// each channel instead, which is the other way to spare the colour, does not
// hold in linear light: a step that is small next to red is large next to the
// green and the blue under it, and lips come out vivid. The image's alpha is
// kept, as applyRGBChange keeps it, so the silhouette cannot grow a halo.
//
// It undoes, where the mask says to, what an enlargement softened: the fields
// the face morpher decides at 192 are blown up to the size of the larger
// picture, and what they paint arrives there as blurred as bilinear leaves it.
func applySharpen(mask, blurred, image Tensor, amount float32) Tensor {
	out := image.Clone()
	m := mask.Plane(0)
	// The tensors hold linear light as v*2-1, and a ratio only means
	// something on the light itself.
	light := func(t Tensor, c, i int) float32 { return (t.Plane(c)[i] + 1) / 2 }
	for i := range m {
		if m[i] == 0 {
			// Not a no-op by arithmetic: the round trip through the
			// light and back moves the last bit.
			continue
		}
		var lit, blur float32
		for c := 0; c < 3; c++ {
			lit += lumaWeights[c] * light(image, c, i)
			blur += lumaWeights[c] * light(blurred, c, i)
		}
		if lit <= 0 {
			continue
		}
		gain := 1 + amount*m[i]*(lit-blur)/lit
		gain = min(max(gain, 1.0/sharpenGain), sharpenGain)
		for c := 0; c < 3; c++ {
			out.Plane(c)[i] = min(light(image, c, i)*gain, 1)*2 - 1
		}
	}
	return out
}

// applyRGBChange blends the three colour channels as applyColorChange does and
// keeps image's alpha channel untouched: apply_rgb_change upstream.
func applyRGBChange(alpha, change, image Tensor) Tensor {
	out := image.Clone()
	a := alpha.Plane(0)
	for c := 0; c < 3; c++ {
		ch, im, o := change.Plane(c), image.Plane(c), out.Plane(c)
		for i := range o {
			o[i] = ch[i]*a[i] + im[i]*(1-a[i])
		}
	}
	return out
}
