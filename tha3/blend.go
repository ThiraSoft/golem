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
