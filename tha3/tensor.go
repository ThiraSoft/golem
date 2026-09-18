package tha3

import "fmt"

// Tensor is one picture or feature map, channel by channel:
// Data[c*H*W + y*W + x], which is how PyTorch lays out a batch of one.
type Tensor struct {
	C, H, W int
	Data    []float32
}

// NewTensor allocates a tensor of zeros.
func NewTensor(c, h, w int) Tensor {
	return Tensor{C: c, H: h, W: w, Data: make([]float32, c*h*w)}
}

// Plane is channel c, sharing t's storage.
func (t Tensor) Plane(c int) []float32 {
	n := t.H * t.W
	return t.Data[c*n : (c+1)*n]
}

// Channels is channels [from, to), sharing t's storage.
func (t Tensor) Channels(from, to int) Tensor {
	n := t.H * t.W
	return Tensor{C: to - from, H: t.H, W: t.W, Data: t.Data[from*n : to*n]}
}

// Clone copies t into storage of its own.
func (t Tensor) Clone() Tensor {
	out := NewTensor(t.C, t.H, t.W)
	copy(out.Data, t.Data)
	return out
}

// Crop copies the h×w window whose top left corner is (y0, x0), in every
// channel. It is PyTorch's image[:, :, y0:y0+h, x0:x0+w].
func (t Tensor) Crop(y0, x0, h, w int) Tensor {
	out := NewTensor(t.C, h, w)
	for c := 0; c < t.C; c++ {
		src, dst := t.Plane(c), out.Plane(c)
		for y := 0; y < h; y++ {
			copy(dst[y*w:(y+1)*w], src[(y0+y)*t.W+x0:])
		}
	}
	return out
}

// Paste writes src into t with its top left corner at (y0, x0).
func (t Tensor) Paste(src Tensor, y0, x0 int) {
	if src.C != t.C {
		panic(fmt.Sprintf("tha3: pasting %d channels into %d", src.C, t.C))
	}
	for c := 0; c < src.C; c++ {
		s, d := src.Plane(c), t.Plane(c)
		for y := 0; y < src.H; y++ {
			row := (y0+y)*t.W + x0
			copy(d[row:row+src.W], s[y*src.W:(y+1)*src.W])
		}
	}
}

// Concat stacks tensors of the same size along the channels, as torch.cat
// does on dimension 1.
func Concat(parts ...Tensor) Tensor {
	c := 0
	for _, p := range parts {
		if p.H != parts[0].H || p.W != parts[0].W {
			panic(fmt.Sprintf("tha3: concatenating %dx%d with %dx%d", p.H, p.W, parts[0].H, parts[0].W))
		}
		c += p.C
	}
	out := NewTensor(c, parts[0].H, parts[0].W)
	offset := 0
	for _, p := range parts {
		offset += copy(out.Data[offset:], p.Data)
	}
	return out
}

// Broadcast is a pose tiled over every pixel, one constant plane per value:
// pose.view(n, c, 1, 1).repeat(1, 1, h, w) upstream.
func Broadcast(values []float32, h, w int) Tensor {
	out := NewTensor(len(values), h, w)
	for c, v := range values {
		plane := out.Plane(c)
		for i := range plane {
			plane[i] = v
		}
	}
	return out
}
