package krea2

// The VAE's decoder: the Wan 2.1 decoder Qwen-Image ships
// (comfy/ldm/wan/vae.py), for a single frame.
//
// One frame makes it a 2D network. A causal 3D convolution pads the time
// axis with zeros in front of the frame, so only the kernel's last temporal
// slice touches it; the decoder is called without a feature cache, so the
// upsamplers' time convolutions never run. What is left is 3×3 and 1×1
// convolutions, RMS norms over the channels, nearest upsampling, and one
// attention in the middle.
//
// Everything runs in three buffers of the size of the largest stage, turned
// round as the network goes: a residual block writes its answer over its
// input, and an upsampler reads its input at half size.

import (
	"encoding/binary"
	"fmt"
	"image"
	"image/color"
	"math"

	"github.com/ThiraSoft/golem/vk"
)

const vaeDim = 96

type vaeConv struct {
	w                    int
	bias                 uint32
	outs, ins, taps, row uint32
}

type vaeRes struct {
	g0, g3    uint32
	c2, c6    vaeConv
	shortcut  *vaeConv
	ins, outs uint32
}

// VAE is the decoder on the card.
type VAE struct {
	k  *vk.K2
	ck *checkpoint

	conv2, conv1 vaeConv
	middle       [2]vaeRes
	attnGamma    uint32
	qkv, proj    vaeConv
	ups          []any // vaeRes or *vaeConv (an upsampler's convolution)
	headGamma    uint32
	head         vaeConv
	maxLatentPix int
	bufs         [3]uint32
	programs     map[[2]int]vaeProgram
}

// OpenVAE uploads the decoder for latents of up to maxLatentPix pixels
// (a 1024 × 1024 picture is 128 × 128).
func OpenVAE(d *vk.Device, path string, maxLatentPix int) (*VAE, error) {
	ck, err := openCheckpoint(path)
	if err != nil {
		return nil, err
	}
	v := &VAE{ck: ck, maxLatentPix: maxLatentPix, programs: map[[2]int]vaeProgram{}}
	fail := func(err error) (*VAE, error) {
		v.Close()
		return nil, err
	}
	var par params
	gamma := func(name string, n int) uint32 {
		if err != nil {
			return 0
		}
		var g []float32
		if g, err = ck.floats(name, n); err != nil {
			return 0
		}
		return par.add(g)
	}
	// conv checks a convolution's shape and gathers its bias; upload sends
	// its weights once the machine exists.
	conv := func(pre string, outs, ins, taps int) vaeConv {
		c := vaeConv{outs: uint32(outs), ins: uint32(ins), taps: uint32(taps), row: uint32((ins*taps + 31) / 32 * 32)}
		if err != nil {
			return c
		}
		t, e := ck.get(pre+".weight", "")
		if e != nil {
			err = e
			return c
		}
		depth := 1
		if len(t.Shape) == 5 {
			depth = t.Shape[2]
		}
		if t.Shape[0] != outs || t.Shape[1] != ins || t.Elems() != outs*ins*depth*taps {
			err = fmt.Errorf("krea2: %s: %s has shape %v, want %d × %d × %d taps", ck.path, pre, t.Shape, outs, ins, taps)
			return c
		}
		c.bias = gamma(pre+".bias", outs)
		return c
	}
	res := func(pre string, ins, outs int) vaeRes {
		r := vaeRes{ins: uint32(ins), outs: uint32(outs)}
		r.g0 = gamma(pre+".residual.0.gamma", ins)
		r.c2 = conv(pre+".residual.2", outs, ins, 9)
		r.g3 = gamma(pre+".residual.3.gamma", outs)
		r.c6 = conv(pre+".residual.6", outs, outs, 9)
		if ins != outs {
			s := conv(pre+".shortcut", outs, ins, 1)
			r.shortcut = &s
		}
		return r
	}

	v.conv2 = conv("conv2", 16, 16, 1)
	v.conv1 = conv("decoder.conv1", 384, 16, 9)
	v.middle[0] = res("decoder.middle.0", 384, 384)
	v.attnGamma = gamma("decoder.middle.1.norm.gamma", 384)
	v.qkv = conv("decoder.middle.1.to_qkv", 3*384, 384, 1)
	v.proj = conv("decoder.middle.1.proj", 384, 384, 1)
	v.middle[1] = res("decoder.middle.2", 384, 384)
	v.headGamma = gamma("decoder.head.0.gamma", vaeDim)
	v.head = conv("decoder.head.2", 3, vaeDim, 9)
	// The upsampling half: three residual blocks a stage, and between two
	// stages a convolution at twice the size that halves the width.
	upPlan := []struct {
		res       bool
		ins, outs int
	}{
		{true, 384, 384}, {true, 384, 384}, {true, 384, 384}, {false, 384, 192},
		{true, 192, 384}, {true, 384, 384}, {true, 384, 384}, {false, 384, 192},
		{true, 192, 192}, {true, 192, 192}, {true, 192, 192}, {false, 192, 96},
		{true, 96, 96}, {true, 96, 96}, {true, 96, 96},
	}
	for i, u := range upPlan {
		pre := fmt.Sprintf("decoder.upsamples.%d", i)
		if u.res {
			v.ups = append(v.ups, res(pre, u.ins, u.outs))
		} else {
			c := conv(pre+".resample.1", u.outs, u.ins, 9)
			v.ups = append(v.ups, &c)
		}
	}
	if err != nil {
		return fail(err)
	}

	var a arena
	stage := 6144 * maxLatentPix // the largest stage: 96 channels at 64 times the latent
	for i := range v.bufs {
		v.bufs[i] = a.take(stage)
	}
	if v.k, err = vk.NewK2(d, a.n, par.data); err != nil {
		return fail(err)
	}
	if err = v.upload(); err != nil {
		return fail(err)
	}
	return v, nil
}

// upload sends every convolution's fp16 rows to the card, reading them
// again from the checkpoint: the handles land in the fields the programs
// read.
func (v *VAE) upload() error {
	up := func(c *vaeConv, name string) error {
		t, err := v.ck.get(name+".weight", "")
		if err != nil {
			return err
		}
		w, err := floats(t)
		if err != nil {
			return err
		}
		depth := 1
		if len(t.Shape) == 5 {
			depth = t.Shape[2]
		}
		outs, ins, taps, row := int(c.outs), int(c.ins), int(c.taps), int(c.row)
		half := make([]byte, 2*outs*row)
		for o := 0; o < outs; o++ {
			for i := 0; i < ins; i++ {
				for tap := 0; tap < taps; tap++ {
					src := ((o*ins+i)*depth+depth-1)*taps + tap
					binary.LittleEndian.PutUint16(half[2*(o*row+i*taps+tap):], floatToHalf(w[src]))
				}
			}
		}
		c.w, err = v.k.AddWeights(half)
		return err
	}
	res := func(r *vaeRes, pre string) error {
		if err := up(&r.c2, pre+".residual.2"); err != nil {
			return err
		}
		if err := up(&r.c6, pre+".residual.6"); err != nil {
			return err
		}
		if r.shortcut != nil {
			return up(r.shortcut, pre+".shortcut")
		}
		return nil
	}
	for _, c := range []struct {
		c    *vaeConv
		name string
	}{{&v.conv2, "conv2"}, {&v.conv1, "decoder.conv1"}, {&v.qkv, "decoder.middle.1.to_qkv"}, {&v.proj, "decoder.middle.1.proj"}, {&v.head, "decoder.head.2"}} {
		if err := up(c.c, c.name); err != nil {
			return err
		}
	}
	if err := res(&v.middle[0], "decoder.middle.0"); err != nil {
		return err
	}
	if err := res(&v.middle[1], "decoder.middle.2"); err != nil {
		return err
	}
	for i, u := range v.ups {
		pre := fmt.Sprintf("decoder.upsamples.%d", i)
		switch u := u.(type) {
		case vaeRes:
			if err := res(&u, pre); err != nil {
				return err
			}
			v.ups[i] = u
		case *vaeConv:
			if err := up(u, pre+".resample.1"); err != nil {
				return err
			}
		}
	}
	return nil
}

func (v *VAE) conv(r *vk.Recorder, c vaeConv, h, w, x, y, residual uint32, upsample bool) {
	up := uint32(0)
	if upsample {
		up = 1
	}
	v.k.Conv(r, c.w, vk.K2Conv{Outputs: c.outs, Inputs: c.ins, H: h, W: w, Taps: c.taps, KStride: c.row,
		X: x, Y: y, Bias: c.bias, Residual: residual, Up: up})
}

func (v *VAE) norm(r *vk.Recorder, src, dst, c, h, w, gamma uint32, silu bool) {
	s := uint32(0)
	if silu {
		s = 1
	}
	v.k.Pix(r, vk.K2Pix{Src: src, Dst: dst, C: c, H: h, W: w, Gamma: gamma, SiLU: s, Op: vk.K2ChanNorm})
}

// residual records a residual block whose input is in b[0]; its answer
// lands in b[0] again, or in b[1] when the block changes the width, and the
// returned order says which buffer holds it.
func (v *VAE) residual(r *vk.Recorder, b [3]uint32, rb vaeRes, h, w uint32) [3]uint32 {
	x, t, u := b[0], b[1], b[2]
	v.norm(r, x, t, rb.ins, h, w, rb.g0, true)
	v.conv(r, rb.c2, h, w, t, u, vk.K2None, false)
	v.norm(r, u, u, rb.outs, h, w, rb.g3, true)
	if rb.shortcut == nil {
		v.conv(r, rb.c6, h, w, u, x, x, false)
		return b
	}
	v.conv(r, *rb.shortcut, h, w, x, t, vk.K2None, false)
	v.conv(r, rb.c6, h, w, u, t, t, false)
	return [3]uint32{t, x, u}
}

// vaeProgram is a recorded decode and the buffer its picture ends in.
type vaeProgram struct {
	p   *vk.Program
	out uint32
}

func (v *VAE) program(h, w int) (vaeProgram, error) {
	key := [2]int{h, w}
	if p, ok := v.programs[key]; ok {
		return p, nil
	}
	var out uint32
	H, W := uint32(h), uint32(w)
	P := H * W
	p, err := v.k.Compile(func(r *vk.Recorder) {
		b := v.bufs
		// The latent is in b[0].
		v.conv(r, v.conv2, H, W, b[0], b[1], vk.K2None, false)
		v.conv(r, v.conv1, H, W, b[1], b[0], vk.K2None, false)
		b = v.residual(r, b, v.middle[0], H, W)
		// The attention: one head of 384 over every pixel.
		x, t, u := b[0], b[1], b[2]
		v.norm(r, x, t, 384, H, W, v.attnGamma, false)
		v.conv(r, v.qkv, H, W, t, u, vk.K2None, false)
		q, k, val, o := t, t+384*P, t+2*384*P, t+3*384*P
		for i, dst := range []uint32{q, k, val} {
			v.k.Pix(r, vk.K2Pix{Src: u + uint32(i)*384*P, Dst: dst, C: 384, H: H, W: W, Op: vk.K2ToTokens})
		}
		v.k.Attn(r, vk.K2Attn{Q: q, QStride: 384, K: k, KStride: 384, V: val, VStride: 384, O: o, OStride: 384,
			Queries: P, Keys: P, Group: 1, Scale: float32(1 / math.Sqrt(384)), Heads: 1, Seqs: 1, HeadDim: 384})
		v.k.Pix(r, vk.K2Pix{Src: o, Dst: u, C: 384, H: H, W: W, Op: vk.K2ToPlanes, Residual: vk.K2None})
		v.conv(r, v.proj, H, W, u, x, x, false)
		b = v.residual(r, b, v.middle[1], H, W)
		for _, up := range v.ups {
			switch up := up.(type) {
			case vaeRes:
				b = v.residual(r, b, up, H, W)
			case *vaeConv:
				H, W = 2*H, 2*W
				v.conv(r, *up, H, W, b[0], b[1], vk.K2None, true)
				b = [3]uint32{b[1], b[0], b[2]}
			}
		}
		v.norm(r, b[0], b[1], vaeDim, H, W, v.headGamma, true)
		v.conv(r, v.head, H, W, b[1], b[2], vk.K2None, false)
		out = b[2]
	})
	if err != nil {
		return vaeProgram{}, err
	}
	v.programs[key] = vaeProgram{p, out}
	return v.programs[key], nil
}

// Decode turns a latent of 16 × h × w (process_latent_out already applied)
// into the picture ComfyUI's VAEDecode gives: 8h × 8w, RGB in [0, 1] as
// float32, row by row, three floats a pixel.
func (v *VAE) Decode(latent []float32, h, w int) ([]float32, error) {
	if len(latent) != 16*h*w {
		return nil, fmt.Errorf("krea2: a latent of %d floats for 16 × %d × %d", len(latent), h, w)
	}
	if h*w > v.maxLatentPix {
		return nil, fmt.Errorf("krea2: a latent of %d × %d is past the %d pixels the VAE was opened for", h, w, v.maxLatentPix)
	}
	p, err := v.program(h, w)
	if err != nil {
		return nil, err
	}
	if err := v.k.Write(int(v.bufs[0]), latent); err != nil {
		return nil, err
	}
	if err := p.p.Run(); err != nil {
		return nil, err
	}
	px := 64 * h * w
	planes, err := v.k.Read(int(p.out), 3*px)
	if err != nil {
		return nil, err
	}
	out := make([]float32, 3*px)
	for i := 0; i < px; i++ {
		for c := 0; c < 3; c++ {
			x := (planes[c*px+i] + 1) / 2
			out[3*i+c] = min(max(x, 0), 1)
		}
	}
	return out, nil
}

// Image is what ComfyUI saves of a decoded picture: each value times 255,
// truncated to a byte.
func Image(rgb []float32, width, height int) *image.NRGBA {
	img := image.NewNRGBA(image.Rect(0, 0, width, height))
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			at := 3 * (y*width + x)
			img.SetNRGBA(x, y, color.NRGBA{byte(255 * rgb[at]), byte(255 * rgb[at+1]), byte(255 * rgb[at+2]), 255})
		}
	}
	return img
}

// Trim gives the decoder's working memory back to the card.
func (v *VAE) Trim() {
	for _, p := range v.programs {
		p.p.Close()
	}
	v.programs = map[[2]int]vaeProgram{}
	v.k.FreeArena()
}

// WeightBytes is what the decoder holds on the card.
func (v *VAE) WeightBytes() int { return v.k.WeightBytes() }

// Close frees the card and the checkpoint.
func (v *VAE) Close() {
	for _, p := range v.programs {
		p.p.Close()
	}
	v.programs = nil
	if v.k != nil {
		v.k.Close()
		v.k = nil
	}
	if v.ck != nil {
		v.ck.close()
		v.ck = nil
	}
}
