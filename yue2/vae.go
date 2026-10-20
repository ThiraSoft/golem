package yue2

// The Oobleck decoder on the card: 64 latent channels at 25 frames a second
// to 48 kHz stereo, 1920 samples a frame. It runs on the Krea 2 machine the
// way flux3's VAE does: one float32 arena, the weights fp16 one buffer per
// tensor, every convolution a product on the matrix cores over an fp16
// operand (shaders/yue2_im2col.comp, which also applies the snake that comes
// before every convolution), and one program recorded per tile length.
//
// The activations are time-major, [t][c]: the products write a column per
// token, and a column here is a time step, so they land in that layout with
// nothing to transpose, the residual adds included. The recordings are
// ggml's [c][t] and are compared after a transpose.

import (
	"fmt"
	"unsafe"

	"github.com/ThiraSoft/golem/vk"
)

// The decoder's tiling, yue2.cpp's: a tile decodes its core plus halo frames
// either side and keeps the core's samples, which carry the whole receptive
// field of the untiled decoder.
const (
	vaeCore = 512
	vaeHalo = 16
)

// vaeConv is one convolution's weight on the card, Outputs rows of Inputs
// halves, and its bias in the parameters (vk.K2None for none).
type vaeConv struct {
	w               int
	outputs, inputs uint32
	bias            uint32
}

type vaeRes struct {
	dil            uint32
	s1a, s1b       uint32
	s2a, s2b       uint32
	conv7, conv1x1 vaeConv
}

type vaeBlock struct {
	in, out        uint32
	stride, kernel uint32
	sa, sb         uint32
	// The transposed convolution's weight, rows oc*kernel + k of in halves,
	// so that the product writes ggml's col2im operand; its bias is added
	// by the gather.
	convT vaeConv
	res   [3]vaeRes
}

// VAE is the decoder on one device.
type VAE struct {
	k        *vk.K2
	conv1    vaeConv
	blocks   [6]vaeBlock
	fsa, fsb uint32
	conv2    vaeConv
	maxTile  int
	// The arena: the latents, two activations the stages take turns in, and
	// the products' fp16 operand.
	in, a, b, x uint32
	floats      int
	progs       map[int]*vk.Program
}

// vaeLengths is the length of the signal before every block and after the
// last: a transposed convolution of stride s, kernel 2s and padding
// ceil(s/2) gives (t-1)s + 2s - 2ceil(s/2), which is st for an even stride
// and st - 1 for an odd one, 1920t - 64 in the end.
func vaeLengths(t0 int) [7]int {
	l := [7]int{t0}
	for i, s := range [6]int{6, 5, 4, 4, 2, 2} {
		l[i+1] = (l[i]-1)*s + 2*s - 2*((s+1)/2)
	}
	return l
}

// vaeRooms is what a tile of t0 frames needs of each room of the arena, in
// floats: the activations a and b, the operand x. Block i reads its input
// from a for even i and writes to b, and the other way round for odd i; its
// input's room then holds the transposed convolution's product and the
// residual units' inner activation.
func vaeRooms(t0 int, w *VAEWeights) (a, b, x int) {
	l := vaeLengths(t0)
	rooms := [2]int{2048 * t0, 0}
	x = 64 * 7 * t0 / 2
	for i, blk := range w.Blocks {
		src, dst := i%2, 1-i%2
		col := l[i] * blk.OutChannels * blk.Kernel
		next := blk.OutChannels * l[i+1]
		rooms[src] = max(rooms[src], col, next)
		rooms[dst] = max(rooms[dst], next)
		x = max(x, (blk.InChannels*l[i]+1)/2, 7*next/2)
	}
	// The last block wrote to a; the audio, two floats a sample, goes to b.
	rooms[1] = max(rooms[1], 2*l[6])
	x = max(x, 64*7*l[6]/2)
	return rooms[0], rooms[1], x
}

// NewVAE puts the decoder on d with an arena for tiles of up to maxFrames
// frames; past the tiling's core, its tiles are what it takes.
func NewVAE(d *vk.Device, w *VAEWeights, maxFrames int) (*VAE, error) {
	v := &VAE{maxTile: min(max(maxFrames, 1), vaeCore+2*vaeHalo), progs: map[int]*vk.Program{}}
	ra, rb, rx := vaeRooms(v.maxTile, w)
	at := 0
	room := func(n int) uint32 {
		o := at
		at += (n + 63) / 64 * 64
		return uint32(o)
	}
	v.in, v.a, v.b, v.x = room(64*v.maxTile), room(ra), room(rb), room(rx)
	v.floats = at

	var params []float32
	param := func(p []float32) uint32 {
		o := uint32(len(params))
		params = append(params, p...)
		return o
	}
	v.conv1.bias = param(w.Conv1Bias)
	for i := range w.Blocks {
		blk, vb := &w.Blocks[i], &v.blocks[i]
		vb.in, vb.out = uint32(blk.InChannels), uint32(blk.OutChannels)
		vb.stride, vb.kernel = uint32(blk.Stride), uint32(blk.Kernel)
		vb.sa, vb.sb = param(blk.SnakeAlpha), param(blk.SnakeBeta)
		vb.convT.bias = param(blk.ConvTBias)
		for r := range vb.res {
			ru, vr := &blk.ResUnits[r], &vb.res[r]
			vr.dil = uint32(ru.Dilation)
			vr.s1a, vr.s1b = param(ru.Snake1Alpha), param(ru.Snake1Beta)
			vr.s2a, vr.s2b = param(ru.Snake2Alpha), param(ru.Snake2Beta)
			vr.conv7.bias = param(ru.Conv1Bias)
			vr.conv1x1.bias = param(ru.Conv2Bias)
		}
	}
	v.fsa, v.fsb = param(w.FinalSnakeAlpha), param(w.FinalSnakeBeta)
	v.conv2.bias = vk.K2None

	k, err := vk.NewK2(d, v.floats, params)
	if err != nil {
		return nil, err
	}
	v.k = k
	// A convolution's weight [OC][IC][K] is already the product's rows of
	// ic*K + k, the order the operand is written in. The rows are narrowed
	// to fp16 over the cores straight into the staging buffers, while the
	// card takes the chunk before (vk.K2.AddWeightRows): one core narrowing
	// into a slice and then uploading it was 0.93 s of an open.
	upload := func(c *vaeConv, rows, inputs int, at func(row, i int) float32) error {
		c.outputs, c.inputs = uint32(rows), uint32(inputs)
		c.w, err = k.AddWeightRows(rows, 2*inputs, func(dst []byte, first, n int) {
			out := unsafe.Slice((*uint16)(unsafe.Pointer(&dst[0])), len(dst)/2)
			spread(n, func(lo, hi int) {
				for r := lo; r < hi; r++ {
					row := out[r*inputs : (r+1)*inputs]
					for i := range row {
						row[i] = narrow(at(first+r, i))
					}
				}
			})
		})
		return err
	}
	plain := func(c *vaeConv, wt []float32, oc int) error {
		n := len(wt) / oc
		return upload(c, oc, n, func(r, i int) float32 { return wt[r*n+i] })
	}
	fail := func(err error) (*VAE, error) {
		v.Close()
		return nil, fmt.Errorf("yue2: VAE weights: %w", err)
	}
	if err := plain(&v.conv1, w.Conv1, 2048); err != nil {
		return fail(err)
	}
	for i := range w.Blocks {
		blk, vb := &w.Blocks[i], &v.blocks[i]
		ic, okk := blk.InChannels, blk.OutChannels*blk.Kernel
		// ConvT is [ic][oc][k]: row oc*K + k of the product is column ic's
		// oc*K + k.
		if err := upload(&vb.convT, okk, ic, func(r, i int) float32 { return blk.ConvT[i*okk+r] }); err != nil {
			return fail(err)
		}
		for r := range vb.res {
			ru, vr := &blk.ResUnits[r], &vb.res[r]
			if err := plain(&vr.conv7, ru.Conv1, blk.OutChannels); err != nil {
				return fail(err)
			}
			if err := plain(&vr.conv1x1, ru.Conv2, blk.OutChannels); err != nil {
				return fail(err)
			}
		}
	}
	if err := plain(&v.conv2, w.Conv2, 2); err != nil {
		return fail(err)
	}
	return v, nil
}

// ArenaBytes is the size of the decoder's arena on the card.
func (v *VAE) ArenaBytes() int { return 4 * v.floats }

// conv records a convolution of c's weight over src, [t][ic], into dst,
// [t][oc], storing or adding, through the snake (sa, sb) unless sa is
// vk.K2None.
func (v *VAE) conv(r *vk.Recorder, c vaeConv, src, dst, t, taps, dil, sa, sb, mode uint32) {
	ic := c.inputs / taps
	pad := (taps - 1) / 2 * dil
	v.k.Yue2Im2Col(r, vk.K2Yue2Im2Col{Src: src, Dst: 2 * v.x, DstStride: c.inputs, T: t, IC: ic,
		K: taps, Dil: dil, Pad: pad, SnakeA: sa, SnakeB: sb, Halves: 1})
	v.k.MM(r, c.w, false, vk.K2MM{Outputs: c.outputs, Inputs: c.inputs, Cols: t,
		X: 2 * v.x, XStride: c.inputs, Y: dst, YStride: c.outputs, Bias: c.bias, Scale: 1,
		Mode: mode, XHalves: 1})
}

// vaeStep is a piece of the decoder that leaves an activation worth looking
// at: at in the arena, channels by frames, time-major.
type vaeStep struct {
	name     string
	record   func(r *vk.Recorder)
	at       uint32
	channels int
	frames   int
}

// steps is the decoder over t0 frames, from the latents at v.in, [t][64], to
// the audio at v.b, [t][2].
func (v *VAE) steps(t0 int) []vaeStep {
	l := vaeLengths(t0)
	var s []vaeStep
	s = append(s, vaeStep{"conv1", func(r *vk.Recorder) {
		v.conv(r, v.conv1, v.in, v.a, uint32(t0), 7, 1, vk.K2None, vk.K2None, vk.K2Store)
	}, v.a, 2048, t0})
	rooms := [2]uint32{v.a, v.b}
	for i := range v.blocks {
		b := &v.blocks[i]
		src, dst := rooms[i%2], rooms[1-i%2]
		tin, tout := uint32(l[i]), uint32(l[i+1])
		s = append(s, vaeStep{fmt.Sprintf("b%d_ct", i), func(r *vk.Recorder) {
			// The snake, then the product into the input's own room, which
			// the snake has finished with; then the gather.
			v.k.Yue2Im2Col(r, vk.K2Yue2Im2Col{Src: src, Dst: 2 * v.x, DstStride: b.in, T: tin, IC: b.in,
				K: 1, Dil: 1, SnakeA: b.sa, SnakeB: b.sb, Halves: 1})
			v.k.MM(r, b.convT.w, false, vk.K2MM{Outputs: b.convT.outputs, Inputs: b.in, Cols: tin,
				X: 2 * v.x, XStride: b.in, Y: src, YStride: b.convT.outputs, Bias: vk.K2None, Scale: 1,
				Mode: vk.K2Store, XHalves: 1})
			v.k.Yue2Col2Im(r, vk.K2Yue2Col2Im{Col: src, Dst: dst, TIn: tin, TOut: tout, OC: b.out,
				K: b.kernel, Stride: b.stride, Pad: (b.stride + 1) / 2, Bias: b.convT.bias})
		}, dst, int(b.out), int(tout)})
		for j := range b.res {
			ru := &b.res[j]
			name := fmt.Sprintf("b%d_ru%d", i, j)
			if j == 2 {
				name = fmt.Sprintf("block_%d", i)
			}
			s = append(s, vaeStep{name, func(r *vk.Recorder) {
				v.conv(r, ru.conv7, dst, src, tout, 7, ru.dil, ru.s1a, ru.s1b, vk.K2Store)
				v.conv(r, ru.conv1x1, src, dst, tout, 1, 1, ru.s2a, ru.s2b, vk.K2Add)
			}, dst, int(b.out), int(tout)})
		}
	}
	s = append(s, vaeStep{"audio", func(r *vk.Recorder) {
		v.conv(r, v.conv2, v.a, v.b, uint32(l[6]), 7, 1, v.fsa, v.fsb, vk.K2Store)
	}, v.b, 2, l[6]})
	return s
}

// tile decodes the frames latents hold, [t][64], and returns the audio
// time-major, [t][2].
func (v *VAE) tile(latents []float32, frames int) ([]float32, error) {
	if frames > v.maxTile {
		return nil, fmt.Errorf("yue2: a VAE tile of %d frames, past the %d the arena holds", frames, v.maxTile)
	}
	p, ok := v.progs[frames]
	if !ok {
		var err error
		p, err = v.k.Compile(func(r *vk.Recorder) {
			for _, s := range v.steps(frames) {
				s.record(r)
			}
		})
		if err != nil {
			return nil, err
		}
		v.progs[frames] = p
	}
	if err := v.k.Write(int(v.in), latents[:64*frames]); err != nil {
		return nil, err
	}
	if err := p.Run(); err != nil {
		return nil, err
	}
	return v.k.Read(int(v.b), 2*vaeLengths(frames)[6])
}

// Decode turns frames of latents, [t][64], into planar stereo,
// [2][1920·frames − 64], in tiles as yue2.cpp does.
func (v *VAE) Decode(latents []float32, frames int) ([]float32, error) {
	return v.DecodeTiled(latents, frames, vaeCore, vaeHalo)
}

// DecodeTiled is Decode with tiles of core frames plus halo either side. At
// core frames or fewer it decodes the whole in one.
func (v *VAE) DecodeTiled(latents []float32, frames, core, halo int) ([]float32, error) {
	return v.decodeTiled(latents, frames, core, halo, nil)
}

// decodeTiled is DecodeTiled asking cancel, when not nil, before every tile.
func (v *VAE) decodeTiled(latents []float32, frames, core, halo int, cancel func() bool) ([]float32, error) {
	if len(latents) < 64*frames {
		return nil, fmt.Errorf("yue2: %d latents for %d frames", len(latents), frames)
	}
	total := 1920*frames - 64
	out := make([]float32, 2*total)
	if frames <= core {
		core, halo = frames, 0
	}
	for start := 0; start < frames; start += core {
		if cancel != nil && cancel() {
			return nil, fmt.Errorf("%w: the decoder at frame %d of %d", ErrCancelled, start, frames)
		}
		end := min(start+core, frames)
		left, right := max(start-halo, 0), min(end+halo, frames)
		audio, err := v.tile(latents[64*left:], right-left)
		if err != nil {
			return nil, err
		}
		// The left halo is exactly 1920 samples a frame; the 64 missing
		// samples are the tail's.
		crop := (start - left) * 1920
		for i, n := 0, min(end*1920, total)-start*1920; i < n; i++ {
			out[start*1920+i] = audio[2*(crop+i)]
			out[total+start*1920+i] = audio[2*(crop+i)+1]
		}
	}
	return out, nil
}

// waypoints decodes frames latents untiled, one step a program, and returns
// every step's activation channel-major, [c][t], as the recordings hold it.
func (v *VAE) waypoints(latents []float32, frames int) (map[string][]float32, error) {
	if frames > v.maxTile {
		return nil, fmt.Errorf("yue2: %d frames, past the %d the arena holds", frames, v.maxTile)
	}
	if err := v.k.Write(int(v.in), latents[:64*frames]); err != nil {
		return nil, err
	}
	got := map[string][]float32{}
	for _, s := range v.steps(frames) {
		p, err := v.k.Compile(s.record)
		if err != nil {
			return nil, err
		}
		err = p.Run()
		p.Close()
		if err != nil {
			return nil, err
		}
		tm, err := v.k.Read(int(s.at), s.channels*s.frames)
		if err != nil {
			return nil, err
		}
		cm := make([]float32, len(tm))
		for t := range s.frames {
			for c := range s.channels {
				cm[c*s.frames+t] = tm[t*s.channels+c]
			}
		}
		got[s.name] = cm
	}
	return got, nil
}

// Close releases the card's memory.
func (v *VAE) Close() {
	for t, p := range v.progs {
		p.Close()
		delete(v.progs, t)
	}
	if v.k != nil {
		v.k.Close()
		v.k = nil
	}
}
