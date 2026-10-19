package flux3

// The Flux2 VAE: AutoencoderKL with 32 latent channels, batch norm, and 2×2 patchification.
//
// Latent representation and ordering in ComfyUI:
//
// In ComfyUI (/mnt/data/dev/ComfyUI-flux3), the model loads as an AutoencoderKL
// with z_channels=32, ch=128, ch_mult=[1, 2, 4, 4], num_res_blocks=2, and
// batch_norm_latent=True (comfy/sd.py lines 676-708).
//
// - Encode (comfy/ldm/models/autoencoder.py lines 223-235):
//   1. Patch 2×2: rearrange("... c (i pi) (j pj) -> ... (c pi pj) i j", pi=2, pj=2),
//      converting 32 channels at 8× downscale to 128 channels at 16× downscale (lines 224-228).
//   2. Batch norm: F.batch_norm(z, running_mean, running_var, eps=1e-4) (lines 230-234).
//   Exact order: patch THEN batch-norm.
//
// - Decode (comfy/ldm/models/autoencoder.py lines 241-250):
//   1. Un-batch-norm: z = z * sqrt(running_var + 1e-4) + running_mean (lines 242-244).
//   2. Unpatch: rearrange("... (c pi pj) i j -> ... c (i pi) (j pj)", pi=2, pj=2),
//      converting 128 channels at 16× downscale to 32 channels at 8× downscale (lines 245-250).
//   Exact order: un-batch-norm THEN unpatch.
//
// - In comfy/latent_formats.py (lines 241-245), process_in and process_out for
//   Flux2 are identity functions (return latent) because the patching and
//   batch normalization are handled directly within AutoencoderKL.
//
// - Preprocessing and cropping:
//   Input RGB in [0, 1] is mapped to [-1, 1] via x * 2.0 - 1.0 (comfy/sd.py line 514).
//   When input dimensions are not multiples of 16, comfy/sd.py lines 1150-1160
//   crops with offset (dim % 16) / 2 and length (dim // 16) * 16.
//   Output RGB from decode is clamped and mapped back via (x + 1.0) / 2.0 (comfy/sd.py line 515).

import (
	"encoding/binary"
	"fmt"
	"math"

	"github.com/ThiraSoft/golem/tensors"
	"github.com/ThiraSoft/golem/vk"
)

const (
	convNormal     = 0
	convUpsample   = 1
	convDownsample = 2
)

type vaeConv struct {
	name                 string
	w                    int
	bias                 uint32
	outs, ins, taps, row uint32
}

type vaeNorm struct {
	weight, bias uint32
	channels     uint32
}

type vaeRes struct {
	norm1, norm2 vaeNorm
	conv1, conv2 vaeConv
	shortcut     *vaeConv
	ins, outs    uint32
}

type vaeAttn struct {
	norm     vaeNorm
	qkv      vaeConv
	proj     vaeConv
	channels uint32
}

type vaeUpBlock struct {
	blocks   [3]vaeRes
	upsample *vaeConv // nil for level 0
}

type vaeDownBlock struct {
	blocks     [2]vaeRes
	downsample *vaeConv // nil for level 3
}

type vaeDecoder struct {
	postQuantConv vaeConv
	convIn        vaeConv
	mid0          vaeRes
	midAttn       vaeAttn
	mid1          vaeRes
	up            [4]vaeUpBlock // 3, 2, 1, 0
	normOut       vaeNorm
	convOut       vaeConv
}

type vaeEncoder struct {
	convIn    vaeConv
	down      [4]vaeDownBlock // 0, 1, 2, 3
	mid0      vaeRes
	midAttn   vaeAttn
	mid1      vaeRes
	normOut   vaeNorm
	convOut   vaeConv
	quantConv vaeConv
}

type vaeDecodeProgram struct {
	p        *vk.Program
	out      uint32
	up0, up1 uint32
	up2, up3 uint32
}

type vaeEncodeProgram struct {
	p   *vk.Program
	out uint32
}

// VAE is the Flux2 AutoencoderKL (decoder and encoder) on the card.
type VAE struct {
	k  *vk.K2
	tm *tensors.Model

	bnMean uint32
	bnVar  uint32

	dec vaeDecoder
	enc vaeEncoder

	maxLatentPix    int
	recordWaypoints bool
	stageFloats     int
	bufs            [3]uint32
	statsBuf        uint32
	wpBufs          [4]uint32 // up_0, up_1, up_2, up_3 copy buffers (only when recordWaypoints is true)

	decodeProgs map[[2]int]vaeDecodeProgram
	encodeProgs map[[2]int]vaeEncodeProgram
}

// OpenVAE opens the VAE from a safetensors file for latents up to maxLatentPix.
// Waypoint buffers are not allocated on this production path.
func OpenVAE(d *vk.Device, path string, maxLatentPix int) (*VAE, error) {
	return openVAE(d, path, maxLatentPix, false)
}

// openVAEForTest opens the VAE with waypoint buffers allocated for testing.
func openVAEForTest(d *vk.Device, path string, maxLatentPix int) (*VAE, error) {
	return openVAE(d, path, maxLatentPix, true)
}

func openVAE(d *vk.Device, path string, maxLatentPix int, waypoints bool) (*VAE, error) {
	maxAllowedPix := int((1<<32/4 - 1024) / 163840)
	if maxLatentPix <= 0 || maxLatentPix > maxAllowedPix {
		return nil, fmt.Errorf("flux3: maxLatentPix %d past arena limit of %d (max resolution %dx%d)", maxLatentPix, maxAllowedPix, int(math.Sqrt(float64(maxAllowedPix)))*16, int(math.Sqrt(float64(maxAllowedPix)))*16)
	}
	if waypoints {
		maxWaypointsPix := int((1<<32/4 - 1024) / (163840 + 139264))
		if maxLatentPix > maxWaypointsPix {
			return nil, fmt.Errorf("flux3: maxLatentPix %d past arena limit with waypoints of %d", maxLatentPix, maxWaypointsPix)
		}
	}

	tm, err := tensors.Open(path)
	if err != nil {
		return nil, fmt.Errorf("flux3: %s: %w", path, err)
	}

	v := &VAE{
		tm:              tm,
		maxLatentPix:    maxLatentPix,
		recordWaypoints: waypoints,
		decodeProgs:     make(map[[2]int]vaeDecodeProgram),
		encodeProgs:     make(map[[2]int]vaeEncodeProgram),
	}
	fail := func(err error) (*VAE, error) {
		v.Close()
		return nil, err
	}

	var par parBuffer

	// Load batch norm running stats (128 channels each).
	bnMeanT, err := tm.Get("bn.running_mean")
	if err != nil {
		return fail(fmt.Errorf("flux3: bn.running_mean: %w", err))
	}
	bnMeanF, err := bnMeanT.F32()
	if err != nil {
		return fail(err)
	}
	v.bnMean = par.add(bnMeanF)

	bnVarT, err := tm.Get("bn.running_var")
	if err != nil {
		return fail(fmt.Errorf("flux3: bn.running_var: %w", err))
	}
	bnVarF, err := bnVarT.F32()
	if err != nil {
		return fail(err)
	}
	v.bnVar = par.add(bnVarF)

	loadNorm := func(pre string, c int) (vaeNorm, error) {
		wT, err := tm.Get(pre + ".weight")
		if err != nil {
			return vaeNorm{}, fmt.Errorf("flux3: %s.weight: %w", pre, err)
		}
		wF, err := wT.F32()
		if err != nil {
			return vaeNorm{}, err
		}
		bT, err := tm.Get(pre + ".bias")
		if err != nil {
			return vaeNorm{}, fmt.Errorf("flux3: %s.bias: %w", pre, err)
		}
		bF, err := bT.F32()
		if err != nil {
			return vaeNorm{}, err
		}
		return vaeNorm{
			weight:   par.add(wF),
			bias:     par.add(bF),
			channels: uint32(c),
		}, nil
	}

	loadConv := func(pre string, outs, ins, taps int) (vaeConv, error) {
		c := vaeConv{
			name: pre,
			outs: uint32(outs),
			ins:  uint32(ins),
			taps: uint32(taps),
			row:  uint32((ins*taps + 31) / 32 * 32),
		}
		t, err := tm.Get(pre + ".weight")
		if err != nil {
			return c, fmt.Errorf("flux3: %s.weight: %w", pre, err)
		}
		if t.Shape[0] < outs {
			return c, fmt.Errorf("flux3: %s shape %v outs < %d", pre, t.Shape, outs)
		}
		bT, err := tm.Get(pre + ".bias")
		if err != nil {
			c.bias = vk.K2None
		} else {
			bF, err := bT.F32()
			if err != nil {
				return c, err
			}
			if len(bF) < outs {
				return c, fmt.Errorf("flux3: %s bias len %d < %d", pre, len(bF), outs)
			}
			c.bias = par.add(bF[:outs])
		}
		return c, nil
	}

	loadRes := func(pre string, ins, outs int) (vaeRes, error) {
		r := vaeRes{ins: uint32(ins), outs: uint32(outs)}
		var err error
		if r.norm1, err = loadNorm(pre+".norm1", ins); err != nil {
			return r, err
		}
		if r.conv1, err = loadConv(pre+".conv1", outs, ins, 9); err != nil {
			return r, err
		}
		if r.norm2, err = loadNorm(pre+".norm2", outs); err != nil {
			return r, err
		}
		if r.conv2, err = loadConv(pre+".conv2", outs, outs, 9); err != nil {
			return r, err
		}
		if ins != outs {
			s, err := loadConv(pre+".nin_shortcut", outs, ins, 1)
			if err != nil {
				return r, err
			}
			r.shortcut = &s
		}
		return r, nil
	}

	loadAttn := func(pre string, c int) (vaeAttn, error) {
		a := vaeAttn{channels: uint32(c)}
		var err error
		if a.norm, err = loadNorm(pre+".norm", c); err != nil {
			return a, err
		}
		// Fuse q, k, v (each c->c 1x1 conv) into one (3c)->c 1x1 conv.
		qB, err := v.tm.Get(pre + ".q.bias")
		if err != nil {
			return a, err
		}
		qBias, err := qB.F32()
		if err != nil {
			return a, err
		}
		kB, err := v.tm.Get(pre + ".k.bias")
		if err != nil {
			return a, err
		}
		kBias, err := kB.F32()
		if err != nil {
			return a, err
		}
		vB, err := v.tm.Get(pre + ".v.bias")
		if err != nil {
			return a, err
		}
		vBias, err := vB.F32()
		if err != nil {
			return a, err
		}
		qkvBias := make([]float32, 3*c)
		copy(qkvBias[0:c], qBias)
		copy(qkvBias[c:2*c], kBias)
		copy(qkvBias[2*c:3*c], vBias)
		a.qkv = vaeConv{
			name: pre + ".qkv",
			outs: uint32(3 * c),
			ins:  uint32(c),
			taps: 1,
			row:  uint32(c),
			bias: par.add(qkvBias),
		}
		if a.proj, err = loadConv(pre+".proj_out", c, c, 1); err != nil {
			return a, err
		}
		return a, nil
	}

	// 1. Decoder
	if v.dec.postQuantConv, err = loadConv("decoder.post_quant_conv", 32, 32, 1); err != nil {
		return fail(err)
	}
	if v.dec.convIn, err = loadConv("decoder.conv_in", 512, 32, 9); err != nil {
		return fail(err)
	}
	if v.dec.mid0, err = loadRes("decoder.mid.block_1", 512, 512); err != nil {
		return fail(err)
	}
	if v.dec.midAttn, err = loadAttn("decoder.mid.attn_1", 512); err != nil {
		return fail(err)
	}
	if v.dec.mid1, err = loadRes("decoder.mid.block_2", 512, 512); err != nil {
		return fail(err)
	}

	// Up blocks: level 3 (512->512), 2 (512->512), 1 (512->256), 0 (256->128)
	upChannels := map[int][2]int{
		3: {512, 512},
		2: {512, 512},
		1: {512, 256},
		0: {256, 128},
	}
	for lvl := 3; lvl >= 0; lvl-- {
		pre := fmt.Sprintf("decoder.up.%d", lvl)
		inC := upChannels[lvl][0]
		outC := upChannels[lvl][1]
		for b := 0; b < 3; b++ {
			bIn := inC
			if b > 0 {
				bIn = outC
			}
			r, err := loadRes(fmt.Sprintf("%s.block.%d", pre, b), bIn, outC)
			if err != nil {
				return fail(err)
			}
			v.dec.up[lvl].blocks[b] = r
		}
		if lvl != 0 {
			upC, err := loadConv(pre+".upsample.conv", outC, outC, 9)
			if err != nil {
				return fail(err)
			}
			v.dec.up[lvl].upsample = &upC
		}
	}
	if v.dec.normOut, err = loadNorm("decoder.norm_out", 128); err != nil {
		return fail(err)
	}
	if v.dec.convOut, err = loadConv("decoder.conv_out", 3, 128, 9); err != nil {
		return fail(err)
	}

	// 2. Encoder
	if v.enc.convIn, err = loadConv("encoder.conv_in", 128, 3, 9); err != nil {
		return fail(err)
	}
	downChannels := [4][2]int{
		{128, 128}, // level 0
		{128, 256}, // level 1
		{256, 512}, // level 2
		{512, 512}, // level 3
	}
	for lvl := 0; lvl < 4; lvl++ {
		pre := fmt.Sprintf("encoder.down.%d", lvl)
		inC := downChannels[lvl][0]
		outC := downChannels[lvl][1]
		for b := 0; b < 2; b++ {
			bIn := inC
			if b > 0 {
				bIn = outC
			}
			r, err := loadRes(fmt.Sprintf("%s.block.%d", pre, b), bIn, outC)
			if err != nil {
				return fail(err)
			}
			v.enc.down[lvl].blocks[b] = r
		}
		if lvl != 3 {
			downC, err := loadConv(pre+".downsample.conv", outC, outC, 9)
			if err != nil {
				return fail(err)
			}
			v.enc.down[lvl].downsample = &downC
		}
	}
	if v.enc.mid0, err = loadRes("encoder.mid.block_1", 512, 512); err != nil {
		return fail(err)
	}
	if v.enc.midAttn, err = loadAttn("encoder.mid.attn_1", 512); err != nil {
		return fail(err)
	}
	if v.enc.mid1, err = loadRes("encoder.mid.block_2", 512, 512); err != nil {
		return fail(err)
	}
	if v.enc.normOut, err = loadNorm("encoder.norm_out", 512); err != nil {
		return fail(err)
	}
	if v.enc.convOut, err = loadConv("encoder.conv_out", 64, 512, 9); err != nil {
		return fail(err)
	}
	// quant_conv: takes first 32 channels of 64
	if v.enc.quantConv, err = loadConv("encoder.quant_conv", 32, 64, 1); err != nil {
		return fail(err)
	}

	// Memory layout:
	// Largest stage is 256 channels at 16× latent pixels: 256 * 256 * maxLatentPix = 65536 * maxLatentPix floats.
	v.stageFloats = 65536 * maxLatentPix
	var a arena
	v.bufs[0] = a.take(65536 * maxLatentPix)
	v.bufs[1] = a.take(65536 * maxLatentPix)
	v.bufs[2] = a.take(32768 * maxLatentPix)
	if waypoints {
		// Waypoint copy buffers allocated ONLY when test opts in via openVAEForTest.
		v.wpBufs[3] = a.take(512 * 16 * maxLatentPix)  // up_3: 512 channels at 4x latent
		v.wpBufs[2] = a.take(512 * 64 * maxLatentPix)  // up_2: 512 channels at 8x latent
		v.wpBufs[1] = a.take(256 * 256 * maxLatentPix) // up_1: 256 channels at 16x latent
		v.wpBufs[0] = a.take(128 * 256 * maxLatentPix) // up_0: 128 channels at 16x latent
	}
	v.statsBuf = a.take(32 * 64 * 3) // stats buffer for GroupNorm

	if v.k, err = vk.NewK2(d, a.n, par.data); err != nil {
		return fail(err)
	}
	if err = v.upload(); err != nil {
		return fail(err)
	}
	return v, nil
}

// upload sends all fp16 convolution weights to the GPU.
func (v *VAE) upload() error {
	up := func(c *vaeConv) error {
		t, err := v.tm.Get(c.name + ".weight")
		if err != nil {
			return err
		}
		floats, err := t.F32()
		if err != nil {
			return err
		}
		outs, ins, taps, row := int(c.outs), int(c.ins), int(c.taps), int(c.row)
		half := make([]byte, 2*outs*row)
		for o := 0; o < outs; o++ {
			for i := 0; i < ins; i++ {
				for tap := 0; tap < taps; tap++ {
					src := (o*int(t.Shape[1])+i)*taps + tap
					binary.LittleEndian.PutUint16(half[2*(o*row+i*taps+tap):], floatToHalf(floats[src]))
				}
			}
		}
		c.w, err = v.k.AddWeights(half)
		return err
	}

	uploadRes := func(r *vaeRes) error {
		if err := up(&r.conv1); err != nil {
			return err
		}
		if err := up(&r.conv2); err != nil {
			return err
		}
		if r.shortcut != nil {
			if err := up(r.shortcut); err != nil {
				return err
			}
		}
		return nil
	}

	uploadAttn := func(a *vaeAttn, pre string) error {
		qT, err := v.tm.Get(pre + ".q.weight")
		if err != nil {
			return err
		}
		qF, err := qT.F32()
		if err != nil {
			return err
		}
		kT, err := v.tm.Get(pre + ".k.weight")
		if err != nil {
			return err
		}
		kF, err := kT.F32()
		if err != nil {
			return err
		}
		vT, err := v.tm.Get(pre + ".v.weight")
		if err != nil {
			return err
		}
		vF, err := vT.F32()
		if err != nil {
			return err
		}
		c := int(a.channels)
		outs := 3 * c
		row := c
		half := make([]byte, 2*outs*row)
		writeWeights := func(floats []float32, outOffset int) {
			for o := 0; o < c; o++ {
				for i := 0; i < c; i++ {
					w := floats[o*c+i]
					idx := 2 * ((outOffset+o)*row + i)
					binary.LittleEndian.PutUint16(half[idx:idx+2], floatToHalf(w))
				}
			}
		}
		writeWeights(qF, 0)
		writeWeights(kF, c)
		writeWeights(vF, 2*c)
		a.qkv.w, err = v.k.AddWeights(half)
		if err != nil {
			return err
		}
		return up(&a.proj)
	}

	// Upload decoder
	for _, c := range []*vaeConv{&v.dec.postQuantConv, &v.dec.convIn, &v.dec.convOut} {
		if err := up(c); err != nil {
			return err
		}
	}
	if err := uploadRes(&v.dec.mid0); err != nil {
		return err
	}
	if err := uploadAttn(&v.dec.midAttn, "decoder.mid.attn_1"); err != nil {
		return err
	}
	if err := uploadRes(&v.dec.mid1); err != nil {
		return err
	}
	for lvl := 0; lvl < 4; lvl++ {
		for b := 0; b < 3; b++ {
			if err := uploadRes(&v.dec.up[lvl].blocks[b]); err != nil {
				return err
			}
		}
		if v.dec.up[lvl].upsample != nil {
			if err := up(v.dec.up[lvl].upsample); err != nil {
				return err
			}
		}
	}

	// Upload encoder
	for _, c := range []*vaeConv{&v.enc.convIn, &v.enc.convOut, &v.enc.quantConv} {
		if err := up(c); err != nil {
			return err
		}
	}
	for lvl := 0; lvl < 4; lvl++ {
		for b := 0; b < 2; b++ {
			if err := uploadRes(&v.enc.down[lvl].blocks[b]); err != nil {
				return err
			}
		}
		if v.enc.down[lvl].downsample != nil {
			if err := up(v.enc.down[lvl].downsample); err != nil {
				return err
			}
		}
	}
	if err := uploadRes(&v.enc.mid0); err != nil {
		return err
	}
	if err := uploadAttn(&v.enc.midAttn, "encoder.mid.attn_1"); err != nil {
		return err
	}
	if err := uploadRes(&v.enc.mid1); err != nil {
		return err
	}

	return nil
}

func (v *VAE) conv(r *vk.Recorder, c vaeConv, h, w, x, y, residual uint32, mode int) {
	up, down := uint32(0), uint32(0)
	switch mode {
	case convUpsample:
		up = 1
	case convDownsample:
		down = 1
	}
	v.k.Conv(r, c.w, vk.K2Conv{
		Outputs:  c.outs,
		Inputs:   c.ins,
		H:        h,
		W:        w,
		Taps:     c.taps,
		KStride:  c.row,
		X:        x,
		Y:        y,
		Bias:     c.bias,
		Residual: residual,
		Up:       up,
		Down:     down,
	})
}

func (v *VAE) norm(r *vk.Recorder, src, dst, c, h, w uint32, n vaeNorm, silu bool) {
	plane := h * w
	cpg := c / 32
	groupElems := cpg * plane
	slices := uint32(max(1, min(64, int((groupElems+8191)/8192))))
	sliceLen := (groupElems + slices - 1) / slices
	v.k.Flux3GStats(r, vk.K2Flux3GStats{
		Src:        src,
		Plane:      plane,
		CPG:        cpg,
		GroupElems: groupElems,
		SliceLen:   sliceLen,
		Slices:     slices,
		Stats:      v.statsBuf,
	})
	s := uint32(0)
	if silu {
		s = 1
	}
	v.k.Flux3GNorm(r, vk.K2Flux3GNorm{
		Src:        src,
		Dst:        dst,
		Plane:      plane,
		CPG:        cpg,
		GroupElems: groupElems,
		Slices:     slices,
		Stats:      v.statsBuf,
		Weight:     n.weight,
		Bias:       n.bias,
		SiLU:       s,
	})
}

func (v *VAE) residual(r *vk.Recorder, b [3]uint32, rb vaeRes, h, w uint32) [3]uint32 {
	x, t, u := b[0], b[1], b[2]
	v.norm(r, x, t, rb.ins, h, w, rb.norm1, true)
	v.conv(r, rb.conv1, h, w, t, u, vk.K2None, convNormal)
	v.norm(r, u, u, rb.outs, h, w, rb.norm2, true)
	if rb.shortcut == nil {
		v.conv(r, rb.conv2, h, w, u, x, x, convNormal)
		return b
	}
	v.conv(r, *rb.shortcut, h, w, x, t, vk.K2None, convNormal)
	v.conv(r, rb.conv2, h, w, u, t, t, convNormal)
	return [3]uint32{t, x, u}
}

func (v *VAE) attention(r *vk.Recorder, b [3]uint32, a vaeAttn, h, w uint32) [3]uint32 {
	x, t, u := b[0], b[1], b[2]
	P := h * w
	v.norm(r, x, t, a.channels, h, w, a.norm, false)
	// Single fused 1x1 convolution computing Q, K, V at once.
	v.conv(r, a.qkv, h, w, t, u, vk.K2None, convNormal)

	qToken := t
	kToken := t + 512*P
	vToken := t + 2*512*P
	oToken := t + 3*512*P
	for i, dst := range []uint32{qToken, kToken, vToken} {
		v.k.Pix(r, vk.K2Pix{Src: u + uint32(i)*512*P, Dst: dst, C: 512, H: h, W: w, Op: vk.K2ToTokens, Residual: vk.K2None})
	}
	v.k.Attn(r, vk.K2Attn{
		Q:       qToken,
		QStride: 512,
		K:       kToken,
		KStride: 512,
		V:       vToken,
		VStride: 512,
		O:       oToken,
		OStride: 512,
		Queries: P,
		Keys:    P,
		Group:   1,
		Scale:   float32(1.0 / math.Sqrt(512)),
		Heads:   1,
		Seqs:    1,
		HeadDim: 512,
	})
	v.k.Pix(r, vk.K2Pix{Src: oToken, Dst: u, C: 512, H: h, W: w, Op: vk.K2ToPlanes, Residual: vk.K2None})
	v.conv(r, a.proj, h, w, u, x, x, convNormal)
	return b
}

func (v *VAE) copyPlane(r *vk.Recorder, src, dst, count uint32) {
	v.k.Pix(r, vk.K2Pix{Src: src, Dst: dst, C: count, H: 1, W: 1, Op: vk.K2ToTokens, Residual: vk.K2None})
}

func (v *VAE) decodeProgram(h, w int, recordWPs bool) (vaeDecodeProgram, error) {
	key := [2]int{h, w}
	if !recordWPs {
		if p, ok := v.decodeProgs[key]; ok {
			return p, nil
		}
	}

	var prog vaeDecodeProgram
	p, err := v.k.Compile(func(r *vk.Recorder) {
		b := v.bufs
		H, W := uint32(h), uint32(w)

		// 1. Un-BN + Unpatch: latent in b[0] -> unpatched 32-ch at 2H, 2W in b[1].
		v.k.Flux3VAEUnpatch(r, vk.K2Flux3VAEPatch{
			Src:  b[0],
			Dst:  b[1],
			H:    H,
			W:    W,
			Mean: v.bnMean,
			Var:  v.bnVar,
		})
		b = [3]uint32{b[1], b[0], b[2]}
		H, W = 2*H, 2*W

		// 2. post_quant_conv
		v.conv(r, v.dec.postQuantConv, H, W, b[0], b[1], vk.K2None, convNormal)
		b = [3]uint32{b[1], b[0], b[2]}

		// 3. conv_in (32 -> 512)
		v.conv(r, v.dec.convIn, H, W, b[0], b[1], vk.K2None, convNormal)
		b = [3]uint32{b[1], b[0], b[2]}

		// 4. Middle
		b = v.residual(r, b, v.dec.mid0, H, W)
		b = v.attention(r, b, v.dec.midAttn, H, W)
		b = v.residual(r, b, v.dec.mid1, H, W)

		// 5. Up blocks: 3, 2, 1, 0
		for lvl := 3; lvl >= 0; lvl-- {
			up := v.dec.up[lvl]
			for bi := 0; bi < 3; bi++ {
				b = v.residual(r, b, up.blocks[bi], H, W)
			}
			if up.upsample != nil {
				H, W = 2*H, 2*W
				v.conv(r, *up.upsample, H, W, b[0], b[1], vk.K2None, convUpsample)
				b = [3]uint32{b[1], b[0], b[2]}
			}
			if recordWPs {
				switch lvl {
				case 3:
					v.copyPlane(r, b[0], v.wpBufs[3], 512*H*W)
					prog.up3 = v.wpBufs[3]
				case 2:
					v.copyPlane(r, b[0], v.wpBufs[2], 512*H*W)
					prog.up2 = v.wpBufs[2]
				case 1:
					v.copyPlane(r, b[0], v.wpBufs[1], 256*H*W)
					prog.up1 = v.wpBufs[1]
				case 0:
					v.copyPlane(r, b[0], v.wpBufs[0], 128*H*W)
					prog.up0 = v.wpBufs[0]
				}
			}
		}

		// 6. Head
		v.norm(r, b[0], b[1], 128, H, W, v.dec.normOut, true)
		v.conv(r, v.dec.convOut, H, W, b[1], b[2], vk.K2None, convNormal)
		prog.out = b[2]
	})
	if err != nil {
		return vaeDecodeProgram{}, err
	}
	prog.p = p
	if !recordWPs {
		v.decodeProgs[key] = prog
	}
	return prog, nil
}

func (v *VAE) encodeProgram(h, w int) (vaeEncodeProgram, error) {
	key := [2]int{h, w}
	if p, ok := v.encodeProgs[key]; ok {
		return p, nil
	}

	var out uint32
	p, err := v.k.Compile(func(r *vk.Recorder) {
		b := v.bufs
		H, W := uint32(h), uint32(w) // image dimensions (16*h_lat, 16*w_lat)

		// 1. conv_in: 3 -> 128
		v.conv(r, v.enc.convIn, H, W, b[0], b[1], vk.K2None, convNormal)
		b = [3]uint32{b[1], b[0], b[2]}

		// 2. Down blocks: 0, 1, 2, 3
		for lvl := 0; lvl < 4; lvl++ {
			down := v.enc.down[lvl]
			for bi := 0; bi < 2; bi++ {
				b = v.residual(r, b, down.blocks[bi], H, W)
			}
			if down.downsample != nil {
				H, W = H/2, W/2
				v.conv(r, *down.downsample, H, W, b[0], b[1], vk.K2None, convDownsample)
				b = [3]uint32{b[1], b[0], b[2]}
			}
		}

		// 3. Middle
		b = v.residual(r, b, v.enc.mid0, H, W)
		b = v.attention(r, b, v.enc.midAttn, H, W)
		b = v.residual(r, b, v.enc.mid1, H, W)

		// 4. End
		v.norm(r, b[0], b[1], 512, H, W, v.enc.normOut, true)
		v.conv(r, v.enc.convOut, H, W, b[1], b[2], vk.K2None, convNormal)

		// quant_conv: 64 -> 32
		v.conv(r, v.enc.quantConv, H, W, b[2], b[1], vk.K2None, convNormal)

		// 5. Patch 2x2 + BN: from 32-ch at 2H_lat, 2W_lat to 128-ch at H_lat, W_lat
		hLat, wLat := H/2, W/2
		v.k.Flux3VAEPatch(r, vk.K2Flux3VAEPatch{
			Src:  b[1],
			Dst:  b[0],
			H:    hLat,
			W:    wLat,
			Mean: v.bnMean,
			Var:  v.bnVar,
		})
		out = b[0]
	})
	if err != nil {
		return vaeEncodeProgram{}, err
	}
	prog := vaeEncodeProgram{p: p, out: out}
	v.encodeProgs[key] = prog
	return prog, nil
}

func toRGB(planes []float32, px int) []float32 {
	out := make([]float32, 3*px)
	for i := 0; i < px; i++ {
		for c := 0; c < 3; c++ {
			x := (planes[c*px+i] + 1.0) / 2.0
			out[3*i+c] = min(max(x, 0.0), 1.0)
		}
	}
	return out
}

func (v *VAE) decodeCore(latent []float32, h, w int, recordWPs bool) ([]float32, map[string][]float32, error) {
	if recordWPs && !v.recordWaypoints {
		return nil, nil, fmt.Errorf("flux3: decodeWithWaypoints called on VAE opened without waypoint buffers (use openVAEForTest)")
	}
	if len(latent) != 128*h*w {
		return nil, nil, fmt.Errorf("flux3: a latent of %d floats for 128 x %d x %d", len(latent), h, w)
	}
	if h*w > v.maxLatentPix {
		return nil, nil, fmt.Errorf("flux3: a latent of %d x %d is past the %d pixels the VAE was opened for", h, w, v.maxLatentPix)
	}
	p, err := v.decodeProgram(h, w, recordWPs)
	if err != nil {
		return nil, nil, err
	}
	if recordWPs {
		defer p.p.Close()
	}

	if err := v.k.Write(int(v.bufs[0]), latent); err != nil {
		return nil, nil, err
	}
	if err := p.p.Run(); err != nil {
		return nil, nil, err
	}

	px := 256 * h * w
	planes, err := v.k.Read(int(p.out), 3*px)
	if err != nil {
		return nil, nil, err
	}

	var wps map[string][]float32
	if recordWPs {
		wps = make(map[string][]float32)
		// up_3: 512 channels, 4h x 4w
		n3 := 512 * (4 * h) * (4 * w)
		d3, err := v.k.Read(int(p.up3), n3)
		if err != nil {
			return nil, nil, err
		}
		wps["vae/up_3"] = d3

		// up_2: 512 channels, 8h x 8w
		n2 := 512 * (8 * h) * (8 * w)
		d2, err := v.k.Read(int(p.up2), n2)
		if err != nil {
			return nil, nil, err
		}
		wps["vae/up_2"] = d2

		// up_1: 256 channels, 16h x 16w
		n1 := 256 * (16 * h) * (16 * w)
		d1, err := v.k.Read(int(p.up1), n1)
		if err != nil {
			return nil, nil, err
		}
		wps["vae/up_1"] = d1

		// up_0: 128 channels, 16h x 16w
		n0 := 128 * (16 * h) * (16 * w)
		d0, err := v.k.Read(int(p.up0), n0)
		if err != nil {
			return nil, nil, err
		}
		wps["vae/up_0"] = d0
	}

	return planes, wps, nil
}

// Decode turns a 128-ch x16 latent into an RGB image [0, 1] as float32,
// row by row, three floats per pixel.
func (v *VAE) Decode(latent []float32, h, w int) ([]float32, error) {
	planes, _, err := v.decodeCore(latent, h, w, false)
	if err != nil {
		return nil, err
	}
	return toRGB(planes, 256*h*w), nil
}

// decodeWithWaypoints turns a 128-ch x16 latent into RGB and returns per-block waypoints.
// Requires VAE opened via openVAEForTest.
func (v *VAE) decodeWithWaypoints(latent []float32, h, w int) ([]float32, map[string][]float32, error) {
	planes, wps, err := v.decodeCore(latent, h, w, true)
	if err != nil {
		return nil, nil, err
	}
	return toRGB(planes, 256*h*w), wps, nil
}

// Encode turns an RGB image in [0, 1] into a 128-ch latent.
func (v *VAE) Encode(rgb []float32, height, width int) ([]float32, error) {
	if len(rgb) != 3*height*width {
		return nil, fmt.Errorf("flux3: rgb len %d != 3 x %d x %d", len(rgb), height, width)
	}

	// Crop to multiples of 16 as in ComfyUI vae_encode_crop_pixels (sd.py lines 1150-1160).
	downscale := 16
	hCrop := (height / downscale) * downscale
	wCrop := (width / downscale) * downscale
	yOff := (height % downscale) / 2
	xOff := (width % downscale) / 2

	hLat := hCrop / downscale
	wLat := wCrop / downscale
	if hLat*wLat > v.maxLatentPix {
		return nil, fmt.Errorf("flux3: a latent of %d x %d is past the %d pixels the VAE was opened for", hLat, wLat, v.maxLatentPix)
	}

	// Convert interleaved RGB [0, 1] to planar [-1, 1]
	plane := hCrop * wCrop
	planes := make([]float32, 3*plane)
	for y := 0; y < hCrop; y++ {
		for x := 0; x < wCrop; x++ {
			srcPix := (y+yOff)*width + (x + xOff)
			dstPix := y*wCrop + x
			planes[0*plane+dstPix] = rgb[3*srcPix+0]*2.0 - 1.0
			planes[1*plane+dstPix] = rgb[3*srcPix+1]*2.0 - 1.0
			planes[2*plane+dstPix] = rgb[3*srcPix+2]*2.0 - 1.0
		}
	}

	p, err := v.encodeProgram(hCrop, wCrop)
	if err != nil {
		return nil, err
	}
	if err := v.k.Write(int(v.bufs[0]), planes); err != nil {
		return nil, err
	}
	if err := p.p.Run(); err != nil {
		return nil, err
	}

	nLat := 128 * hLat * wLat
	return v.k.Read(int(p.out), nLat)
}

// Trim gives the VAE's working memory back to the card.
func (v *VAE) Trim() {
	for _, p := range v.decodeProgs {
		p.p.Close()
	}
	v.decodeProgs = make(map[[2]int]vaeDecodeProgram)
	for _, p := range v.encodeProgs {
		p.p.Close()
	}
	v.encodeProgs = make(map[[2]int]vaeEncodeProgram)
	v.k.FreeArena()
}

// WeightBytes is what the VAE holds on the card.
func (v *VAE) WeightBytes() int {
	return v.k.WeightBytes()
}

// Close frees the card and the model tensors.
func (v *VAE) Close() {
	for _, p := range v.decodeProgs {
		p.p.Close()
	}
	v.decodeProgs = nil
	for _, p := range v.encodeProgs {
		p.p.Close()
	}
	v.encodeProgs = nil
	if v.k != nil {
		v.k.Close()
		v.k = nil
	}
	if v.tm != nil {
		v.tm.Close()
		v.tm = nil
	}
}
