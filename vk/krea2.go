package vk

// Krea 2 on the card: the kernels its three networks are made of, and the
// small machine that runs them. krea2/ lays the networks out; this file only
// knows buffers, offsets and dispatches.
//
// Every activation lives in one float32 arena, at offsets counted in floats,
// and every small parameter (norm scales, biases, the DiT's modulation
// vectors) in one float32 parameter buffer. The weights of the products are
// one buffer per tensor, fp8 or fp16, which is what keeps a 12 GB network
// under the four gigabytes one storage buffer addresses. Every kernel binds
// the same three things: a weight buffer, the arena, the parameters; and a
// kernel that reads no weight binds the parameters in its place. The products
// bind a fourth, the DiT's LoRA (SetLoRA), or the parameters when there is
// none.

import (
	_ "embed"
	"fmt"
	"time"
	"unsafe"
)

//go:generate glslc -O --target-env=vulkan1.1 -fshader-stage=compute -DFP8 shaders/krea2_mm.comp -o shaders/krea2_mm8.spv
//go:generate glslc -O --target-env=vulkan1.1 -fshader-stage=compute shaders/krea2_mm.comp -o shaders/krea2_mm16.spv
//go:generate glslc -O --target-env=vulkan1.1 -fshader-stage=compute -DFP8 -DXH shaders/krea2_mm.comp -o shaders/krea2_mm8h.spv
//go:generate glslc -O --target-env=vulkan1.1 -fshader-stage=compute -DXH shaders/krea2_mm.comp -o shaders/krea2_mm16h.spv
//go:generate glslc -O --target-env=vulkan1.1 -fshader-stage=compute -DFP8 -DXH -DBM=256 -DBN=256 shaders/krea2_mm.comp -o shaders/krea2_mm8hb.spv
//go:generate glslc -O --target-env=vulkan1.1 -fshader-stage=compute -DXH -DBM=256 -DBN=256 shaders/krea2_mm.comp -o shaders/krea2_mm16hb.spv
//go:generate glslc -O --target-env=vulkan1.1 -fshader-stage=compute shaders/krea2_norm.comp -o shaders/krea2_norm.spv
//go:generate glslc -O --target-env=vulkan1.1 -fshader-stage=compute shaders/krea2_rope.comp -o shaders/krea2_rope.spv
//go:generate glslc -O --target-env=vulkan1.1 -fshader-stage=compute shaders/krea2_act.comp -o shaders/krea2_act.spv
//go:generate glslc -O --target-env=vulkan1.1 -fshader-stage=compute shaders/krea2_attn.comp -o shaders/krea2_attn128.spv
//go:generate glslc -O --target-env=vulkan1.1 -fshader-stage=compute -DHD=384 -DBC=16 shaders/krea2_attn.comp -o shaders/krea2_attn384.spv
//go:generate glslc -O --target-env=vulkan1.1 -fshader-stage=compute -DHD=512 -DBC=16 shaders/krea2_attn.comp -o shaders/krea2_attn512.spv
//go:generate glslc -O --target-env=vulkan1.1 -fshader-stage=compute shaders/krea2_flash.comp -o shaders/krea2_flash128.spv
//go:generate glslc -O --target-env=vulkan1.1 -fshader-stage=compute -DGLOBAL shaders/krea2_flash.comp -o shaders/krea2_flashg128.spv
//go:generate glslc -O --target-env=vulkan1.1 -fshader-stage=compute shaders/krea2_kvhalf.comp -o shaders/krea2_kvhalf.spv
//go:generate glslc -O --target-env=vulkan1.1 -fshader-stage=compute shaders/krea2_conv.comp -o shaders/krea2_conv.spv
//go:generate glslc -O --target-env=vulkan1.1 -fshader-stage=compute shaders/krea2_pix.comp -o shaders/krea2_pix.spv
//go:generate glslc -O --target-env=vulkan1.1 -fshader-stage=compute shaders/flux3_norm.comp -o shaders/flux3_norm.spv
//go:generate glslc -O --target-env=vulkan1.1 -fshader-stage=compute shaders/flux3_qkrope.comp -o shaders/flux3_qkrope.spv
//go:generate glslc -O --target-env=vulkan1.1 -fshader-stage=compute shaders/flux3_gate.comp -o shaders/flux3_gate.spv
//go:generate glslc -O --target-env=vulkan1.1 -fshader-stage=compute shaders/flux3_gstats.comp -o shaders/flux3_gstats.spv
//go:generate glslc -O --target-env=vulkan1.1 -fshader-stage=compute shaders/flux3_gnorm.comp -o shaders/flux3_gnorm.spv
//go:generate glslc -O --target-env=vulkan1.1 -fshader-stage=compute shaders/flux3_vae_unpatch.comp -o shaders/flux3_vae_unpatch.spv
//go:generate glslc -O --target-env=vulkan1.1 -fshader-stage=compute shaders/flux3_vae_patch.comp -o shaders/flux3_vae_patch.spv

//go:embed shaders/krea2_mm8.spv
var krea2MM8SPIRV []byte

//go:embed shaders/krea2_mm16.spv
var krea2MM16SPIRV []byte

//go:embed shaders/krea2_mm8h.spv
var krea2MM8HSPIRV []byte

//go:embed shaders/krea2_mm16h.spv
var krea2MM16HSPIRV []byte

//go:embed shaders/krea2_mm8hb.spv
var krea2MM8HBSPIRV []byte

//go:embed shaders/krea2_mm16hb.spv
var krea2MM16HBSPIRV []byte

//go:embed shaders/krea2_norm.spv
var krea2NormSPIRV []byte

//go:embed shaders/krea2_rope.spv
var krea2RopeSPIRV []byte

//go:embed shaders/krea2_act.spv
var krea2ActSPIRV []byte

//go:embed shaders/krea2_attn128.spv
var krea2Attn128SPIRV []byte

//go:embed shaders/krea2_attn384.spv
var krea2Attn384SPIRV []byte

//go:embed shaders/krea2_attn512.spv
var krea2Attn512SPIRV []byte

//go:embed shaders/krea2_flash128.spv
var krea2Flash128SPIRV []byte

//go:embed shaders/krea2_flashg128.spv
var krea2FlashG128SPIRV []byte

//go:embed shaders/krea2_kvhalf.spv
var krea2KVHalfSPIRV []byte

//go:embed shaders/krea2_conv.spv
var krea2ConvSPIRV []byte

//go:embed shaders/krea2_pix.spv
var krea2PixSPIRV []byte

//go:embed shaders/flux3_norm.spv
var flux3NormSPIRV []byte

//go:embed shaders/flux3_qkrope.spv
var flux3QKRopeSPIRV []byte

//go:embed shaders/flux3_gate.spv
var flux3GateSPIRV []byte

//go:embed shaders/flux3_gstats.spv
var flux3GStatsSPIRV []byte

//go:embed shaders/flux3_gnorm.spv
var flux3GNormSPIRV []byte

//go:embed shaders/flux3_vae_unpatch.spv
var flux3VAEUnpatchSPIRV []byte

//go:embed shaders/flux3_vae_patch.spv
var flux3VAEPatchSPIRV []byte

type k2Kernel int

const (
	k2MM8 k2Kernel = iota
	k2MM16
	k2MM8H
	k2MM16H
	k2MM8HB
	k2MM16HB
	k2Norm
	k2Rope
	k2Act
	k2Attn128
	k2Attn384
	k2Attn512
	k2Flash128
	k2FlashG128
	k2KVHalf
	k2Conv
	k2Pix
	k2Flux3Norm
	k2Flux3Gate
	k2Flux3QKRope
	k2Flux3GStats
	k2Flux3GNorm
	k2Flux3VAEUnpatch
	k2Flux3VAEPatch
	k2Kernels
)

// K2MM is one product: Y[col][row] = Scale·Σ W[row][k]·X[col][k] + bias.
// See shaders/krea2_mm.comp.
type K2MM struct {
	Outputs, Inputs, Cols uint32
	WAt                   uint32 // the weight's first byte in its buffer
	X, XStride            uint32
	Y, YStride            uint32
	Bias                  uint32 // ^0 for none
	Scale                 float32
	Mode                  uint32 // K2Store, K2Add, K2GatedAdd
	GateA, GateB          uint32
	// A LoRA's B in the buffer SetLoRA gave, at byte LoRAAt, of LoRARank
	// columns (0 for none), against t at LoRAT in the arena.
	LoRAAt, LoRARank   uint32
	LoRAT, LoRATStride uint32
	// XHalves, when not 0, says X is fp16, and X and XStride count halves,
	// multiples of eight: what Norm and Act write with Halves set. The
	// kernels do not read it; it picks them.
	XHalves uint32
}

// What a product does with its answer.
const (
	K2Store    = 0
	K2Add      = 1
	K2GatedAdd = 2
	// K2SwiGLUHalves writes silu(gate)·up as fp16, Outputs/2 a column, Y and
	// YStride counting halves, from weights whose rows go in runs of 32 of
	// the gate's then 32 of the up's (SwiGLURows).
	K2SwiGLUHalves = 3
)

// SwiGLURow is where row r of a SwiGLU's stacked weight, the gate's n rows
// then the up's, goes for K2SwiGLUHalves: runs of 32 of each in turn.
func SwiGLURow(r, n int) int {
	up := 0
	if r >= n {
		r, up = r-n, 32
	}
	return r/32*64 + up + r%32
}

// K2None is the offset that says a kernel has no such input.
const K2None = ^uint32(0)

// K2Norm is an RMS norm over rows. See shaders/krea2_norm.comp.
type K2Norm struct {
	Src, SrcStride uint32
	Dst, DstStride uint32
	N, Rows        uint32
	Weight         uint32
	OnePlus        uint32
	Eps            float32
	ScaleA, ScaleB uint32
	ShiftA, ShiftB uint32
	// Halves writes the answer as fp16, and Dst and DstStride count halves;
	// N is even.
	Halves uint32
}

// K2Rope turns every pair of every head of every token in place.
type K2Rope struct {
	X, Stride  uint32
	Heads, Dim uint32
	Tokens     uint32
	Table      uint32
	Halves     uint32
}

// The elementwise steps of shaders/krea2_act.comp.
const (
	K2Gelu    = 0
	K2SwiGLU  = 1
	K2Sigmoid = 2
	K2Sum     = 3
	K2Copy    = 4
	K2SiLU    = 5
)

type K2Act struct {
	X, XStride uint32
	Y, YStride uint32
	N, Rows    uint32
	Op         uint32
	// With Halves set, X is left alone and the answer goes to Dst as fp16,
	// Dst and DstStride counting halves; N is even.
	Dst, DstStride uint32
	Halves         uint32
}

// K2Attn is attention over Seqs sequences at once. See shaders/krea2_attn.comp.
type K2Attn struct {
	Q, QStride, QSeq uint32
	K, KStride, KSeq uint32
	V, VStride, VSeq uint32
	O, OStride, OSeq uint32
	Queries, Keys    uint32
	Group            uint32
	Causal           uint32
	Tiles            uint32
	Scale            float32
	// Not pushed.
	Heads, Seqs, HeadDim int
}

func (a K2Attn) push() unsafe.Pointer {
	type push struct {
		q, qs, qq, k, ks, kq, v, vs, vq, o, os, oq, queries, keys, group, causal, tiles uint32
		scale                                                                           float32
	}
	p := &push{a.Q, a.QStride, a.QSeq, a.K, a.KStride, a.KSeq, a.V, a.VStride, a.VSeq,
		a.O, a.OStride, a.OSeq, a.Queries, a.Keys, a.Group, a.Causal, a.Tiles, a.Scale}
	return unsafe.Pointer(p)
}

// K2Conv is a convolution over channel-major planes, 3×3 with a padding of
// one or 1×1, as a product. See shaders/krea2_conv.comp.
type K2Conv struct {
	Outputs, Inputs uint32 // channels
	H, W            uint32
	Taps            uint32 // 9 or 1
	WAt, KStride    uint32 // weight's first byte; halves a row of weights
	X, Y            uint32
	Bias            uint32
	Residual        uint32 // added to the answer; K2None for none
	Up              uint32 // 1: the input is half the size, read upsampled
	Down            uint32 // 1: the input is twice the size, read downsampled
}

// The pixel steps of shaders/krea2_pix.comp.
const (
	K2ChanNorm = 0 // RMS over channels, times √C·γ, optionally SiLU
	K2Up2      = 1 // nearest, twice the size
	K2ToTokens = 2 // channel-major to token-major
	K2ToPlanes = 3 // token-major to channel-major, plus a residual
)

type K2Pix struct {
	Src, Dst uint32
	C, H, W  uint32
	Gamma    uint32
	SiLU     uint32
	Op       uint32
	Residual uint32
}

// K2Flux3Norm is LayerNorm with adaLN modulation.
type K2Flux3Norm struct {
	Src, SrcStride uint32
	Dst, DstStride uint32
	N, Rows        uint32
	Scale, Shift   uint32
	Eps            float32
	Halves         uint32
}

// K2Flux3Gate is gated addition: X += Out * Gate.
type K2Flux3Gate struct {
	X, XStride     uint32
	Out, OutStride uint32
	Gate           uint32
	N, Rows        uint32
}

// K2Flux3QKRope is the RMS norm of every head of 128 of the queries and the
// keys, times their weights, then their rotary position from a table of a
// token's 64 cosines then 64 sines, in place. See shaders/flux3_qkrope.comp.
type K2Flux3QKRope struct {
	Q, K, Stride     uint32
	Heads, Tokens    uint32
	Table            uint32
	QWeight, KWeight uint32
	Eps              float32
}

// K2Flux3GStats is pass 1 of GroupNorm with 32 groups.
type K2Flux3GStats struct {
	Src, Plane, CPG, GroupElems uint32
	SliceLen, Slices, Stats     uint32
}

// K2Flux3GNorm is pass 2 of GroupNorm with 32 groups.
type K2Flux3GNorm struct {
	Src, Dst                    uint32
	Plane, CPG, GroupElems      uint32
	Slices, Stats, Weight, Bias uint32
	SiLU                        uint32
}

// K2Flux3VAEPatch is VAE patch and unpatch.
type K2Flux3VAEPatch struct {
	Src, Dst  uint32
	H, W      uint32
	Mean, Var uint32
}

// K2 is a device set up for Krea 2's kernels, with one arena and one
// parameter buffer.
type K2 struct {
	d           *Device
	pipes       [k2Kernels]*Pipeline
	sets        map[k2SetKey]*Set
	arenaFloats int
	arena       *Buffer
	params      *Buffer
	readback    *Buffer
	weights     []*Buffer
	lora        int // the weights the products' fourth buffer is; -1 for none
	// The fp16 keys and values of the attention, which lives and dies with
	// the arena; ReserveAttention sizes it.
	kv      *Buffer
	kvBytes int
}

type k2SetKey struct {
	k k2Kernel
	w int
}

var k2Specs = [k2Kernels]struct {
	spirv *[]byte
	push  uintptr
}{
	k2MM8:             {&krea2MM8SPIRV, unsafe.Sizeof(K2MM{})},
	k2MM16:            {&krea2MM16SPIRV, unsafe.Sizeof(K2MM{})},
	k2MM8H:            {&krea2MM8HSPIRV, unsafe.Sizeof(K2MM{})},
	k2MM16H:           {&krea2MM16HSPIRV, unsafe.Sizeof(K2MM{})},
	k2MM8HB:           {&krea2MM8HBSPIRV, unsafe.Sizeof(K2MM{})},
	k2MM16HB:          {&krea2MM16HBSPIRV, unsafe.Sizeof(K2MM{})},
	k2Norm:            {&krea2NormSPIRV, unsafe.Sizeof(K2Norm{})},
	k2Rope:            {&krea2RopeSPIRV, unsafe.Sizeof(K2Rope{})},
	k2Act:             {&krea2ActSPIRV, unsafe.Sizeof(K2Act{})},
	k2Attn128:         {&krea2Attn128SPIRV, 18 * 4},
	k2Attn384:         {&krea2Attn384SPIRV, 18 * 4},
	k2Attn512:         {&krea2Attn512SPIRV, 18 * 4},
	k2Flash128:        {&krea2Flash128SPIRV, 18 * 4},
	k2FlashG128:       {&krea2FlashG128SPIRV, 18 * 4},
	k2KVHalf:          {&krea2KVHalfSPIRV, 8 * 4},
	k2Conv:            {&krea2ConvSPIRV, unsafe.Sizeof(K2Conv{})},
	k2Pix:             {&krea2PixSPIRV, unsafe.Sizeof(K2Pix{})},
	k2Flux3Norm:       {&flux3NormSPIRV, unsafe.Sizeof(K2Flux3Norm{})},
	k2Flux3Gate:       {&flux3GateSPIRV, unsafe.Sizeof(K2Flux3Gate{})},
	k2Flux3QKRope:     {&flux3QKRopeSPIRV, unsafe.Sizeof(K2Flux3QKRope{})},
	k2Flux3GStats:     {&flux3GStatsSPIRV, unsafe.Sizeof(K2Flux3GStats{})},
	k2Flux3GNorm:      {&flux3GNormSPIRV, unsafe.Sizeof(K2Flux3GNorm{})},
	k2Flux3VAEUnpatch: {&flux3VAEUnpatchSPIRV, unsafe.Sizeof(K2Flux3VAEPatch{})},
	k2Flux3VAEPatch:   {&flux3VAEPatchSPIRV, unsafe.Sizeof(K2Flux3VAEPatch{})},
}

// NewK2 makes the pipelines, an arena of arenaFloats and a parameter buffer
// holding params. The device must have the matrix cores: there is no other
// path, and a model this size without them is not one anybody would wait for.
func NewK2(d *Device, arenaFloats int, params []float32) (*K2, error) {
	if !d.Coopmat() {
		return nil, fmt.Errorf("vk: Krea 2 wants cooperative matrices, which this device does not offer")
	}
	k := &K2{d: d, sets: map[k2SetKey]*Set{}, lora: -1}
	fail := func(err error) (*K2, error) {
		k.Close()
		return nil, err
	}
	k.arenaFloats = max(arenaFloats, 1)
	if uint64(k.arenaFloats)*4 >= 1<<32 {
		return fail(fmt.Errorf("vk: Krea 2 arena of %d floats is past what one storage buffer addresses", k.arenaFloats))
	}
	var err error
	if len(params) == 0 {
		params = []float32{0}
	}
	if k.params, err = d.Upload(floatBytes(params)); err != nil {
		return fail(err)
	}
	for i, spec := range k2Specs {
		bindings := 3
		if k2Kernel(i).product() {
			bindings = 4
		}
		wave := uint32(coopmatWave)
		if k2Kernel(i) == k2Flash128 || k2Kernel(i) == k2FlashG128 {
			// Written for the matrix cores' own width; see the shader.
			wave = 32
		}
		if k.pipes[i], err = d.newPipeline(*spec.spirv, bindings, uint32(spec.push), wave, nil); err != nil {
			return fail(fmt.Errorf("vk: Krea 2 kernel %d: %w", i, err))
		}
	}
	return k, nil
}

// Device is the device the machine runs on.
func (k *K2) Device() *Device { return k.d }

// AddWeights uploads one tensor's bytes and returns its handle.
func (k *K2) AddWeights(data []byte) (int, error) { return k.addWeights(data, false) }

// AddHostWeights keeps one tensor's bytes in system memory the card reads
// across the bus, and returns its handle. It is for weights read once per
// use, which cost the bus once rather than a place on the card for good.
func (k *K2) AddHostWeights(data []byte) (int, error) { return k.addWeights(data, true) }

func (k *K2) addWeights(data []byte, host bool) (int, error) {
	if len(data)%16 != 0 {
		// The products read sixteen bytes at a time.
		padded := make([]byte, (len(data)+15)/16*16)
		copy(padded, data)
		data = padded
	}
	var b *Buffer
	var err error
	if host {
		b, err = k.d.HostResident(data)
	} else {
		b, err = k.d.Upload(data)
	}
	if err != nil {
		return 0, err
	}
	k.weights = append(k.weights, b)
	return len(k.weights) - 1, nil
}

// SetLoRA makes the weights of handle h, or none when h is -1, what the
// products read a LoRA's B from.
func (k *K2) SetLoRA(h int) {
	if h == k.lora {
		return
	}
	for key, s := range k.sets {
		if key.k.product() {
			s.Close()
			delete(k.sets, key)
		}
	}
	k.lora = h
}

// FreeWeights gives back the tensors of these handles, which no program may
// read afterwards. The handles are not handed out again.
func (k *K2) FreeWeights(handles ...int) {
	for _, h := range handles {
		for key, s := range k.sets {
			if key.w == h {
				s.Close()
				delete(k.sets, key)
			}
		}
		if k.weights[h] != nil {
			k.weights[h].Close()
			k.weights[h] = nil
		}
		if h == k.lora {
			k.SetLoRA(-1)
		}
	}
}

// Arena makes sure the arena exists, making it again after FreeArena.
//
// The arena of another machine freed just before (FLUX 3's DiT before its
// VAE decodes) goes back to the card some tens of milliseconds later: asked
// for at once, a large arena lands in system memory, where a decode takes
// past the kernel's ten seconds and the card is reset. So it waits, a second
// at most, for the room DeviceLocalFree reports, then asks whatever it got.
func (k *K2) Arena() error {
	if k.arena != nil {
		return nil
	}
	want := uint64(k.arenaFloats) * 4
	for deadline := time.Now().Add(time.Second); ; time.Sleep(5 * time.Millisecond) {
		if free := k.d.DeviceLocalFree(); free == 0 || free >= want || time.Now().After(deadline) {
			break
		}
	}
	var err error
	if k.arena, err = k.d.Local(k.arenaFloats*4, bufferUsageStorage|bufferUsageTransferSrc|bufferUsageTransferDst); err != nil {
		return err
	}
	if k.kvBytes > 0 {
		if k.kv, err = k.d.Local(k.kvBytes, bufferUsageStorage); err != nil {
			k.arena.Close()
			k.arena = nil
			return err
		}
	}
	return nil
}

// kvRows is the keys a head of the fp16 copy holds: the keys rounded up to a
// tile of every attention kernel.
func kvRows(keys uint32) uint32 { return (keys + 63) / 64 * 64 }

// ReserveAttention makes room, beside the arena, for an fp16 copy of the
// keys and values of attentions of up to keys keys over kvHeads heads of 128,
// which the long attentions read instead of the arena (see Attn). It takes
// effect with the next Arena; without it, or past it, Attn reads the arena.
func (k *K2) ReserveAttention(keys, kvHeads int) {
	k.kvBytes = 2 * 2 * kvHeads * 128 * int(kvRows(uint32(keys)))
}

// KVBytes is what ReserveAttention keeps on the card while the arena is.
func (k *K2) KVBytes() int { return k.kvBytes }

// FreeArena gives the arena's memory back to the card. Every program
// recorded before it is dead: close them first, and record again after
// Arena.
func (k *K2) FreeArena() {
	for key, s := range k.sets {
		s.Close()
		delete(k.sets, key)
	}
	if k.arena != nil {
		k.arena.Close()
		k.arena = nil
	}
	if k.kv != nil {
		k.kv.Close()
		k.kv = nil
	}
}

// WeightBytes is what the uploaded weights hold on the card.
func (k *K2) WeightBytes() int {
	n := 0
	for _, b := range k.weights {
		if b != nil {
			n += b.Size()
		}
	}
	return n
}

func (k *K2) set(kern k2Kernel, w int) *Set {
	key := k2SetKey{kern, w}
	if s, ok := k.sets[key]; ok {
		return s
	}
	first := k.params
	if w >= 0 {
		first = k.weights[w]
	}
	if kern == k2KVHalf || kern == k2FlashG128 {
		first = k.kv
	}
	bufs := []*Buffer{first, k.arena, k.params}
	if kern.product() {
		lora := k.params
		if k.lora >= 0 {
			lora = k.weights[k.lora]
		}
		bufs = append(bufs, lora)
	}
	s, err := k.pipes[kern].NewSet(bufs)
	if err != nil {
		panic(fmt.Sprintf("vk: Krea 2 set for kernel %d: %v", kern, err))
	}
	k.sets[key] = s
	return s
}

// grid spreads n workgroups over two axes when one does not hold them; every
// Krea 2 kernel folds the second axis back into one index.
func grid(n int) (uint32, uint32) {
	if n <= 65535 {
		return uint32(n), 1
	}
	return 65535, uint32((n + 65534) / 65535)
}

func (k *K2) dispatch(r *Recorder, kern k2Kernel, w int, groups int, push unsafe.Pointer) {
	if groups <= 0 {
		return
	}
	x, y := grid(groups)
	r.DispatchColumns(k.set(kern, w), x, y, push)
	r.Barrier()
}

func (k k2Kernel) product() bool { return k.tile() > 0 }

// tile is a product's tile, rows and columns; 0 for any other kernel.
func (k k2Kernel) tile() uint32 {
	switch k {
	case k2MM8, k2MM16, k2MM8H, k2MM16H:
		return 128
	case k2MM8HB, k2MM16HB:
		return 256
	}
	return 0
}

// k2BigTiles is how many tiles of 256 × 256 a product needs before it takes
// them: fewer leave the card's 64 compute units half idle, and the tiles of
// 128 are quicker there (the DiT's k and v, 1536 rows).
const k2BigTiles = 128

// k2BigTiles16 is the same for fp16 weights, whose tiles of 128 lose less
// to their reads: FLUX 3's MLP output over 4096 tokens (192 big tiles) is
// quicker in tiles of 128, its input (1152) and its 3072-square products
// at 4 MP (780) in tiles of 256.
const k2BigTiles16 = 512

// MM records a product whose weights are fp8 (fp8 true) or fp16.
func (k *K2) MM(r *Recorder, w int, fp8 bool, m K2MM) {
	kern := k2MM16
	if fp8 {
		kern = k2MM8
	}
	if m.XHalves != 0 {
		if (m.X|m.XStride)%8 != 0 {
			panic(fmt.Sprintf("vk: Krea 2 product of fp16 X at %d, stride %d", m.X, m.XStride))
		}
		kern += k2MM8H - k2MM8
		// A tile of 256 × 256 reads half the bytes a product of 128 does,
		// which is what these products wait on.
		big := ((m.Outputs + 255) / 256) * ((m.Cols + 255) / 256)
		if kern == k2MM8H && big >= k2BigTiles {
			kern = k2MM8HB
		}
		if kern == k2MM16H && big >= k2BigTiles16 {
			kern = k2MM16HB
		}
	}
	b := kern.tile()
	tiles := int((m.Outputs+b-1)/b) * int((m.Cols+b-1)/b)
	k.dispatch(r, kern, w, tiles, unsafe.Pointer(&m))
}

func (k *K2) Norm(r *Recorder, n K2Norm) { k.dispatch(r, k2Norm, -1, int(n.Rows), unsafe.Pointer(&n)) }

func (k *K2) Rope(r *Recorder, p K2Rope) {
	k.dispatch(r, k2Rope, -1, groups256(int(p.Tokens*p.Heads*p.Dim/2)), unsafe.Pointer(&p))
}

func (k *K2) Act(r *Recorder, a K2Act) {
	n := int(a.N * a.Rows)
	if a.Halves != 0 {
		n /= 2
	}
	k.dispatch(r, k2Act, -1, groups256(n), unsafe.Pointer(&a))
}

// k2FlashRows is the queries a workgroup of shaders/krea2_flash.comp takes.
const k2FlashRows = 128

// flash says whether shaders/krea2_flash.comp takes this attention: no mask,
// heads of 128, enough queries to fill its workgroups, and keys and values
// it can read four floats at a time.
func (a K2Attn) flash() bool {
	return a.HeadDim == 128 && a.Causal == 0 && a.Queries >= 512 &&
		(a.K|a.KStride|a.KSeq|a.V|a.VStride|a.VSeq)%4 == 0
}

// Attn records attention; HeadDim is 128, 384 or 512.
func (k *K2) Attn(r *Recorder, a K2Attn) {
	kern := k2Attn128
	if a.HeadDim == 384 {
		kern = k2Attn384
	} else if a.HeadDim == 512 {
		kern = k2Attn512
	} else if a.HeadDim != 128 {
		panic(fmt.Sprintf("vk: Krea 2 attention has no kernel for heads of %d", a.HeadDim))
	}
	rows := uint32(64)
	if a.flash() {
		kern, rows = k2Flash128, k2FlashRows
		if a.Seqs == 1 && k.kv != nil && 2*2*(a.Heads/int(a.Group))*128*int(kvRows(a.Keys)) <= k.kvBytes {
			k.attnFromHalves(r, a)
			return
		}
	}
	a.Tiles = (a.Queries + rows - 1) / rows
	groups := int(a.Tiles) * a.Seqs
	if groups > 65535 || a.Heads > 65535 {
		panic(fmt.Sprintf("vk: Krea 2 attention of %d×%d workgroups", groups, a.Heads))
	}
	r.DispatchColumns(k.set(kern, -1), uint32(groups), uint32(a.Heads), a.push())
	r.Barrier()
}

// attnFromHalves records the long attention in two: the keys and values to
// fp16 in k.kv, the values transposed, then shaders/krea2_flash.comp reading
// them there.
func (k *K2) attnFromHalves(r *Recorder, a K2Attn) {
	kvHeads := uint32(a.Heads) / a.Group
	np := kvRows(a.Keys)
	half := struct{ k, kStride, v, vStride, tokens, np, kh, vt uint32 }{
		a.K, a.KStride, a.V, a.VStride, a.Keys, np, 0, kvHeads * 128 * np}
	r.DispatchColumns(k.set(k2KVHalf, -1), np/64, kvHeads, unsafe.Pointer(&half))
	r.Barrier()
	a.K, a.KSeq = half.kh, 0
	a.V, a.VStride, a.VSeq = half.vt, np, 0
	a.Tiles = (a.Queries + k2FlashRows - 1) / k2FlashRows
	r.DispatchColumns(k.set(k2FlashG128, -1), a.Tiles, uint32(a.Heads), a.push())
	r.Barrier()
}

// Conv records a convolution whose weights are fp16.
func (k *K2) Conv(r *Recorder, w int, c K2Conv) {
	tiles := int((c.Outputs+127)/128) * int((c.H*c.W+127)/128)
	k.dispatch(r, k2Conv, w, tiles, unsafe.Pointer(&c))
}

func (k *K2) Pix(r *Recorder, p K2Pix) {
	n := int(p.H * p.W)
	switch p.Op {
	case K2Up2:
		n = int(p.C * p.H * p.W * 4)
	case K2ToTokens, K2ToPlanes:
		n = int(p.C * p.H * p.W)
	}
	k.dispatch(r, k2Pix, -1, groups256(n), unsafe.Pointer(&p))
}

func (k *K2) Flux3Norm(r *Recorder, n K2Flux3Norm) {
	k.dispatch(r, k2Flux3Norm, -1, int(n.Rows), unsafe.Pointer(&n))
}

func (k *K2) Flux3QKRope(r *Recorder, q K2Flux3QKRope) {
	k.dispatch(r, k2Flux3QKRope, -1, int((2*q.Tokens*q.Heads+3)/4), unsafe.Pointer(&q))
}

func (k *K2) Flux3Gate(r *Recorder, g K2Flux3Gate) {
	k.dispatch(r, k2Flux3Gate, -1, groups256(int(g.N*g.Rows)), unsafe.Pointer(&g))
}

func (k *K2) Flux3GStats(r *Recorder, s K2Flux3GStats) {
	if s.Slices > 65535 {
		panic(fmt.Sprintf("vk: Flux3 GroupNorm stats of %d slices is past the 65535 limit", s.Slices))
	}
	r.DispatchColumns(k.set(k2Flux3GStats, -1), s.Slices, 32, unsafe.Pointer(&s))
	r.Barrier()
}

func (k *K2) Flux3GNorm(r *Recorder, n K2Flux3GNorm) {
	groups := (n.GroupElems + 255) / 256
	if groups > 65535 {
		panic(fmt.Sprintf("vk: Flux3 GroupNorm of %d workgroups is past the 65535 limit", groups))
	}
	r.DispatchColumns(k.set(k2Flux3GNorm, -1), groups, 32, unsafe.Pointer(&n))
	r.Barrier()
}

func (k *K2) Flux3VAEUnpatch(r *Recorder, u K2Flux3VAEPatch) {
	n := 128 * int(u.H*u.W)
	k.dispatch(r, k2Flux3VAEUnpatch, -1, groups256(n), unsafe.Pointer(&u))
}

func (k *K2) Flux3VAEPatch(r *Recorder, p K2Flux3VAEPatch) {
	n := 128 * int(p.H*p.W)
	k.dispatch(r, k2Flux3VAEPatch, -1, groups256(n), unsafe.Pointer(&p))
}

// CopyStrided records count copies of n floats inside the arena, the i-th
// from src+i*srcStride to dst+i*dstStride (all in floats), in one transfer,
// then a barrier. It is how a band of planes is cut out of a picture.
func (k *K2) CopyStrided(r *Recorder, dst, dstStride, src, srcStride, n, count uint32) {
	if count == 0 || n == 0 {
		return
	}
	regions := make([]bufferCopy, count)
	for i := range regions {
		regions[i] = bufferCopy{
			srcOffset: 4 * (uint64(src) + uint64(i)*uint64(srcStride)),
			dstOffset: 4 * (uint64(dst) + uint64(i)*uint64(dstStride)),
			size:      4 * uint64(n),
		}
	}
	vkCmdCopyBuffer(r.cb, k.arena.handle, k.arena.handle, count, &regions[0])
	r.Barrier()
}

// Compile records a program over this machine's buffers.
func (k *K2) Compile(record func(r *Recorder)) (*Program, error) {
	if err := k.Arena(); err != nil {
		return nil, err
	}
	return k.d.Compile(record)
}

// Write puts floats into the arena at an offset in floats.
func (k *K2) Write(at int, data []float32) error {
	if len(data) == 0 {
		return nil
	}
	if err := k.Arena(); err != nil {
		return err
	}
	return k.d.CopyInto(k.arena, at*4, floatBytes(data))
}

// Read copies n floats out of the arena.
func (k *K2) Read(at, n int) ([]float32, error) {
	if err := k.Arena(); err != nil {
		return nil, err
	}
	if k.readback == nil || k.readback.Size() < n*4 {
		if k.readback != nil {
			k.readback.Close()
		}
		var err error
		if k.readback, err = k.d.Readback(max(n*4, 1<<20), bufferUsageStorage|bufferUsageTransferDst); err != nil {
			k.readback = nil
			return nil, err
		}
	}
	if err := k.d.Submit(func(r *Recorder) { r.CopyFrom(k.readback, 0, k.arena, at*4, n*4) }); err != nil {
		return nil, err
	}
	out := make([]float32, n)
	copy(out, k.readback.Floats()[:n])
	return out, nil
}

// Close releases everything.
func (k *K2) Close() {
	for key, s := range k.sets {
		s.Close()
		delete(k.sets, key)
	}
	for i, p := range k.pipes {
		if p != nil {
			p.Close()
			k.pipes[i] = nil
		}
	}
	for _, b := range k.weights {
		if b != nil {
			b.Close()
		}
	}
	k.weights = nil
	for _, b := range []**Buffer{&k.arena, &k.params, &k.readback} {
		if *b != nil {
			(*b).Close()
			*b = nil
		}
	}
}
