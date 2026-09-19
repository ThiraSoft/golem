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
// the same three things — a weight buffer, the arena, the parameters — and a
// kernel that reads no weight binds the parameters in its place.

import (
	_ "embed"
	"fmt"
	"unsafe"
)

//go:generate glslc -O --target-env=vulkan1.1 -fshader-stage=compute -DFP8 shaders/krea2_mm.comp -o shaders/krea2_mm8.spv
//go:generate glslc -O --target-env=vulkan1.1 -fshader-stage=compute shaders/krea2_mm.comp -o shaders/krea2_mm16.spv
//go:generate glslc -O --target-env=vulkan1.1 -fshader-stage=compute shaders/krea2_norm.comp -o shaders/krea2_norm.spv
//go:generate glslc -O --target-env=vulkan1.1 -fshader-stage=compute shaders/krea2_rope.comp -o shaders/krea2_rope.spv
//go:generate glslc -O --target-env=vulkan1.1 -fshader-stage=compute shaders/krea2_act.comp -o shaders/krea2_act.spv
//go:generate glslc -O --target-env=vulkan1.1 -fshader-stage=compute shaders/krea2_attn.comp -o shaders/krea2_attn128.spv
//go:generate glslc -O --target-env=vulkan1.1 -fshader-stage=compute -DHD=384 -DBC=32 shaders/krea2_attn.comp -o shaders/krea2_attn384.spv
//go:generate glslc -O --target-env=vulkan1.1 -fshader-stage=compute shaders/krea2_conv.comp -o shaders/krea2_conv.spv
//go:generate glslc -O --target-env=vulkan1.1 -fshader-stage=compute shaders/krea2_pix.comp -o shaders/krea2_pix.spv

//go:embed shaders/krea2_mm8.spv
var krea2MM8SPIRV []byte

//go:embed shaders/krea2_mm16.spv
var krea2MM16SPIRV []byte

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

//go:embed shaders/krea2_conv.spv
var krea2ConvSPIRV []byte

//go:embed shaders/krea2_pix.spv
var krea2PixSPIRV []byte

type k2Kernel int

const (
	k2MM8 k2Kernel = iota
	k2MM16
	k2Norm
	k2Rope
	k2Act
	k2Attn128
	k2Attn384
	k2Conv
	k2Pix
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
}

// What a product does with its answer.
const (
	K2Store    = 0
	K2Add      = 1
	K2GatedAdd = 2
)

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
}

type k2SetKey struct {
	k k2Kernel
	w int
}

var k2Specs = [k2Kernels]struct {
	spirv *[]byte
	push  uintptr
}{
	k2MM8:     {&krea2MM8SPIRV, unsafe.Sizeof(K2MM{})},
	k2MM16:    {&krea2MM16SPIRV, unsafe.Sizeof(K2MM{})},
	k2Norm:    {&krea2NormSPIRV, unsafe.Sizeof(K2Norm{})},
	k2Rope:    {&krea2RopeSPIRV, unsafe.Sizeof(K2Rope{})},
	k2Act:     {&krea2ActSPIRV, unsafe.Sizeof(K2Act{})},
	k2Attn128: {&krea2Attn128SPIRV, 18 * 4},
	k2Attn384: {&krea2Attn384SPIRV, 18 * 4},
	k2Conv:    {&krea2ConvSPIRV, unsafe.Sizeof(K2Conv{})},
	k2Pix:     {&krea2PixSPIRV, unsafe.Sizeof(K2Pix{})},
}

// NewK2 makes the pipelines, an arena of arenaFloats and a parameter buffer
// holding params. The device must have the matrix cores: there is no other
// path, and a model this size without them is not one anybody would wait for.
func NewK2(d *Device, arenaFloats int, params []float32) (*K2, error) {
	if !d.Coopmat() {
		return nil, fmt.Errorf("vk: Krea 2 wants cooperative matrices, which this device does not offer")
	}
	k := &K2{d: d, sets: map[k2SetKey]*Set{}}
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
		if k.pipes[i], err = d.newPipeline(*spec.spirv, 3, uint32(spec.push), coopmatWave, nil); err != nil {
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

// Arena makes sure the arena exists, making it again after FreeArena.
func (k *K2) Arena() error {
	if k.arena != nil {
		return nil
	}
	var err error
	k.arena, err = k.d.Local(k.arenaFloats*4, bufferUsageStorage|bufferUsageTransferSrc|bufferUsageTransferDst)
	return err
}

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
	s, err := k.pipes[kern].NewSet([]*Buffer{first, k.arena, k.params})
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

// MM records a product whose weights are fp8 (fp8 true) or fp16.
func (k *K2) MM(r *Recorder, w int, fp8 bool, m K2MM) {
	kern := k2MM16
	if fp8 {
		kern = k2MM8
	}
	tiles := int((m.Outputs+127)/128) * int((m.Cols+127)/128)
	k.dispatch(r, kern, w, tiles, unsafe.Pointer(&m))
}

func (k *K2) Norm(r *Recorder, n K2Norm) { k.dispatch(r, k2Norm, -1, int(n.Rows), unsafe.Pointer(&n)) }

func (k *K2) Rope(r *Recorder, p K2Rope) {
	k.dispatch(r, k2Rope, -1, groups256(int(p.Tokens*p.Heads*p.Dim/2)), unsafe.Pointer(&p))
}

func (k *K2) Act(r *Recorder, a K2Act) {
	k.dispatch(r, k2Act, -1, groups256(int(a.N*a.Rows)), unsafe.Pointer(&a))
}

// Attn records attention; HeadDim is 128 or 384.
func (k *K2) Attn(r *Recorder, a K2Attn) {
	kern := k2Attn128
	if a.HeadDim == 384 {
		kern = k2Attn384
	} else if a.HeadDim != 128 {
		panic(fmt.Sprintf("vk: Krea 2 attention has no kernel for heads of %d", a.HeadDim))
	}
	a.Tiles = (a.Queries + 63) / 64
	groups := int(a.Tiles) * a.Seqs
	if groups > 65535 || a.Heads > 65535 {
		panic(fmt.Sprintf("vk: Krea 2 attention of %d×%d workgroups", groups, a.Heads))
	}
	r.DispatchColumns(k.set(kern, -1), uint32(groups), uint32(a.Heads), a.push())
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
