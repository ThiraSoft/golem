package flux3

// The DiT: FLUX 3 Image Klein's multi-modal transformer (comfy/ldm/flux3/model.py),
// from ComfyUI's bf16 checkpoint converted to fp16 (default) or fp8 e4m3 with per-tensor scale.

import (
	"encoding/binary"
	"fmt"
	"math"
	"runtime"
	"runtime/debug"
	"sync"
	"sync/atomic"
	"syscall"
	"unsafe"

	"github.com/ThiraSoft/golem/tensors"
	"github.com/ThiraSoft/golem/vk"
)

func madviseDontNeed(b []byte) {
	if len(b) == 0 {
		return
	}
	ptr := uintptr(unsafe.Pointer(&b[0]))
	pageSize := uintptr(4096)
	alignedStart := ptr &^ (pageSize - 1)
	endPtr := ptr + uintptr(len(b))
	alignedEnd := (endPtr + pageSize - 1) &^ (pageSize - 1)
	_, _, _ = syscall.Syscall(syscall.SYS_MADVISE, alignedStart, alignedEnd-alignedStart, uintptr(syscall.MADV_DONTNEED))
}

const (
	ditWidth       = 3072
	ditHeads       = 24
	ditHeadDim     = 128
	ditFFN         = 9216 // 3.0 * hidden_size
	ditInChannels  = 128
	ditTxtIn       = CondWidth // 20480
	ditTDim        = 256
	ditEps         = 1e-6
	MaxImageTokens = 4096 // 1024x1024: (1024/16)*(1024/16) = 64*64 = 4096
	MaxRefTokens   = 4096 // what OpenDiT sizes for: one 1 MP reference
	MaxTextTokens  = 960
	// RefTokens is one reference at the workflows' 1 MP, MaxRefs how many
	// ReferenceMultiLatent takes.
	RefTokens = 4096
	MaxRefs   = 4
	// mlpChunk is how many tokens one pass of a block's MLP covers: its
	// 2 x 9216 wide input is the largest buffer a token needs, so the arena
	// holds it for these tokens only.
	mlpChunk = 4096
)

// Precision specifies the weight storage precision in DiT.
type Precision int

const (
	PrecisionFP16 Precision = iota
	PrecisionFP8
)

// e4m3 decodes an fp8 e4m3fn byte into float32.
func e4m3(b byte) float32 {
	sign := float32(1)
	if b&0x80 != 0 {
		sign = -1
	}
	e, m := int(b>>3&15), float64(b&7)
	if e == 0 {
		return sign * float32(m/8*math.Pow(2, -6))
	}
	return sign * float32((1+m/8)*math.Pow(2, float64(e-7)))
}

// e4m3Midpoints holds the 126 midpoints between adjacent positive e4m3fn values (0..126).
var e4m3Midpoints [126]float32
var bf16ToHalfLUT [65536]uint16

func init() {
	var vals [127]float32
	for b := 0; b < 127; b++ {
		vals[b] = e4m3(byte(b))
	}
	for i := 0; i < 126; i++ {
		e4m3Midpoints[i] = (vals[i] + vals[i+1]) * 0.5
	}
	for u := 0; u < 65536; u++ {
		f := math.Float32frombits(uint32(u) << 16)
		bf16ToHalfLUT[u] = floatToHalf(f)
	}
}

func floatToHalf(f float32) uint16 {
	b := math.Float32bits(f)
	sign := uint16(b>>16) & 0x8000
	exp := int(b>>23&0xff) - 127 + 15
	mant := b & 0x7fffff
	switch {
	case exp >= 31:
		return sign | 0x7c00
	case exp <= 0:
		if exp < -10 {
			return sign
		}
		mant |= 0x800000
		shift := uint(14 - exp)
		half := mant >> shift
		rem := mant & (1<<shift - 1)
		if rem > 1<<(shift-1) || (rem == 1<<(shift-1) && half&1 == 1) {
			half++
		}
		return sign | uint16(half)
	}
	half := uint32(exp)<<10 | mant>>13
	rem := mant & 0x1fff
	if rem > 0x1000 || (rem == 0x1000 && half&1 == 1) {
		half++
	}
	return sign | uint16(half)
}

func floatToE4M3(f float32) byte {
	if math.IsNaN(float64(f)) {
		return 0x7f
	}
	sign := byte(0)
	if f < 0 || math.Float32bits(f)&0x80000000 != 0 {
		sign = 0x80
		f = -f
	}
	if f >= 448.0 {
		return sign | 0x7e
	}
	lo, hi := 0, 126
	for lo < hi {
		mid := (lo + hi) / 2
		if f > e4m3Midpoints[mid] {
			lo = mid + 1
		} else {
			hi = mid
		}
	}
	return sign | byte(lo)
}

type linearWeight struct {
	w       int
	scale   float32
	outputs int
	inputs  int
}

type ditBlock struct {
	q, k, v       linearWeight
	qNorm, kNorm  uint32
	attnOut       linearWeight
	mlpIn, mlpOut linearWeight
}

type span struct {
	start, end int
	modOffset  uint32
}

type stepKey struct {
	slot        int
	h, w        int
	refTokens   int
	txtTokens   int
	inStepEarly bool
	zeroTarget  bool
}

type earlyTKey struct {
	slot   int
	tokens int
}

type earlyRKey struct {
	slot   int
	tokens int
}

type ropeKey struct {
	h, w      int
	refTokens int
	txtTokens int
	refShapes string
}

type countedProgram struct {
	p               *vk.Program
	earlyDispatches int
}

// DiT is FLUX 3's transformer on the Vulkan device.
type DiT struct {
	k  *vk.K2
	tm *tensors.Model

	precision         Precision
	hiddenSize        int
	numHeads          int
	fusedQKV          bool
	flowMap           bool
	depth             int
	depthSingleBlocks int
	arenaBytes        int
	maxImageTokens    int
	maxRefTokens      int

	inStepEarly bool
	zeroTarget  bool

	recordDispatch  func()
	earlyDispatches int
	// timeline, when set, gets a stamp after every dispatch, named for the
	// kernel's family: what the profile tests read.
	timeline *vk.Timeline

	embImg       linearWeight
	embRef       linearWeight
	txtIn        linearWeight
	timeInIn     linearWeight
	timeInOut    linearWeight
	refTimeInIn  linearWeight
	refTimeInOut linearWeight
	earlyModImg  linearWeight
	earlyModRef  linearWeight
	earlyModTxt  linearWeight
	singleModImg linearWeight
	singleModRef linearWeight
	singleModTxt linearWeight
	finalMod     linearWeight
	finalLinear  linearWeight

	imgEarlyBlocks []ditBlock
	refEarlyBlocks []ditBlock
	txtEarlyBlocks []ditBlock
	singleBlocks   []ditBlock

	temb, tembNext, tembZero uint32
	tA, tVec, tVecZero       uint32
	tTarget                  uint32

	modEarlyImg  uint32
	modEarlyRef  uint32
	modEarlyTxt  uint32
	modSingleImg uint32
	modSingleRef uint32
	modSingleTxt uint32
	modFinal     uint32

	ctx       [2]uint32
	txtStates [2]uint32
	refLatent [2]uint32
	refStates [2]uint32
	img       uint32
	embImgBuf uint32
	imgOut    uint32

	x, nh, q, kBuf, vBuf, att, mlp, half, table uint32
	// zeroPar is ditWidth zeros in the parameters: the gated adds' second gate.
	zeroPar uint32

	ctxTokens [2]int
	refTokens [2]int
	refShapes [2][][2]int
	refIDs    [2][]float32

	ropeKey   ropeKey
	ropeTable []float32

	stepProgs   map[stepKey][]countedProgram
	earlyTProgs map[earlyTKey]countedProgram
	earlyRProgs map[earlyRKey]countedProgram
}

type arena struct {
	n int
}

func (a *arena) take(size int) uint32 {
	at := a.n
	a.n += size
	for a.n%64 != 0 {
		a.n++
	}
	return uint32(at)
}

type parBuffer struct {
	data []float32
}

func (p *parBuffer) add(v []float32) uint32 {
	at := len(p.data)
	p.data = append(p.data, v...)
	for len(p.data)%16 != 0 {
		p.data = append(p.data, 0)
	}
	return uint32(at)
}

// parallelChunks partitions [0, n) across GOMAXPROCS workers, calling fn(worker, start, end).
// When GOMAXPROCS is 1 or n < 65536, it runs fn(0, 0, n) inline.
func parallelChunks(n int, fn func(w, s, e int)) {
	numWorkers := runtime.GOMAXPROCS(0)
	if numWorkers <= 1 || n < 65536 {
		fn(0, 0, n)
		return
	}
	chunk := (n + numWorkers - 1) / numWorkers
	var wg sync.WaitGroup
	for w := 0; w < numWorkers; w++ {
		start := w * chunk
		end := min(start+chunk, n)
		if start >= end {
			continue
		}
		wg.Add(1)
		go func(worker, s, e int) {
			defer wg.Done()
			fn(worker, s, e)
		}(w, start, end)
	}
	wg.Wait()
}

// OpenDiT loads the FLUX 3 transformer from a safetensors file onto the Vulkan
// device, with an arena for a 1 MP picture and one 1 MP reference.
func OpenDiT(d *vk.Device, path string, prec ...Precision) (*DiT, error) {
	precision := PrecisionFP16
	if len(prec) > 0 {
		precision = prec[0]
	}
	return OpenDiTSized(d, path, precision, MaxImageTokens, MaxRefTokens)
}

// OpenDiTSized is OpenDiT with an arena for up to maxImageTokens image tokens
// and maxRefTokens reference tokens (all references together).
func OpenDiTSized(d *vk.Device, path string, precision Precision, maxImageTokens, maxRefTokens int) (*DiT, error) {
	if precision != PrecisionFP8 && precision != PrecisionFP16 {
		return nil, fmt.Errorf("flux3: unsupported precision %v", precision)
	}
	if maxImageTokens < 1 || maxRefTokens < 0 {
		return nil, fmt.Errorf("flux3: a DiT for %d image and %d reference tokens", maxImageTokens, maxRefTokens)
	}
	if n := ditArenaFloats(maxImageTokens, maxRefTokens); uint64(n)*4 >= 1<<32 {
		return nil, fmt.Errorf("flux3: a DiT arena for %d image and %d reference tokens is %d bytes, past the 4 GiB one storage buffer holds", maxImageTokens, maxRefTokens, uint64(n)*4)
	}
	tm, err := tensors.Open(path)
	if err != nil {
		return nil, fmt.Errorf("flux3: %s: %w", path, err)
	}

	m := &DiT{
		tm:             tm,
		precision:      precision,
		maxImageTokens: maxImageTokens,
		maxRefTokens:   maxRefTokens,
		stepProgs:      make(map[stepKey][]countedProgram),
		earlyTProgs:    make(map[earlyTKey]countedProgram),
		earlyRProgs:    make(map[earlyRKey]countedProgram),
	}
	fail := func(err error) (*DiT, error) {
		m.Close()
		return nil, err
	}

	// Detect model architecture from keys.
	names := tm.Names()
	keySet := make(map[string]bool, len(names))
	for _, n := range names {
		keySet[n] = true
	}

	probeTensor, err := tm.Get("content_mode_blocks.image.0.attn_out.weight")
	if err != nil {
		return fail(fmt.Errorf("flux3: missing content_mode_blocks.image.0.attn_out.weight: %w", err))
	}
	m.hiddenSize = probeTensor.Shape[0]
	m.numHeads = m.hiddenSize / ditHeadDim
	m.fusedQKV = keySet["single_blocks.0.qkv.weight"]
	m.flowMap = keySet["ref_time_in.in_layer.weight"]

	if m.fusedQKV {
		return fail(fmt.Errorf("flux3: fused_qkv is not supported"))
	}

	// Count depths.
	countBlocks := func(prefix string) int {
		c := 0
		for {
			testKey := fmt.Sprintf("%s%d.attn_out.weight", prefix, c)
			if !keySet[testKey] {
				break
			}
			c++
		}
		return c
	}
	m.depth = countBlocks("txt_mode_blocks.")
	m.depthSingleBlocks = countBlocks("single_blocks.")

	if m.depth <= 0 {
		return fail(fmt.Errorf("flux3: invalid depth %d", m.depth))
	}
	if m.depthSingleBlocks <= 0 {
		return fail(fmt.Errorf("flux3: invalid depth_single_blocks %d", m.depthSingleBlocks))
	}

	var par parBuffer
	m.zeroPar = par.add(make([]float32, ditWidth))
	vec := func(name string, n int) (uint32, error) {
		t, err := tm.Get(name)
		if err != nil {
			return 0, err
		}
		f, err := t.F32()
		if err != nil {
			return 0, err
		}
		madviseDontNeed(t.Raw)
		if len(f) != n {
			return 0, fmt.Errorf("flux3: %s len %d, want %d", name, len(f), n)
		}
		return par.add(f), nil
	}

	var a arena
	m.layout(&a)

	// Pre-load small parameter norms.
	loadBlockNorms := func(prefix string) (qNorm, kNorm uint32, err error) {
		qKey := prefix + "norm.query_norm.scale"
		if !keySet[qKey] {
			qKey = prefix + "norm.query_norm.weight"
		}
		if qNorm, err = vec(qKey, ditHeadDim); err != nil {
			return 0, 0, err
		}
		kKey := prefix + "norm.key_norm.scale"
		if !keySet[kKey] {
			kKey = prefix + "norm.key_norm.weight"
		}
		kNorm, err = vec(kKey, ditHeadDim)
		return qNorm, kNorm, err
	}

	m.imgEarlyBlocks = make([]ditBlock, m.depth)
	m.refEarlyBlocks = make([]ditBlock, m.depth)
	m.txtEarlyBlocks = make([]ditBlock, m.depth)
	m.singleBlocks = make([]ditBlock, m.depthSingleBlocks)

	for i := 0; i < m.depth; i++ {
		if m.imgEarlyBlocks[i].qNorm, m.imgEarlyBlocks[i].kNorm, err = loadBlockNorms(fmt.Sprintf("content_mode_blocks.image.%d.", i)); err != nil {
			return fail(err)
		}
		if m.refEarlyBlocks[i].qNorm, m.refEarlyBlocks[i].kNorm, err = loadBlockNorms(fmt.Sprintf("content_mode_blocks.image_cond.%d.", i)); err != nil {
			return fail(err)
		}
		if m.txtEarlyBlocks[i].qNorm, m.txtEarlyBlocks[i].kNorm, err = loadBlockNorms(fmt.Sprintf("txt_mode_blocks.%d.", i)); err != nil {
			return fail(err)
		}
	}
	for i := 0; i < m.depthSingleBlocks; i++ {
		if m.singleBlocks[i].qNorm, m.singleBlocks[i].kNorm, err = loadBlockNorms(fmt.Sprintf("single_blocks.%d.", i)); err != nil {
			return fail(err)
		}
	}

	if m.k, err = vk.NewK2(d, a.n, par.data); err != nil {
		return fail(err)
	}
	// The attention's fp16 keys and values, beside the arena.
	m.k.ReserveAttention(maxImageTokens+maxRefTokens+MaxTextTokens, ditHeads)
	m.arenaBytes = a.n*4 + m.k.KVBytes()

	var f8Buf []byte
	var f16Buf []byte
	var lut [65536]byte
	lastInvScale := float32(-1)
	up := func(name string, rows, cols int) (linearWeight, error) {
		t, err := tm.Get(name)
		if err != nil {
			return linearWeight{}, err
		}
		if t.Elems() != rows*cols {
			return linearWeight{}, fmt.Errorf("flux3: %s elems %d != %d", name, t.Elems(), rows*cols)
		}

		raw := t.Raw
		n := rows * cols

		if m.precision == PrecisionFP16 {
			if cap(f16Buf) < 2*n {
				f16Buf = make([]byte, 2*n)
			} else {
				f16Buf = f16Buf[:2*n]
			}
			var overflows int64
			parallelChunks(n, func(w, s, e int) {
				var ov int64
				for i := s; i < e; i++ {
					u := binary.LittleEndian.Uint16(raw[i*2:])
					h := bf16ToHalfLUT[u]
					if (h & 0x7c00) == 0x7c00 {
						ov++
					}
					binary.LittleEndian.PutUint16(f16Buf[i*2:], h)
				}
				if ov > 0 {
					atomic.AddInt64(&overflows, ov)
				}
			})
			madviseDontNeed(raw)
			if overflows > 0 {
				return linearWeight{}, fmt.Errorf("flux3: %s had %d bf16 values overflow to inf/nan in fp16", name, overflows)
			}
			h, err := m.k.AddWeights(f16Buf)
			if err != nil {
				return linearWeight{}, err
			}
			return linearWeight{w: h, scale: 1.0, outputs: rows, inputs: cols}, nil
		}

		numWorkers := runtime.GOMAXPROCS(0)
		localMax := make([]float32, max(1, numWorkers))
		parallelChunks(n, func(w, s, e int) {
			var mVal float32
			for i := s; i < e; i++ {
				u := binary.LittleEndian.Uint16(raw[i*2:])
				f := math.Float32frombits(uint32(u) << 16)
				if f < 0 {
					f = -f
				}
				if f > mVal {
					mVal = f
				}
			}
			localMax[w] = mVal
		})
		var maxAbs float32
		for _, mVal := range localMax {
			if mVal > maxAbs {
				maxAbs = mVal
			}
		}
		scale := maxAbs / 448.0
		if maxAbs == 0 {
			scale = 1.0
		}
		invScale := 1.0 / scale

		if invScale != lastInvScale {
			for u := 0; u < 65536; u++ {
				f := math.Float32frombits(uint32(u)<<16) * invScale
				lut[u] = floatToE4M3(f)
			}
			lastInvScale = invScale
		}

		if cap(f8Buf) < n {
			f8Buf = make([]byte, n)
		} else {
			f8Buf = f8Buf[:n]
		}
		parallelChunks(n, func(w, s, e int) {
			for i := s; i < e; i++ {
				u := binary.LittleEndian.Uint16(raw[i*2:])
				f8Buf[i] = lut[u]
			}
		})
		madviseDontNeed(raw)

		h, err := m.k.AddWeights(f8Buf)
		if err != nil {
			return linearWeight{}, err
		}
		return linearWeight{w: h, scale: scale, outputs: rows, inputs: cols}, nil
	}

	loadBlockLinear := func(b *ditBlock, prefix string) (err error) {
		if b.q, err = up(prefix+"q_proj.weight", ditWidth, ditWidth); err != nil {
			return err
		}
		if b.k, err = up(prefix+"k_proj.weight", ditWidth, ditWidth); err != nil {
			return err
		}
		if b.v, err = up(prefix+"v_proj.weight", ditWidth, ditWidth); err != nil {
			return err
		}
		if b.attnOut, err = up(prefix+"attn_out.weight", ditWidth, ditWidth); err != nil {
			return err
		}
		if b.mlpIn, err = up(prefix+"mlp_in.weight", 2*ditFFN, ditWidth); err != nil {
			return err
		}
		b.mlpOut, err = up(prefix+"mlp_out.weight", ditWidth, ditFFN)
		return err
	}

	for i := 0; i < m.depth; i++ {
		if err = loadBlockLinear(&m.imgEarlyBlocks[i], fmt.Sprintf("content_mode_blocks.image.%d.", i)); err != nil {
			return fail(err)
		}
		if err = loadBlockLinear(&m.refEarlyBlocks[i], fmt.Sprintf("content_mode_blocks.image_cond.%d.", i)); err != nil {
			return fail(err)
		}
		if err = loadBlockLinear(&m.txtEarlyBlocks[i], fmt.Sprintf("txt_mode_blocks.%d.", i)); err != nil {
			return fail(err)
		}
	}
	for i := 0; i < m.depthSingleBlocks; i++ {
		if err = loadBlockLinear(&m.singleBlocks[i], fmt.Sprintf("single_blocks.%d.", i)); err != nil {
			return fail(err)
		}
	}

	if m.embImg, err = up("emb_in.image.weight", ditWidth, ditInChannels); err != nil {
		return fail(err)
	}
	if m.embRef, err = up("emb_in.image_cond.weight", ditWidth, ditInChannels); err != nil {
		return fail(err)
	}
	if m.txtIn, err = up("txt_in.weight", ditWidth, ditTxtIn); err != nil {
		return fail(err)
	}
	if m.timeInIn, err = up("time_in.in_layer.weight", ditWidth, ditTDim); err != nil {
		return fail(err)
	}
	if m.timeInOut, err = up("time_in.out_layer.weight", ditWidth, ditWidth); err != nil {
		return fail(err)
	}
	if m.flowMap {
		if m.refTimeInIn, err = up("ref_time_in.in_layer.weight", ditWidth, ditTDim); err != nil {
			return fail(err)
		}
		if m.refTimeInOut, err = up("ref_time_in.out_layer.weight", ditWidth, ditWidth); err != nil {
			return fail(err)
		}
	}
	if m.earlyModImg, err = up("early_stream_modulations.image.lin.weight", 3*ditWidth, ditWidth); err != nil {
		return fail(err)
	}
	if m.earlyModRef, err = up("early_stream_modulations.image_cond.lin.weight", 3*ditWidth, ditWidth); err != nil {
		return fail(err)
	}
	if m.earlyModTxt, err = up("early_stream_modulations.txt.lin.weight", 3*ditWidth, ditWidth); err != nil {
		return fail(err)
	}
	if m.singleModImg, err = up("single_stream_modulations.image.lin.weight", 3*ditWidth, ditWidth); err != nil {
		return fail(err)
	}
	if m.singleModRef, err = up("single_stream_modulations.image_cond.lin.weight", 3*ditWidth, ditWidth); err != nil {
		return fail(err)
	}
	if m.singleModTxt, err = up("single_stream_modulations.txt.lin.weight", 3*ditWidth, ditWidth); err != nil {
		return fail(err)
	}
	if m.finalMod, err = up("final_layer.image.adaLN_modulation.1.weight", 2*ditWidth, ditWidth); err != nil {
		return fail(err)
	}
	if m.finalLinear, err = up("final_layer.image.linear.weight", ditInChannels, ditWidth); err != nil {
		return fail(err)
	}

	f8Buf = nil
	f16Buf = nil
	tm.Close()
	m.tm = nil
	runtime.GC()
	debug.FreeOSMemory()

	return m, nil
}

// ditArenaFloats is the arena a DiT for these limits takes, in floats.
func ditArenaFloats(maxImageTokens, maxRefTokens int) int {
	m := DiT{maxImageTokens: maxImageTokens, maxRefTokens: maxRefTokens}
	var a arena
	m.layout(&a)
	return a.n
}

// layout places the arena's buffers for the DiT's limits. The fp16 ones come
// first, so no index into them in halves passes 2^31; the MLP's are sized for
// mlpChunk tokens, which recordBlock walks.
func (m *DiT) layout(a *arena) {
	N := m.maxImageTokens + m.maxRefTokens + MaxTextTokens
	m.nh = a.take(N * ditWidth / 2)
	m.half = a.take(mlpChunk * ditFFN / 2)

	m.temb = a.take(ditTDim)
	m.tembNext = a.take(ditTDim)
	m.tembZero = a.take(ditTDim)

	m.tA = a.take(ditWidth)
	m.tVec = a.take(ditWidth)
	m.tVecZero = a.take(ditWidth)
	m.tTarget = a.take(ditWidth)

	m.modEarlyImg = a.take(3 * ditWidth)
	m.modEarlyRef = a.take(3 * ditWidth)
	m.modEarlyTxt = a.take(3 * ditWidth)
	m.modSingleImg = a.take(3 * ditWidth)
	m.modSingleRef = a.take(3 * ditWidth)
	m.modSingleTxt = a.take(3 * ditWidth)
	m.modFinal = a.take(2 * ditWidth)

	m.ctx[0] = a.take(MaxTextTokens * ditTxtIn)
	m.ctx[1] = a.take(MaxTextTokens * ditTxtIn)
	m.txtStates[0] = a.take(MaxTextTokens * ditWidth)
	m.txtStates[1] = a.take(MaxTextTokens * ditWidth)

	m.refLatent[0] = a.take(m.maxRefTokens * ditInChannels)
	m.refLatent[1] = a.take(m.maxRefTokens * ditInChannels)
	m.refStates[0] = a.take(m.maxRefTokens * ditWidth)
	m.refStates[1] = a.take(m.maxRefTokens * ditWidth)

	m.img = a.take(m.maxImageTokens * ditInChannels)
	// The image stream's embedding, kept for the waypoint tests up to 1 MP.
	m.embImgBuf = a.take(min(m.maxImageTokens, MaxImageTokens) * ditWidth)
	m.imgOut = a.take(m.maxImageTokens * ditInChannels)

	m.x = a.take(N * ditWidth)
	m.q = a.take(N * ditWidth)
	m.kBuf = a.take(N * ditWidth)
	m.vBuf = a.take(N * ditWidth)
	m.att = a.take(N * ditWidth)
	m.mlp = a.take(mlpChunk * 2 * ditFFN)
	m.table = a.take(N * ditHeadDim)
}

// MaxImageTokens and MaxRefTokens are the limits the DiT was opened for.
func (m *DiT) MaxImageTokens() int { return m.maxImageTokens }
func (m *DiT) MaxRefTokens() int   { return m.maxRefTokens }

// ArenaBytes returns the size in bytes of the Vulkan arena allocated by K2.
func (m *DiT) ArenaBytes() int {
	return m.arenaBytes
}

func (m *DiT) dispatch() {
	if m.recordDispatch != nil {
		m.recordDispatch()
	}
}

func (m *DiT) mm(r *vk.Recorder, w int, mm vk.K2MM) {
	m.dispatch()
	m.k.MM(r, w, m.precision == PrecisionFP8, mm)
	label := "mm"
	switch {
	case mm.Outputs == 2*ditFFN:
		label = "mm mlp in"
	case mm.Inputs == ditFFN:
		label = "mm mlp out"
	}
	m.timeline.Stamp(r, label)
}

func (m *DiT) norm(r *vk.Recorder, n vk.K2Norm) {
	m.dispatch()
	m.k.Norm(r, n)
	m.timeline.Stamp(r, "qknorm")
}

func (m *DiT) flux3Norm(r *vk.Recorder, n vk.K2Flux3Norm) {
	m.dispatch()
	m.k.Flux3Norm(r, n)
	m.timeline.Stamp(r, "norm")
}

func (m *DiT) flux3Gate(r *vk.Recorder, g vk.K2Flux3Gate) {
	m.dispatch()
	m.k.Flux3Gate(r, g)
	m.timeline.Stamp(r, "gate")
}

func (m *DiT) rope(r *vk.Recorder, p vk.K2Rope) {
	m.dispatch()
	m.k.Rope(r, p)
	m.timeline.Stamp(r, "rope")
}

func (m *DiT) act(r *vk.Recorder, a vk.K2Act) {
	m.dispatch()
	m.k.Act(r, a)
	m.timeline.Stamp(r, "act")
}

func (m *DiT) attn(r *vk.Recorder, a vk.K2Attn) {
	m.dispatch()
	m.k.Attn(r, a)
	m.timeline.Stamp(r, "attn")
}

func (m *DiT) forget() {
	for _, progs := range m.stepProgs {
		for _, p := range progs {
			if p.p != nil {
				p.p.Close()
			}
		}
	}
	clear(m.stepProgs)
	for _, p := range m.earlyTProgs {
		if p.p != nil {
			p.p.Close()
		}
	}
	clear(m.earlyTProgs)
	for _, p := range m.earlyRProgs {
		if p.p != nil {
			p.p.Close()
		}
	}
	clear(m.earlyRProgs)
	m.ctxTokens = [2]int{}
	m.refTokens = [2]int{}
	m.refShapes = [2][][2]int{}
	m.refIDs = [2][]float32{}
	m.ropeKey = ropeKey{}
	m.ropeTable = nil
}

func (m *DiT) Close() {
	m.forget()
	if m.k != nil {
		m.k.Close()
		m.k = nil
	}
	if m.tm != nil {
		m.tm.Close()
		m.tm = nil
	}
}

func (m *DiT) Trim() {
	m.forget()
	if m.k != nil {
		m.k.FreeArena()
	}
}

func (m *DiT) WeightBytes() int {
	if m.k != nil {
		return m.k.WeightBytes()
	}
	return 0
}

func timestepEmbedding(t float32) []float32 {
	half := ditTDim / 2
	t1000 := 1000.0 * float64(t)
	out := make([]float32, ditTDim)
	for i := 0; i < half; i++ {
		freq := math.Exp(-math.Log(10000.0) * float64(i) / float64(half))
		arg := t1000 * freq
		out[i] = float32(math.Cos(arg))
		out[half+i] = float32(math.Sin(arg))
	}
	return out
}

func packLatent(x []float32, h, w int) []float32 {
	hw := h * w
	out := make([]float32, hw*ditInChannels)
	for tok := 0; tok < hw; tok++ {
		r := tok / w
		c := tok % w
		for ch := 0; ch < ditInChannels; ch++ {
			out[tok*ditInChannels+ch] = x[ch*hw+r*w+c]
		}
	}
	return out
}

func unpackLatent(x []float32, h, w int) []float32 {
	hw := h * w
	out := make([]float32, hw*ditInChannels)
	for tok := 0; tok < hw; tok++ {
		r := tok / w
		c := tok % w
		for ch := 0; ch < ditInChannels; ch++ {
			out[ch*hw+r*w+c] = x[tok*ditInChannels+ch]
		}
	}
	return out
}

func (m *DiT) SetText(slot int, cond []float32, tokens int) error {
	if slot != 0 && slot != 1 {
		return fmt.Errorf("flux3: slot %d out of range", slot)
	}
	if tokens <= 0 || tokens > MaxTextTokens {
		m.ctxTokens[slot] = 0
		return fmt.Errorf("flux3: %d text tokens out of range [1, %d]", tokens, MaxTextTokens)
	}
	if len(cond) != tokens*ditTxtIn {
		m.ctxTokens[slot] = 0
		return fmt.Errorf("flux3: cond len %d != %d", len(cond), tokens*ditTxtIn)
	}
	if err := m.k.Write(int(m.ctx[slot]), cond); err != nil {
		m.ctxTokens[slot] = 0
		return err
	}
	if !m.flowMap && !m.inStepEarly {
		if err := m.runEarlyText(slot, tokens); err != nil {
			m.ctxTokens[slot] = 0
			return err
		}
	}
	m.ctxTokens[slot] = tokens
	return nil
}

func (m *DiT) SetRefs(slot int, latents [][]float32, shapes [][2]int) error {
	if slot != 0 && slot != 1 {
		return fmt.Errorf("flux3: slot %d out of range", slot)
	}
	if len(latents) != len(shapes) {
		m.refTokens[slot] = 0
		m.refShapes[slot] = nil
		m.refIDs[slot] = nil
		return fmt.Errorf("flux3: len(latents)=%d != len(shapes)=%d", len(latents), len(shapes))
	}
	if len(latents) == 0 {
		m.refTokens[slot] = 0
		m.refShapes[slot] = nil
		m.refIDs[slot] = nil
		return nil
	}

	var packed []float32
	totalRefTokens := 0
	for i, lat := range latents {
		sh := shapes[i]
		expectedLen := sh[0] * sh[1] * ditInChannels
		if len(lat) != expectedLen {
			m.refTokens[slot] = 0
			m.refShapes[slot] = nil
			m.refIDs[slot] = nil
			return fmt.Errorf("flux3: ref %d shape %v len %d != %d", i, sh, len(lat), expectedLen)
		}
		p := packLatent(lat, sh[0], sh[1])
		packed = append(packed, p...)
		totalRefTokens += sh[0] * sh[1]
	}
	if totalRefTokens > m.maxRefTokens {
		m.refTokens[slot] = 0
		m.refShapes[slot] = nil
		m.refIDs[slot] = nil
		return fmt.Errorf("flux3: %d ref tokens exceeds the %d this DiT was opened for", totalRefTokens, m.maxRefTokens)
	}

	refIDs := RefIDs(shapes)

	if err := m.k.Write(int(m.refLatent[slot]), packed); err != nil {
		m.refTokens[slot] = 0
		m.refShapes[slot] = nil
		m.refIDs[slot] = nil
		return err
	}
	if !m.flowMap && !m.inStepEarly {
		if err := m.runEarlyRefs(slot, totalRefTokens, refIDs); err != nil {
			m.refTokens[slot] = 0
			m.refShapes[slot] = nil
			m.refIDs[slot] = nil
			return err
		}
	}
	m.refTokens[slot] = totalRefTokens
	m.refShapes[slot] = shapes
	m.refIDs[slot] = refIDs
	return nil
}

func (m *DiT) recordBlock(r *vk.Recorder, b *ditBlock, tokens int, xOffset uint32, ropeOffset uint32, spans []span) {
	// 1. Modulated LayerNorm for each span into nh (fp16).
	for _, sp := range spans {
		rows := uint32(sp.end - sp.start)
		src := xOffset + uint32(sp.start*ditWidth)
		dst := 2*m.nh + uint32(sp.start*ditWidth)
		scale := sp.modOffset + uint32(ditWidth)
		shift := sp.modOffset
		m.flux3Norm(r, vk.K2Flux3Norm{
			Src: src, SrcStride: uint32(ditWidth),
			Dst: dst, DstStride: uint32(ditWidth),
			N: uint32(ditWidth), Rows: rows,
			Scale: scale, Shift: shift,
			Eps: ditEps, Halves: 1,
		})
	}

	// 2. Linear projections from nh.
	nhHalves := 2 * m.nh
	toks := uint32(tokens)
	m.mm(r, b.q.w, vk.K2MM{
		Outputs: uint32(ditWidth), Inputs: uint32(ditWidth), Cols: toks,
		X: nhHalves, XStride: uint32(ditWidth),
		Y: m.q, YStride: uint32(ditWidth),
		Bias: vk.K2None, Scale: b.q.scale, Mode: vk.K2Store, XHalves: 1,
	})
	m.mm(r, b.k.w, vk.K2MM{
		Outputs: uint32(ditWidth), Inputs: uint32(ditWidth), Cols: toks,
		X: nhHalves, XStride: uint32(ditWidth),
		Y: m.kBuf, YStride: uint32(ditWidth),
		Bias: vk.K2None, Scale: b.k.scale, Mode: vk.K2Store, XHalves: 1,
	})
	m.mm(r, b.v.w, vk.K2MM{
		Outputs: uint32(ditWidth), Inputs: uint32(ditWidth), Cols: toks,
		X: nhHalves, XStride: uint32(ditWidth),
		Y: m.vBuf, YStride: uint32(ditWidth),
		Bias: vk.K2None, Scale: b.v.scale, Mode: vk.K2Store, XHalves: 1,
	})

	// 3. QKNorm.
	m.norm(r, vk.K2Norm{
		Src: m.q, SrcStride: uint32(ditHeadDim),
		Dst: m.q, DstStride: uint32(ditHeadDim),
		N: uint32(ditHeadDim), Rows: toks * uint32(ditHeads),
		Weight: b.qNorm, OnePlus: 0, Eps: ditEps,
		ScaleA: vk.K2None,
	})
	m.norm(r, vk.K2Norm{
		Src: m.kBuf, SrcStride: uint32(ditHeadDim),
		Dst: m.kBuf, DstStride: uint32(ditHeadDim),
		N: uint32(ditHeadDim), Rows: toks * uint32(ditHeads),
		Weight: b.kNorm, OnePlus: 0, Eps: ditEps,
		ScaleA: vk.K2None,
	})

	// 4. Rope.
	m.rope(r, vk.K2Rope{
		X: m.q, Stride: uint32(ditWidth),
		Heads: uint32(ditHeads), Dim: uint32(ditHeadDim),
		Tokens: toks, Table: ropeOffset,
	})
	m.rope(r, vk.K2Rope{
		X: m.kBuf, Stride: uint32(ditWidth),
		Heads: uint32(ditHeads), Dim: uint32(ditHeadDim),
		Tokens: toks, Table: ropeOffset,
	})

	// 5. Attention.
	m.attn(r, vk.K2Attn{
		Q: m.q, QStride: uint32(ditWidth),
		K: m.kBuf, KStride: uint32(ditWidth),
		V: m.vBuf, VStride: uint32(ditWidth),
		O: m.att, OStride: uint32(ditWidth),
		Queries: toks, Keys: toks, Group: 1,
		Scale: float32(1.0 / math.Sqrt(ditHeadDim)),
		Heads: ditHeads, Seqs: 1, HeadDim: ditHeadDim,
	})

	// 6-9. Attention Out, then the MLP from nh mlpChunk tokens at a time
	// (MLP In, SwiGLU into half as fp16, MLP Out), each added to x times its
	// span's gate: x += g·(attn + mlp) as two gated adds, with no buffer
	// between. Neither reads x, which the norm above was the last to read.
	for _, sp := range spans {
		gate := sp.modOffset + uint32(2*ditWidth)
		m.mm(r, b.attnOut.w, vk.K2MM{
			Outputs: uint32(ditWidth), Inputs: uint32(ditWidth), Cols: uint32(sp.end - sp.start),
			X: m.att + uint32(sp.start*ditWidth), XStride: uint32(ditWidth),
			Y: xOffset + uint32(sp.start*ditWidth), YStride: uint32(ditWidth),
			Bias: vk.K2None, Scale: b.attnOut.scale, Mode: vk.K2GatedAdd,
			GateA: gate, GateB: m.zeroPar,
		})
		for c0 := sp.start; c0 < sp.end; c0 += mlpChunk {
			cols := uint32(min(mlpChunk, sp.end-c0))
			m.mm(r, b.mlpIn.w, vk.K2MM{
				Outputs: uint32(2 * ditFFN), Inputs: uint32(ditWidth), Cols: cols,
				X: nhHalves + uint32(c0*ditWidth), XStride: uint32(ditWidth),
				Y: m.mlp, YStride: uint32(2 * ditFFN),
				Bias: vk.K2None, Scale: b.mlpIn.scale, Mode: vk.K2Store, XHalves: 1,
			})
			m.act(r, vk.K2Act{
				X: m.mlp, XStride: uint32(2 * ditFFN),
				Y: m.mlp + uint32(ditFFN), YStride: uint32(2 * ditFFN),
				N: uint32(ditFFN), Rows: cols,
				Op:  vk.K2SwiGLU,
				Dst: 2 * m.half, DstStride: uint32(ditFFN), Halves: 1,
			})
			m.mm(r, b.mlpOut.w, vk.K2MM{
				Outputs: uint32(ditWidth), Inputs: uint32(ditFFN), Cols: cols,
				X: 2 * m.half, XStride: uint32(ditFFN),
				Y: xOffset + uint32(c0*ditWidth), YStride: uint32(ditWidth),
				Bias: vk.K2None, Scale: b.mlpOut.scale, Mode: vk.K2GatedAdd, XHalves: 1,
				GateA: gate, GateB: m.zeroPar,
			})
		}
	}
}

func (m *DiT) runEarlyText(slot, tokens int) error {
	key := earlyTKey{slot: slot, tokens: tokens}
	p, ok := m.earlyTProgs[key]
	if !ok {
		var count int
		rawP, err := m.k.Compile(func(r *vk.Recorder) {
			m.recordDispatch = func() { count++ }
			defer func() { m.recordDispatch = nil }()
			T := uint32(tokens)
			// 1. time_in(0) -> tVecZero
			m.mm(r, m.timeInIn.w, vk.K2MM{
				Outputs: uint32(ditWidth), Inputs: uint32(ditTDim), Cols: 1,
				X: m.tembZero, XStride: uint32(ditTDim),
				Y: m.tA, YStride: uint32(ditWidth),
				Bias: vk.K2None, Scale: m.timeInIn.scale, Mode: vk.K2Store,
			})
			m.act(r, vk.K2Act{X: m.tA, XStride: uint32(ditWidth), N: uint32(ditWidth), Rows: 1, Op: vk.K2SiLU})
			m.mm(r, m.timeInOut.w, vk.K2MM{
				Outputs: uint32(ditWidth), Inputs: uint32(ditWidth), Cols: 1,
				X: m.tA, XStride: uint32(ditWidth),
				Y: m.tVecZero, YStride: uint32(ditWidth),
				Bias: vk.K2None, Scale: m.timeInOut.scale, Mode: vk.K2Store,
			})

			// 2. Modulations for text
			m.act(r, vk.K2Act{X: m.tVecZero, XStride: uint32(ditWidth), N: uint32(ditWidth), Rows: 1, Op: vk.K2SiLU})
			m.mm(r, m.earlyModTxt.w, vk.K2MM{
				Outputs: uint32(3 * ditWidth), Inputs: uint32(ditWidth), Cols: 1,
				X: m.tVecZero, XStride: uint32(ditWidth),
				Y: m.modEarlyTxt, YStride: uint32(3 * ditWidth),
				Bias: vk.K2None, Scale: m.earlyModTxt.scale, Mode: vk.K2Store,
			})

			// 3. txt_in
			m.mm(r, m.txtIn.w, vk.K2MM{
				Outputs: uint32(ditWidth), Inputs: uint32(ditTxtIn), Cols: T,
				X: m.ctx[slot], XStride: uint32(ditTxtIn),
				Y: m.txtStates[slot], YStride: uint32(ditWidth),
				Bias: vk.K2None, Scale: m.txtIn.scale, Mode: vk.K2Store,
			})

			// 4. Early text blocks
			// Rope table for text alone at offset 0 of m.table
			spans := []span{{start: 0, end: int(T), modOffset: m.modEarlyTxt}}
			for i := 0; i < m.depth; i++ {
				m.recordBlock(r, &m.txtEarlyBlocks[i], int(T), m.txtStates[slot], m.table, spans)
			}
		})
		if err != nil {
			return err
		}
		p = countedProgram{p: rawP, earlyDispatches: count}
		m.earlyTProgs[key] = p
	}

	// Prepare tembZero and rope table for text
	if err := m.k.Write(int(m.tembZero), timestepEmbedding(0.0)); err != nil {
		return err
	}
	txtIDs := TextIDs(tokens)
	txtRope := RopeTable(txtIDs)
	if err := m.k.Write(int(m.table), txtRope); err != nil {
		return err
	}

	if err := p.p.Run(); err != nil {
		return err
	}
	m.earlyDispatches += p.earlyDispatches
	return nil
}

func (m *DiT) runEarlyRefs(slot, tokens int, refIDs []float32) error {
	key := earlyRKey{slot: slot, tokens: tokens}
	p, ok := m.earlyRProgs[key]
	if !ok {
		var count int
		rawP, err := m.k.Compile(func(r *vk.Recorder) {
			m.recordDispatch = func() { count++ }
			defer func() { m.recordDispatch = nil }()
			R := uint32(tokens)
			// 1. time_in(0) -> tVecZero
			m.mm(r, m.timeInIn.w, vk.K2MM{
				Outputs: uint32(ditWidth), Inputs: uint32(ditTDim), Cols: 1,
				X: m.tembZero, XStride: uint32(ditTDim),
				Y: m.tA, YStride: uint32(ditWidth),
				Bias: vk.K2None, Scale: m.timeInIn.scale, Mode: vk.K2Store,
			})
			m.act(r, vk.K2Act{X: m.tA, XStride: uint32(ditWidth), N: uint32(ditWidth), Rows: 1, Op: vk.K2SiLU})
			m.mm(r, m.timeInOut.w, vk.K2MM{
				Outputs: uint32(ditWidth), Inputs: uint32(ditWidth), Cols: 1,
				X: m.tA, XStride: uint32(ditWidth),
				Y: m.tVecZero, YStride: uint32(ditWidth),
				Bias: vk.K2None, Scale: m.timeInOut.scale, Mode: vk.K2Store,
			})

			// 2. Modulations for ref
			m.act(r, vk.K2Act{X: m.tVecZero, XStride: uint32(ditWidth), N: uint32(ditWidth), Rows: 1, Op: vk.K2SiLU})
			m.mm(r, m.earlyModRef.w, vk.K2MM{
				Outputs: uint32(3 * ditWidth), Inputs: uint32(ditWidth), Cols: 1,
				X: m.tVecZero, XStride: uint32(ditWidth),
				Y: m.modEarlyRef, YStride: uint32(3 * ditWidth),
				Bias: vk.K2None, Scale: m.earlyModRef.scale, Mode: vk.K2Store,
			})

			// 3. emb_in["image_cond"]
			m.mm(r, m.embRef.w, vk.K2MM{
				Outputs: uint32(ditWidth), Inputs: uint32(ditInChannels), Cols: R,
				X: m.refLatent[slot], XStride: uint32(ditInChannels),
				Y: m.refStates[slot], YStride: uint32(ditWidth),
				Bias: vk.K2None, Scale: m.embRef.scale, Mode: vk.K2Store,
			})

			// 4. Early ref blocks
			spans := []span{{start: 0, end: int(R), modOffset: m.modEarlyRef}}
			for i := 0; i < m.depth; i++ {
				m.recordBlock(r, &m.refEarlyBlocks[i], int(R), m.refStates[slot], m.table, spans)
			}
		})
		if err != nil {
			return err
		}
		p = countedProgram{p: rawP, earlyDispatches: count}
		m.earlyRProgs[key] = p
	}

	if err := m.k.Write(int(m.tembZero), timestepEmbedding(0.0)); err != nil {
		return err
	}
	refRope := RopeTable(refIDs)
	if err := m.k.Write(int(m.table), refRope); err != nil {
		return err
	}

	if err := p.p.Run(); err != nil {
		return err
	}
	m.earlyDispatches += p.earlyDispatches
	return nil
}

func singleBlockChunks(depth int) [][2]int {
	var chunks [][2]int
	curStart := 0
	for b := 0; b < depth; b++ {
		if b == 0 || b == 13 || (b-curStart+1) >= 4 || b == depth-1 {
			chunks = append(chunks, [2]int{curStart, b + 1})
			curStart = b + 1
		}
	}
	return chunks
}

func (m *DiT) stepPrograms(slot, h, w, refToks, txtToks int) ([]countedProgram, error) {
	key := stepKey{
		slot:        slot,
		h:           h,
		w:           w,
		refTokens:   refToks,
		txtTokens:   txtToks,
		inStepEarly: m.inStepEarly,
		zeroTarget:  m.zeroTarget,
	}
	if progs, ok := m.stepProgs[key]; ok {
		return progs, nil
	}

	I := h * w
	R := refToks
	T := txtToks
	totalTokens := I + R + T

	spansJoint := []span{
		{start: 0, end: I, modOffset: m.modSingleImg},
	}
	if R > 0 {
		spansJoint = append(spansJoint, span{start: I, end: I + R, modOffset: m.modSingleRef})
	}
	spansJoint = append(spansJoint, span{start: I + R, end: totalTokens, modOffset: m.modSingleTxt})

	var progs []countedProgram

	// Submission 0: Head + early blocks + sequence assembly
	var p0EarlyCount int
	p0, err := m.k.Compile(func(r *vk.Recorder) {
		// 1. Timestep embeddings
		// time_in(temb) -> tVec
		m.mm(r, m.timeInIn.w, vk.K2MM{
			Outputs: uint32(ditWidth), Inputs: uint32(ditTDim), Cols: 1,
			X: m.temb, XStride: uint32(ditTDim),
			Y: m.tA, YStride: uint32(ditWidth),
			Bias: vk.K2None, Scale: m.timeInIn.scale, Mode: vk.K2Store,
		})
		m.act(r, vk.K2Act{X: m.tA, XStride: uint32(ditWidth), N: uint32(ditWidth), Rows: 1, Op: vk.K2SiLU})
		m.mm(r, m.timeInOut.w, vk.K2MM{
			Outputs: uint32(ditWidth), Inputs: uint32(ditWidth), Cols: 1,
			X: m.tA, XStride: uint32(ditWidth),
			Y: m.tVec, YStride: uint32(ditWidth),
			Bias: vk.K2None, Scale: m.timeInOut.scale, Mode: vk.K2Store,
		})

		// If flow_map: ref_time_in(tembNext) -> tTarget, add to tVec
		if m.flowMap && !m.zeroTarget {
			m.mm(r, m.refTimeInIn.w, vk.K2MM{
				Outputs: uint32(ditWidth), Inputs: uint32(ditTDim), Cols: 1,
				X: m.tembNext, XStride: uint32(ditTDim),
				Y: m.tA, YStride: uint32(ditWidth),
				Bias: vk.K2None, Scale: m.refTimeInIn.scale, Mode: vk.K2Store,
			})
			m.act(r, vk.K2Act{X: m.tA, XStride: uint32(ditWidth), N: uint32(ditWidth), Rows: 1, Op: vk.K2SiLU})
			m.mm(r, m.refTimeInOut.w, vk.K2MM{
				Outputs: uint32(ditWidth), Inputs: uint32(ditWidth), Cols: 1,
				X: m.tA, XStride: uint32(ditWidth),
				Y: m.tTarget, YStride: uint32(ditWidth),
				Bias: vk.K2None, Scale: m.refTimeInOut.scale, Mode: vk.K2Store,
			})
			m.act(r, vk.K2Act{X: m.tVec, XStride: uint32(ditWidth), Y: m.tTarget, YStride: uint32(ditWidth), N: uint32(ditWidth), Rows: 1, Op: vk.K2Sum})
		}

		// time_in(tembZero) -> tVecZero
		m.mm(r, m.timeInIn.w, vk.K2MM{
			Outputs: uint32(ditWidth), Inputs: uint32(ditTDim), Cols: 1,
			X: m.tembZero, XStride: uint32(ditTDim),
			Y: m.tA, YStride: uint32(ditWidth),
			Bias: vk.K2None, Scale: m.timeInIn.scale, Mode: vk.K2Store,
		})
		m.act(r, vk.K2Act{X: m.tA, XStride: uint32(ditWidth), N: uint32(ditWidth), Rows: 1, Op: vk.K2SiLU})
		m.mm(r, m.timeInOut.w, vk.K2MM{
			Outputs: uint32(ditWidth), Inputs: uint32(ditWidth), Cols: 1,
			X: m.tA, XStride: uint32(ditWidth),
			Y: m.tVecZero, YStride: uint32(ditWidth),
			Bias: vk.K2None, Scale: m.timeInOut.scale, Mode: vk.K2Store,
		})
		if m.flowMap && !m.zeroTarget {
			m.act(r, vk.K2Act{X: m.tVecZero, XStride: uint32(ditWidth), Y: m.tTarget, YStride: uint32(ditWidth), N: uint32(ditWidth), Rows: 1, Op: vk.K2Sum})
		}

		// 2. Modulations
		// Image modulations from silu(tVec)
		m.act(r, vk.K2Act{X: m.tVec, XStride: uint32(ditWidth), N: uint32(ditWidth), Rows: 1, Op: vk.K2SiLU})
		m.mm(r, m.earlyModImg.w, vk.K2MM{
			Outputs: uint32(3 * ditWidth), Inputs: uint32(ditWidth), Cols: 1,
			X: m.tVec, XStride: uint32(ditWidth),
			Y: m.modEarlyImg, YStride: uint32(3 * ditWidth),
			Bias: vk.K2None, Scale: m.earlyModImg.scale, Mode: vk.K2Store,
		})
		m.mm(r, m.singleModImg.w, vk.K2MM{
			Outputs: uint32(3 * ditWidth), Inputs: uint32(ditWidth), Cols: 1,
			X: m.tVec, XStride: uint32(ditWidth),
			Y: m.modSingleImg, YStride: uint32(3 * ditWidth),
			Bias: vk.K2None, Scale: m.singleModImg.scale, Mode: vk.K2Store,
		})
		m.mm(r, m.finalMod.w, vk.K2MM{
			Outputs: uint32(2 * ditWidth), Inputs: uint32(ditWidth), Cols: 1,
			X: m.tVec, XStride: uint32(ditWidth),
			Y: m.modFinal, YStride: uint32(2 * ditWidth),
			Bias: vk.K2None, Scale: m.finalMod.scale, Mode: vk.K2Store,
		})

		// Zero-time modulations (text and ref) from silu(tVecZero)
		m.act(r, vk.K2Act{X: m.tVecZero, XStride: uint32(ditWidth), N: uint32(ditWidth), Rows: 1, Op: vk.K2SiLU})
		m.mm(r, m.earlyModTxt.w, vk.K2MM{
			Outputs: uint32(3 * ditWidth), Inputs: uint32(ditWidth), Cols: 1,
			X: m.tVecZero, XStride: uint32(ditWidth),
			Y: m.modEarlyTxt, YStride: uint32(3 * ditWidth),
			Bias: vk.K2None, Scale: m.earlyModTxt.scale, Mode: vk.K2Store,
		})
		m.mm(r, m.singleModTxt.w, vk.K2MM{
			Outputs: uint32(3 * ditWidth), Inputs: uint32(ditWidth), Cols: 1,
			X: m.tVecZero, XStride: uint32(ditWidth),
			Y: m.modSingleTxt, YStride: uint32(3 * ditWidth),
			Bias: vk.K2None, Scale: m.singleModTxt.scale, Mode: vk.K2Store,
		})
		if R > 0 {
			m.mm(r, m.earlyModRef.w, vk.K2MM{
				Outputs: uint32(3 * ditWidth), Inputs: uint32(ditWidth), Cols: 1,
				X: m.tVecZero, XStride: uint32(ditWidth),
				Y: m.modEarlyRef, YStride: uint32(3 * ditWidth),
				Bias: vk.K2None, Scale: m.earlyModRef.scale, Mode: vk.K2Store,
			})
			m.mm(r, m.singleModRef.w, vk.K2MM{
				Outputs: uint32(3 * ditWidth), Inputs: uint32(ditWidth), Cols: 1,
				X: m.tVecZero, XStride: uint32(ditWidth),
				Y: m.modSingleRef, YStride: uint32(3 * ditWidth),
				Bias: vk.K2None, Scale: m.singleModRef.scale, Mode: vk.K2Store,
			})
		}

		// 3. Early text & ref streams (when flow_map or inStepEarly)
		if m.flowMap || m.inStepEarly {
			m.recordDispatch = func() { p0EarlyCount++ }
			// Early text: txt_in + 5 blocks
			m.mm(r, m.txtIn.w, vk.K2MM{
				Outputs: uint32(ditWidth), Inputs: uint32(ditTxtIn), Cols: uint32(T),
				X: m.ctx[slot], XStride: uint32(ditTxtIn),
				Y: m.txtStates[slot], YStride: uint32(ditWidth),
				Bias: vk.K2None, Scale: m.txtIn.scale, Mode: vk.K2Store,
			})
			spansTxt := []span{{start: 0, end: T, modOffset: m.modEarlyTxt}}
			txtRopeOffset := m.table + uint32((I+R)*ditHeadDim)
			for i := 0; i < m.depth; i++ {
				m.recordBlock(r, &m.txtEarlyBlocks[i], T, m.txtStates[slot], txtRopeOffset, spansTxt)
			}

			// Early ref: emb_in["image_cond"] + 5 blocks
			if R > 0 {
				m.mm(r, m.embRef.w, vk.K2MM{
					Outputs: uint32(ditWidth), Inputs: uint32(ditInChannels), Cols: uint32(R),
					X: m.refLatent[slot], XStride: uint32(ditInChannels),
					Y: m.refStates[slot], YStride: uint32(ditWidth),
					Bias: vk.K2None, Scale: m.embRef.scale, Mode: vk.K2Store,
				})
				spansRef := []span{{start: 0, end: R, modOffset: m.modEarlyRef}}
				refRopeOffset := m.table + uint32(I*ditHeadDim)
				for i := 0; i < m.depth; i++ {
					m.recordBlock(r, &m.refEarlyBlocks[i], R, m.refStates[slot], refRopeOffset, spansRef)
				}
			}
			m.recordDispatch = nil
		}

		// 4. Early image stream: emb_in["image"] + 5 blocks
		m.mm(r, m.embImg.w, vk.K2MM{
			Outputs: uint32(ditWidth), Inputs: uint32(ditInChannels), Cols: uint32(I),
			X: m.img, XStride: uint32(ditInChannels),
			Y: m.x, YStride: uint32(ditWidth),
			Bias: vk.K2None, Scale: m.embImg.scale, Mode: vk.K2Store,
		})
		if I <= MaxImageTokens {
			m.act(r, vk.K2Act{
				X: m.embImgBuf, XStride: uint32(ditWidth),
				Y: m.x, YStride: uint32(ditWidth),
				N: uint32(ditWidth), Rows: uint32(I),
				Op: vk.K2Copy,
			})
		}
		spansImg := []span{{start: 0, end: I, modOffset: m.modEarlyImg}}
		for i := 0; i < m.depth; i++ {
			m.recordBlock(r, &m.imgEarlyBlocks[i], I, m.x, m.table, spansImg)
		}

		// 5. Concatenate joint sequence [img, refs, txt] into m.x
		// Image is already at m.x[0..I*ditWidth]
		if R > 0 {
			m.act(r, vk.K2Act{
				X: m.x + uint32(I*ditWidth), XStride: uint32(ditWidth),
				Y: m.refStates[slot], YStride: uint32(ditWidth),
				N: uint32(ditWidth), Rows: uint32(R),
				Op: vk.K2Copy,
			})
		}
		m.act(r, vk.K2Act{
			X: m.x + uint32((I+R)*ditWidth), XStride: uint32(ditWidth),
			Y: m.txtStates[slot], YStride: uint32(ditWidth),
			N: uint32(ditWidth), Rows: uint32(T),
			Op: vk.K2Copy,
		})
	})
	if err != nil {
		return nil, err
	}
	progs = append(progs, countedProgram{p: p0, earlyDispatches: p0EarlyCount})

	chunks := singleBlockChunks(m.depthSingleBlocks)
	for _, ch := range chunks {
		startBlk, endBlk := ch[0], ch[1]
		p, err := m.k.Compile(func(r *vk.Recorder) {
			for bIdx := startBlk; bIdx < endBlk; bIdx++ {
				m.recordBlock(r, &m.singleBlocks[bIdx], totalTokens, m.x, m.table, spansJoint)
			}
		})
		if err != nil {
			for _, pr := range progs {
				pr.p.Close()
			}
			return nil, err
		}
		progs = append(progs, countedProgram{p: p, earlyDispatches: 0})
	}

	// Tail submission: Final layer on image tokens
	pTail, err := m.k.Compile(func(r *vk.Recorder) {
		m.flux3Norm(r, vk.K2Flux3Norm{
			Src: m.x, SrcStride: uint32(ditWidth),
			Dst: 2 * m.nh, DstStride: uint32(ditWidth),
			N: uint32(ditWidth), Rows: uint32(I),
			Scale: m.modFinal + uint32(ditWidth), Shift: m.modFinal,
			Eps: ditEps, Halves: 1,
		})
		m.mm(r, m.finalLinear.w, vk.K2MM{
			Outputs: uint32(ditInChannels), Inputs: uint32(ditWidth), Cols: uint32(I),
			X: 2 * m.nh, XStride: uint32(ditWidth),
			Y: m.imgOut, YStride: uint32(ditInChannels),
			Bias: vk.K2None, Scale: m.finalLinear.scale, Mode: vk.K2Store, XHalves: 1,
		})
	})
	if err != nil {
		for _, pr := range progs {
			pr.p.Close()
		}
		return nil, err
	}
	progs = append(progs, countedProgram{p: pTail, earlyDispatches: 0})

	m.stepProgs[key] = progs
	return progs, nil
}

func (m *DiT) Step(slot int, latent []float32, h, w int, sigma, next float32) ([]float32, error) {
	out, _, err := m.stepWithWaypoints(slot, latent, h, w, sigma, next, false)
	return out, err
}

func (m *DiT) stepWithWaypoints(slot int, latent []float32, h, w int, sigma, next float32, recordWps bool) ([]float32, map[string][]float32, error) {
	if slot != 0 && slot != 1 {
		return nil, nil, fmt.Errorf("flux3: slot %d out of range", slot)
	}
	if m.ctxTokens[slot] == 0 {
		return nil, nil, fmt.Errorf("flux3: no text in slot %d", slot)
	}
	I := h * w
	if I > m.maxImageTokens {
		return nil, nil, fmt.Errorf("flux3: %d patches exceeds the %d image tokens this DiT was opened for", I, m.maxImageTokens)
	}
	if len(latent) != I*ditInChannels {
		return nil, nil, fmt.Errorf("flux3: latent len %d != %d", len(latent), I*ditInChannels)
	}

	R := m.refTokens[slot]
	T := m.ctxTokens[slot]
	total := I + R + T

	progs, err := m.stepPrograms(slot, h, w, R, T)
	if err != nil {
		return nil, nil, err
	}

	packedImg := packLatent(latent, h, w)
	if err := m.k.Write(int(m.img), packedImg); err != nil {
		return nil, nil, err
	}
	if err := m.k.Write(int(m.temb), timestepEmbedding(sigma)); err != nil {
		return nil, nil, err
	}
	if m.flowMap && !m.zeroTarget {
		if err := m.k.Write(int(m.tembNext), timestepEmbedding(next)); err != nil {
			return nil, nil, err
		}
	}
	if err := m.k.Write(int(m.tembZero), timestepEmbedding(0.0)); err != nil {
		return nil, nil, err
	}

	rKey := ropeKey{
		h: h, w: w,
		refTokens: R,
		txtTokens: T,
		refShapes: fmt.Sprint(m.refShapes[slot]),
	}
	if m.ropeTable == nil || m.ropeKey != rKey {
		imgIDs := ImageIDs(h, w, 0)
		var allIDs []float32
		allIDs = append(allIDs, imgIDs...)
		if R > 0 {
			allIDs = append(allIDs, m.refIDs[slot]...)
		}
		allIDs = append(allIDs, TextIDs(T)...)
		m.ropeKey = rKey
		m.ropeTable = RopeTable(allIDs)
	}
	if err := m.k.Write(int(m.table), m.ropeTable); err != nil {
		return nil, nil, err
	}

	var wps map[string][]float32
	if recordWps {
		wps = make(map[string][]float32)
	}

	// Execution:
	// prog 0: Head + early blocks
	if err := progs[0].p.Run(); err != nil {
		return nil, nil, err
	}
	m.earlyDispatches += progs[0].earlyDispatches

	if recordWps {
		if I <= MaxImageTokens {
			if emb, err := m.k.Read(int(m.embImgBuf), I*ditWidth); err == nil {
				wps["wp.emb_img"] = emb
			}
		}
		if txtE, err := m.k.Read(int(m.txtStates[slot]), T*ditWidth); err == nil {
			wps["wp.txt_early"] = txtE
		}
		if imgE, err := m.k.Read(int(m.x), I*ditWidth); err == nil {
			wps["wp.img_early"] = imgE
		}
		if R > 0 {
			if refE, err := m.k.Read(int(m.refStates[slot]), R*ditWidth); err == nil {
				wps["ref_states"] = refE
			}
		}
	}

	chunks := singleBlockChunks(m.depthSingleBlocks)
	for i, ch := range chunks {
		if err := progs[1+i].p.Run(); err != nil {
			return nil, nil, err
		}
		m.earlyDispatches += progs[1+i].earlyDispatches
		if recordWps {
			if ch[1] == 1 {
				if s0, err := m.k.Read(int(m.x), total*ditWidth); err == nil {
					wps["wp.single_0"] = s0
				}
			}
			if ch[1] == 14 {
				if s13, err := m.k.Read(int(m.x), total*ditWidth); err == nil {
					wps["wp.single_13"] = s13
				}
			}
			if ch[1] == m.depthSingleBlocks {
				if sLast, err := m.k.Read(int(m.x), total*ditWidth); err == nil {
					wps[fmt.Sprintf("wp.single_%d", m.depthSingleBlocks-1)] = sLast
				}
			}
		}
	}

	pTail := progs[len(progs)-1]
	if err := pTail.p.Run(); err != nil {
		return nil, nil, err
	}
	m.earlyDispatches += pTail.earlyDispatches

	rawOut, err := m.k.Read(int(m.imgOut), I*ditInChannels)
	if err != nil {
		return nil, nil, err
	}
	unpacked := unpackLatent(rawOut, h, w)
	return unpacked, wps, nil
}
