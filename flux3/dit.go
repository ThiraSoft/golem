package flux3

// The DiT: FLUX 3 Image Klein's multi-modal transformer (comfy/ldm/flux3/model.py),
// from ComfyUI's bf16 checkpoint converted to fp8 e4m3 with per-tensor scale.

import (
	"encoding/binary"
	"fmt"
	"math"
	"runtime"
	"runtime/debug"
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
	MaxRefTokens   = 2048
	MaxTextTokens  = 512
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

func init() {
	var vals [127]float32
	for b := 0; b < 127; b++ {
		vals[b] = e4m3(byte(b))
	}
	for i := 0; i < 126; i++ {
		e4m3Midpoints[i] = (vals[i] + vals[i+1]) * 0.5
	}
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

// DiT is FLUX 3's transformer on the Vulkan device.
type DiT struct {
	k  *vk.K2
	tm *tensors.Model

	hiddenSize        int
	numHeads          int
	fusedQKV          bool
	flowMap           bool
	depth             int
	depthSingleBlocks int

	EarlyDispatches int

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

	x, nh, q, kBuf, vBuf, att, out, mlp, half, table uint32

	ctxTokens [2]int
	refTokens [2]int
	refShapes [2][][2]int
	refIDs    [2][]float32

	ropeKey   [4]int
	ropeTable []float32

	stepProgs   map[stepKey][]*vk.Program
	earlyTProgs map[int]*vk.Program
	earlyRProgs map[int]*vk.Program
}

type stepKey struct {
	slot      int
	h, w      int
	refTokens int
	txtTokens int
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

// OpenDiT loads the FLUX 3 transformer from a safetensors file onto the Vulkan device.
func OpenDiT(d *vk.Device, path string) (*DiT, error) {
	tm, err := tensors.Open(path)
	if err != nil {
		return nil, fmt.Errorf("flux3: %s: %w", path, err)
	}

	m := &DiT{
		tm:          tm,
		stepProgs:   make(map[stepKey][]*vk.Program),
		earlyTProgs: make(map[int]*vk.Program),
		earlyRProgs: make(map[int]*vk.Program),
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

	if m.depth == 0 {
		m.depth = 5
	}
	if m.depthSingleBlocks == 0 {
		m.depthSingleBlocks = 28
	}

	var par parBuffer
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

	// Setup arena layout.
	var a arena
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

	m.refLatent[0] = a.take(MaxRefTokens * ditInChannels)
	m.refLatent[1] = a.take(MaxRefTokens * ditInChannels)
	m.refStates[0] = a.take(MaxRefTokens * ditWidth)
	m.refStates[1] = a.take(MaxRefTokens * ditWidth)

	m.img = a.take(MaxImageTokens * ditInChannels)
	m.embImgBuf = a.take(MaxImageTokens * ditWidth)
	m.imgOut = a.take(MaxImageTokens * ditInChannels)

	N := MaxImageTokens + MaxRefTokens + MaxTextTokens
	m.x = a.take(N * ditWidth)
	m.nh = a.take(N * ditWidth)
	m.q = a.take(N * ditWidth)
	m.kBuf = a.take(N * ditWidth)
	m.vBuf = a.take(N * ditWidth)
	m.att = a.take(N * ditWidth)
	m.out = a.take(N * ditWidth)
	m.mlp = a.take(N * 2 * ditFFN)
	m.half = a.take(N * ditFFN)
	m.table = a.take(N * ditHeadDim)

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

	var f8Buf []byte
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
		var maxAbs float32
		for i := 0; i < n; i++ {
			u := binary.LittleEndian.Uint16(raw[i*2:])
			f := math.Float32frombits(uint32(u) << 16)
			if f < 0 {
				f = -f
			}
			if f > maxAbs {
				maxAbs = f
			}
		}
		scale := maxAbs / 448.0
		if maxAbs == 0 {
			scale = 1.0
		}
		invScale := 1.0 / scale

		var lut [65536]byte
		for u := 0; u < 65536; u++ {
			f := math.Float32frombits(uint32(u)<<16) * invScale
			lut[u] = floatToE4M3(f)
		}

		if cap(f8Buf) < n {
			f8Buf = make([]byte, n)
		} else {
			f8Buf = f8Buf[:n]
		}
		for i := 0; i < n; i++ {
			u := binary.LittleEndian.Uint16(raw[i*2:])
			f8Buf[i] = lut[u]
		}
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
	tm.Close()
	m.tm = nil
	runtime.GC()
	debug.FreeOSMemory()

	return m, nil
}

func (m *DiT) Close() {
	for _, progs := range m.stepProgs {
		for _, p := range progs {
			if p != nil {
				p.Close()
			}
		}
	}
	for _, p := range m.earlyTProgs {
		if p != nil {
			p.Close()
		}
	}
	for _, p := range m.earlyRProgs {
		if p != nil {
			p.Close()
		}
	}
	m.stepProgs = make(map[stepKey][]*vk.Program)
	m.earlyTProgs = make(map[int]*vk.Program)
	m.earlyRProgs = make(map[int]*vk.Program)
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
	if len(cond) != tokens*ditTxtIn {
		return fmt.Errorf("flux3: cond len %d != %d", len(cond), tokens*ditTxtIn)
	}
	m.ctxTokens[slot] = tokens
	if err := m.k.Write(int(m.ctx[slot]), cond); err != nil {
		return err
	}
	if !m.flowMap {
		if err := m.runEarlyText(slot); err != nil {
			return err
		}
	}
	return nil
}

func (m *DiT) SetRefs(slot int, latents [][]float32, shapes [][2]int) error {
	if slot != 0 && slot != 1 {
		return fmt.Errorf("flux3: slot %d out of range", slot)
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
			return fmt.Errorf("flux3: ref %d shape %v len %d != %d", i, sh, len(lat), expectedLen)
		}
		p := packLatent(lat, sh[0], sh[1])
		packed = append(packed, p...)
		totalRefTokens += sh[0] * sh[1]
	}
	m.refTokens[slot] = totalRefTokens
	m.refShapes[slot] = shapes
	m.refIDs[slot] = RefIDs(shapes)

	if err := m.k.Write(int(m.refLatent[slot]), packed); err != nil {
		return err
	}
	if !m.flowMap {
		if err := m.runEarlyRefs(slot); err != nil {
			return err
		}
	}
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
		m.k.Flux3Norm(r, vk.K2Flux3Norm{
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
	m.k.MM(r, b.q.w, true, vk.K2MM{
		Outputs: uint32(ditWidth), Inputs: uint32(ditWidth), Cols: toks,
		X: nhHalves, XStride: uint32(ditWidth),
		Y: m.q, YStride: uint32(ditWidth),
		Bias: vk.K2None, Scale: b.q.scale, Mode: vk.K2Store, XHalves: 1,
	})
	m.k.MM(r, b.k.w, true, vk.K2MM{
		Outputs: uint32(ditWidth), Inputs: uint32(ditWidth), Cols: toks,
		X: nhHalves, XStride: uint32(ditWidth),
		Y: m.kBuf, YStride: uint32(ditWidth),
		Bias: vk.K2None, Scale: b.k.scale, Mode: vk.K2Store, XHalves: 1,
	})
	m.k.MM(r, b.v.w, true, vk.K2MM{
		Outputs: uint32(ditWidth), Inputs: uint32(ditWidth), Cols: toks,
		X: nhHalves, XStride: uint32(ditWidth),
		Y: m.vBuf, YStride: uint32(ditWidth),
		Bias: vk.K2None, Scale: b.v.scale, Mode: vk.K2Store, XHalves: 1,
	})
	m.k.MM(r, b.mlpIn.w, true, vk.K2MM{
		Outputs: uint32(2 * ditFFN), Inputs: uint32(ditWidth), Cols: toks,
		X: nhHalves, XStride: uint32(ditWidth),
		Y: m.mlp, YStride: uint32(2 * ditFFN),
		Bias: vk.K2None, Scale: b.mlpIn.scale, Mode: vk.K2Store, XHalves: 1,
	})

	// 3. QKNorm.
	m.k.Norm(r, vk.K2Norm{
		Src: m.q, SrcStride: uint32(ditHeadDim),
		Dst: m.q, DstStride: uint32(ditHeadDim),
		N: uint32(ditHeadDim), Rows: toks * uint32(ditHeads),
		Weight: b.qNorm, OnePlus: 0, Eps: ditEps,
		ScaleA: vk.K2None,
	})
	m.k.Norm(r, vk.K2Norm{
		Src: m.kBuf, SrcStride: uint32(ditHeadDim),
		Dst: m.kBuf, DstStride: uint32(ditHeadDim),
		N: uint32(ditHeadDim), Rows: toks * uint32(ditHeads),
		Weight: b.kNorm, OnePlus: 0, Eps: ditEps,
		ScaleA: vk.K2None,
	})

	// 4. Rope.
	m.k.Rope(r, vk.K2Rope{
		X: m.q, Stride: uint32(ditWidth),
		Heads: uint32(ditHeads), Dim: uint32(ditHeadDim),
		Tokens: toks, Table: ropeOffset,
	})
	m.k.Rope(r, vk.K2Rope{
		X: m.kBuf, Stride: uint32(ditWidth),
		Heads: uint32(ditHeads), Dim: uint32(ditHeadDim),
		Tokens: toks, Table: ropeOffset,
	})

	// 5. Attention.
	m.k.Attn(r, vk.K2Attn{
		Q: m.q, QStride: uint32(ditWidth),
		K: m.kBuf, KStride: uint32(ditWidth),
		V: m.vBuf, VStride: uint32(ditWidth),
		O: m.att, OStride: uint32(ditWidth),
		Queries: toks, Keys: toks, Group: 1,
		Scale: float32(1.0 / math.Sqrt(ditHeadDim)),
		Heads: ditHeads, Seqs: 1, HeadDim: ditHeadDim,
	})

	// 6. Attention Out linear (float32 operand -> m.out, Mode: Store).
	m.k.MM(r, b.attnOut.w, true, vk.K2MM{
		Outputs: uint32(ditWidth), Inputs: uint32(ditWidth), Cols: toks,
		X: m.att, XStride: uint32(ditWidth),
		Y: m.out, YStride: uint32(ditWidth),
		Bias: vk.K2None, Scale: b.attnOut.scale, Mode: vk.K2Store, XHalves: 0,
	})

	// 7. SwiGLU activation into half (fp16).
	m.k.Act(r, vk.K2Act{
		X: m.mlp, XStride: uint32(2 * ditFFN),
		Y: m.mlp + uint32(ditFFN), YStride: uint32(2 * ditFFN),
		N: uint32(ditFFN), Rows: toks,
		Op:  vk.K2SwiGLU,
		Dst: 2 * m.half, DstStride: uint32(ditFFN), Halves: 1,
	})

	// 8. MLP Out linear (fp16 operand -> add to m.out).
	m.k.MM(r, b.mlpOut.w, true, vk.K2MM{
		Outputs: uint32(ditWidth), Inputs: uint32(ditFFN), Cols: toks,
		X: 2 * m.half, XStride: uint32(ditFFN),
		Y: m.out, YStride: uint32(ditWidth),
		Bias: vk.K2None, Scale: b.mlpOut.scale, Mode: vk.K2Add, XHalves: 1,
	})

	// 9. Gated residual add for each span: x += out * gate.
	for _, sp := range spans {
		rows := uint32(sp.end - sp.start)
		xDst := xOffset + uint32(sp.start*ditWidth)
		outSrc := m.out + uint32(sp.start*ditWidth)
		gate := sp.modOffset + uint32(2*ditWidth)
		m.k.Flux3Gate(r, vk.K2Flux3Gate{
			X: xDst, XStride: uint32(ditWidth),
			Out: outSrc, OutStride: uint32(ditWidth),
			Gate: gate,
			N:    uint32(ditWidth), Rows: rows,
		})
	}
}

func (m *DiT) runEarlyText(slot int) error {
	m.EarlyDispatches++
	p, ok := m.earlyTProgs[slot]
	if !ok {
		var err error
		p, err = m.k.Compile(func(r *vk.Recorder) {
			T := uint32(m.ctxTokens[slot])
			// 1. time_in(0) -> tVecZero
			m.k.MM(r, m.timeInIn.w, true, vk.K2MM{
				Outputs: uint32(ditWidth), Inputs: uint32(ditTDim), Cols: 1,
				X: m.tembZero, XStride: uint32(ditTDim),
				Y: m.tA, YStride: uint32(ditWidth),
				Bias: vk.K2None, Scale: m.timeInIn.scale, Mode: vk.K2Store,
			})
			m.k.Act(r, vk.K2Act{X: m.tA, XStride: uint32(ditWidth), N: uint32(ditWidth), Rows: 1, Op: vk.K2SiLU})
			m.k.MM(r, m.timeInOut.w, true, vk.K2MM{
				Outputs: uint32(ditWidth), Inputs: uint32(ditWidth), Cols: 1,
				X: m.tA, XStride: uint32(ditWidth),
				Y: m.tVecZero, YStride: uint32(ditWidth),
				Bias: vk.K2None, Scale: m.timeInOut.scale, Mode: vk.K2Store,
			})

			// 2. Modulations for text
			m.k.Act(r, vk.K2Act{X: m.tVecZero, XStride: uint32(ditWidth), N: uint32(ditWidth), Rows: 1, Op: vk.K2SiLU})
			m.k.MM(r, m.earlyModTxt.w, true, vk.K2MM{
				Outputs: uint32(3 * ditWidth), Inputs: uint32(ditWidth), Cols: 1,
				X: m.tVecZero, XStride: uint32(ditWidth),
				Y: m.modEarlyTxt, YStride: uint32(3 * ditWidth),
				Bias: vk.K2None, Scale: m.earlyModTxt.scale, Mode: vk.K2Store,
			})

			// 3. txt_in
			m.k.MM(r, m.txtIn.w, true, vk.K2MM{
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
		m.earlyTProgs[slot] = p
	}

	// Prepare tembZero and rope table for text
	if err := m.k.Write(int(m.tembZero), timestepEmbedding(0.0)); err != nil {
		return err
	}
	txtIDs := TextIDs(m.ctxTokens[slot])
	txtRope := RopeTable(txtIDs)
	if err := m.k.Write(int(m.table), txtRope); err != nil {
		return err
	}

	return p.Run()
}

func (m *DiT) runEarlyRefs(slot int) error {
	m.EarlyDispatches++
	p, ok := m.earlyRProgs[slot]
	if !ok {
		var err error
		p, err = m.k.Compile(func(r *vk.Recorder) {
			R := uint32(m.refTokens[slot])
			// 1. time_in(0) -> tVecZero
			m.k.MM(r, m.timeInIn.w, true, vk.K2MM{
				Outputs: uint32(ditWidth), Inputs: uint32(ditTDim), Cols: 1,
				X: m.tembZero, XStride: uint32(ditTDim),
				Y: m.tA, YStride: uint32(ditWidth),
				Bias: vk.K2None, Scale: m.timeInIn.scale, Mode: vk.K2Store,
			})
			m.k.Act(r, vk.K2Act{X: m.tA, XStride: uint32(ditWidth), N: uint32(ditWidth), Rows: 1, Op: vk.K2SiLU})
			m.k.MM(r, m.timeInOut.w, true, vk.K2MM{
				Outputs: uint32(ditWidth), Inputs: uint32(ditWidth), Cols: 1,
				X: m.tA, XStride: uint32(ditWidth),
				Y: m.tVecZero, YStride: uint32(ditWidth),
				Bias: vk.K2None, Scale: m.timeInOut.scale, Mode: vk.K2Store,
			})

			// 2. Modulations for ref
			m.k.Act(r, vk.K2Act{X: m.tVecZero, XStride: uint32(ditWidth), N: uint32(ditWidth), Rows: 1, Op: vk.K2SiLU})
			m.k.MM(r, m.earlyModRef.w, true, vk.K2MM{
				Outputs: uint32(3 * ditWidth), Inputs: uint32(ditWidth), Cols: 1,
				X: m.tVecZero, XStride: uint32(ditWidth),
				Y: m.modEarlyRef, YStride: uint32(3 * ditWidth),
				Bias: vk.K2None, Scale: m.earlyModRef.scale, Mode: vk.K2Store,
			})

			// 3. emb_in["image_cond"]
			m.k.MM(r, m.embRef.w, true, vk.K2MM{
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
		m.earlyRProgs[slot] = p
	}

	if err := m.k.Write(int(m.tembZero), timestepEmbedding(0.0)); err != nil {
		return err
	}
	refRope := RopeTable(m.refIDs[slot])
	if err := m.k.Write(int(m.table), refRope); err != nil {
		return err
	}

	return p.Run()
}

func (m *DiT) stepPrograms(slot, h, w, refToks, txtToks int) ([]*vk.Program, error) {
	key := stepKey{slot: slot, h: h, w: w, refTokens: refToks, txtTokens: txtToks}
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

	var progs []*vk.Program

	// Submission 0: Head + early blocks + sequence assembly
	p0, err := m.k.Compile(func(r *vk.Recorder) {
		// 1. Timestep embeddings
		// time_in(temb) -> tVec
		m.k.MM(r, m.timeInIn.w, true, vk.K2MM{
			Outputs: uint32(ditWidth), Inputs: uint32(ditTDim), Cols: 1,
			X: m.temb, XStride: uint32(ditTDim),
			Y: m.tA, YStride: uint32(ditWidth),
			Bias: vk.K2None, Scale: m.timeInIn.scale, Mode: vk.K2Store,
		})
		m.k.Act(r, vk.K2Act{X: m.tA, XStride: uint32(ditWidth), N: uint32(ditWidth), Rows: 1, Op: vk.K2SiLU})
		m.k.MM(r, m.timeInOut.w, true, vk.K2MM{
			Outputs: uint32(ditWidth), Inputs: uint32(ditWidth), Cols: 1,
			X: m.tA, XStride: uint32(ditWidth),
			Y: m.tVec, YStride: uint32(ditWidth),
			Bias: vk.K2None, Scale: m.timeInOut.scale, Mode: vk.K2Store,
		})

		// If flow_map: ref_time_in(tembNext) -> tTarget, add to tVec
		if m.flowMap {
			m.k.MM(r, m.refTimeInIn.w, true, vk.K2MM{
				Outputs: uint32(ditWidth), Inputs: uint32(ditTDim), Cols: 1,
				X: m.tembNext, XStride: uint32(ditTDim),
				Y: m.tA, YStride: uint32(ditWidth),
				Bias: vk.K2None, Scale: m.refTimeInIn.scale, Mode: vk.K2Store,
			})
			m.k.Act(r, vk.K2Act{X: m.tA, XStride: uint32(ditWidth), N: uint32(ditWidth), Rows: 1, Op: vk.K2SiLU})
			m.k.MM(r, m.refTimeInOut.w, true, vk.K2MM{
				Outputs: uint32(ditWidth), Inputs: uint32(ditWidth), Cols: 1,
				X: m.tA, XStride: uint32(ditWidth),
				Y: m.tTarget, YStride: uint32(ditWidth),
				Bias: vk.K2None, Scale: m.refTimeInOut.scale, Mode: vk.K2Store,
			})
			m.k.Act(r, vk.K2Act{X: m.tVec, XStride: uint32(ditWidth), Y: m.tTarget, YStride: uint32(ditWidth), N: uint32(ditWidth), Rows: 1, Op: vk.K2Sum})
		}

		// time_in(tembZero) -> tVecZero
		m.k.MM(r, m.timeInIn.w, true, vk.K2MM{
			Outputs: uint32(ditWidth), Inputs: uint32(ditTDim), Cols: 1,
			X: m.tembZero, XStride: uint32(ditTDim),
			Y: m.tA, YStride: uint32(ditWidth),
			Bias: vk.K2None, Scale: m.timeInIn.scale, Mode: vk.K2Store,
		})
		m.k.Act(r, vk.K2Act{X: m.tA, XStride: uint32(ditWidth), N: uint32(ditWidth), Rows: 1, Op: vk.K2SiLU})
		m.k.MM(r, m.timeInOut.w, true, vk.K2MM{
			Outputs: uint32(ditWidth), Inputs: uint32(ditWidth), Cols: 1,
			X: m.tA, XStride: uint32(ditWidth),
			Y: m.tVecZero, YStride: uint32(ditWidth),
			Bias: vk.K2None, Scale: m.timeInOut.scale, Mode: vk.K2Store,
		})
		if m.flowMap {
			m.k.Act(r, vk.K2Act{X: m.tVecZero, XStride: uint32(ditWidth), Y: m.tTarget, YStride: uint32(ditWidth), N: uint32(ditWidth), Rows: 1, Op: vk.K2Sum})
		}

		// 2. Modulations
		// Image modulations from silu(tVec)
		m.k.Act(r, vk.K2Act{X: m.tVec, XStride: uint32(ditWidth), N: uint32(ditWidth), Rows: 1, Op: vk.K2SiLU})
		m.k.MM(r, m.earlyModImg.w, true, vk.K2MM{
			Outputs: uint32(3 * ditWidth), Inputs: uint32(ditWidth), Cols: 1,
			X: m.tVec, XStride: uint32(ditWidth),
			Y: m.modEarlyImg, YStride: uint32(3 * ditWidth),
			Bias: vk.K2None, Scale: m.earlyModImg.scale, Mode: vk.K2Store,
		})
		m.k.MM(r, m.singleModImg.w, true, vk.K2MM{
			Outputs: uint32(3 * ditWidth), Inputs: uint32(ditWidth), Cols: 1,
			X: m.tVec, XStride: uint32(ditWidth),
			Y: m.modSingleImg, YStride: uint32(3 * ditWidth),
			Bias: vk.K2None, Scale: m.singleModImg.scale, Mode: vk.K2Store,
		})
		m.k.MM(r, m.finalMod.w, true, vk.K2MM{
			Outputs: uint32(2 * ditWidth), Inputs: uint32(ditWidth), Cols: 1,
			X: m.tVec, XStride: uint32(ditWidth),
			Y: m.modFinal, YStride: uint32(2 * ditWidth),
			Bias: vk.K2None, Scale: m.finalMod.scale, Mode: vk.K2Store,
		})

		// Zero-time modulations (text and ref) from silu(tVecZero)
		m.k.Act(r, vk.K2Act{X: m.tVecZero, XStride: uint32(ditWidth), N: uint32(ditWidth), Rows: 1, Op: vk.K2SiLU})
		m.k.MM(r, m.earlyModTxt.w, true, vk.K2MM{
			Outputs: uint32(3 * ditWidth), Inputs: uint32(ditWidth), Cols: 1,
			X: m.tVecZero, XStride: uint32(ditWidth),
			Y: m.modEarlyTxt, YStride: uint32(3 * ditWidth),
			Bias: vk.K2None, Scale: m.earlyModTxt.scale, Mode: vk.K2Store,
		})
		m.k.MM(r, m.singleModTxt.w, true, vk.K2MM{
			Outputs: uint32(3 * ditWidth), Inputs: uint32(ditWidth), Cols: 1,
			X: m.tVecZero, XStride: uint32(ditWidth),
			Y: m.modSingleTxt, YStride: uint32(3 * ditWidth),
			Bias: vk.K2None, Scale: m.singleModTxt.scale, Mode: vk.K2Store,
		})
		if R > 0 {
			m.k.MM(r, m.earlyModRef.w, true, vk.K2MM{
				Outputs: uint32(3 * ditWidth), Inputs: uint32(ditWidth), Cols: 1,
				X: m.tVecZero, XStride: uint32(ditWidth),
				Y: m.modEarlyRef, YStride: uint32(3 * ditWidth),
				Bias: vk.K2None, Scale: m.earlyModRef.scale, Mode: vk.K2Store,
			})
			m.k.MM(r, m.singleModRef.w, true, vk.K2MM{
				Outputs: uint32(3 * ditWidth), Inputs: uint32(ditWidth), Cols: 1,
				X: m.tVecZero, XStride: uint32(ditWidth),
				Y: m.modSingleRef, YStride: uint32(3 * ditWidth),
				Bias: vk.K2None, Scale: m.singleModRef.scale, Mode: vk.K2Store,
			})
		}

		// 3. Early text & ref streams (when flow_map)
		if m.flowMap {
			// Early text: txt_in + 5 blocks
			m.k.MM(r, m.txtIn.w, true, vk.K2MM{
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
				m.k.MM(r, m.embRef.w, true, vk.K2MM{
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
		}

		// 4. Early image stream: emb_in["image"] + 5 blocks
		m.k.MM(r, m.embImg.w, true, vk.K2MM{
			Outputs: uint32(ditWidth), Inputs: uint32(ditInChannels), Cols: uint32(I),
			X: m.img, XStride: uint32(ditInChannels),
			Y: m.embImgBuf, YStride: uint32(ditWidth),
			Bias: vk.K2None, Scale: m.embImg.scale, Mode: vk.K2Store,
		})
		m.k.Act(r, vk.K2Act{
			X: m.x, XStride: uint32(ditWidth),
			Y: m.embImgBuf, YStride: uint32(ditWidth),
			N: uint32(ditWidth), Rows: uint32(I),
			Op: vk.K2Copy,
		})
		spansImg := []span{{start: 0, end: I, modOffset: m.modEarlyImg}}
		for i := 0; i < m.depth; i++ {
			m.recordBlock(r, &m.imgEarlyBlocks[i], I, m.x, m.table, spansImg)
		}

		// 5. Concatenate joint sequence [img, refs, txt] into m.x
		// Image is already at m.x[0..I*ditWidth]
		if R > 0 {
			m.k.Act(r, vk.K2Act{
				X: m.x + uint32(I*ditWidth), XStride: uint32(ditWidth),
				Y: m.refStates[slot], YStride: uint32(ditWidth),
				N: uint32(ditWidth), Rows: uint32(R),
				Op: vk.K2Copy,
			})
		}
		m.k.Act(r, vk.K2Act{
			X: m.x + uint32((I+R)*ditWidth), XStride: uint32(ditWidth),
			Y: m.txtStates[slot], YStride: uint32(ditWidth),
			N: uint32(ditWidth), Rows: uint32(T),
			Op: vk.K2Copy,
		})
	})
	if err != nil {
		return nil, err
	}
	progs = append(progs, p0)

	// Single blocks: 28 blocks split across submissions (with boundaries at block 0, 13, 27 for waypoints)
	// Chunk 1: block 0 (waypoint single_0)
	// Chunk 2: blocks 1..4 (4)
	// Chunk 3: blocks 5..8 (4)
	// Chunk 4: blocks 9..12 (4)
	// Chunk 5: block 13 (waypoint single_13)
	// Chunk 6: blocks 14..17 (4)
	// Chunk 7: blocks 18..21 (4)
	// Chunk 8: blocks 22..25 (4)
	// Chunk 9: block 26 (1)
	// Chunk 10: block 27 (waypoint single_27)
	chunks := [][2]int{
		{0, 1},
		{1, 5},
		{5, 9},
		{9, 13},
		{13, 14},
		{14, 18},
		{18, 22},
		{22, 26},
		{26, 27},
		{27, 28},
	}

	for _, ch := range chunks {
		startBlk, endBlk := ch[0], ch[1]
		p, err := m.k.Compile(func(r *vk.Recorder) {
			for bIdx := startBlk; bIdx < endBlk; bIdx++ {
				m.recordBlock(r, &m.singleBlocks[bIdx], totalTokens, m.x, m.table, spansJoint)
			}
		})
		if err != nil {
			for _, pr := range progs {
				pr.Close()
			}
			return nil, err
		}
		progs = append(progs, p)
	}

	// Tail submission: Final layer on image tokens
	pTail, err := m.k.Compile(func(r *vk.Recorder) {
		m.k.Flux3Norm(r, vk.K2Flux3Norm{
			Src: m.x, SrcStride: uint32(ditWidth),
			Dst: 2 * m.nh, DstStride: uint32(ditWidth),
			N: uint32(ditWidth), Rows: uint32(I),
			Scale: m.modFinal + uint32(ditWidth), Shift: m.modFinal,
			Eps: ditEps, Halves: 1,
		})
		m.k.MM(r, m.finalLinear.w, true, vk.K2MM{
			Outputs: uint32(ditInChannels), Inputs: uint32(ditWidth), Cols: uint32(I),
			X: 2 * m.nh, XStride: uint32(ditWidth),
			Y: m.imgOut, YStride: uint32(ditInChannels),
			Bias: vk.K2None, Scale: m.finalLinear.scale, Mode: vk.K2Store, XHalves: 1,
		})
	})
	if err != nil {
		for _, pr := range progs {
			pr.Close()
		}
		return nil, err
	}
	progs = append(progs, pTail)

	m.stepProgs[key] = progs
	return progs, nil
}

func (m *DiT) Step(slot int, latent []float32, h, w int, sigma, next float32) ([]float32, error) {
	out, _, err := m.stepInternal(slot, latent, h, w, sigma, next, false)
	return out, err
}

func (m *DiT) StepWithWaypoints(slot int, latent []float32, h, w int, sigma, next float32) ([]float32, map[string][]float32, error) {
	return m.stepInternal(slot, latent, h, w, sigma, next, true)
}

func (m *DiT) stepInternal(slot int, latent []float32, h, w int, sigma, next float32, recordWps bool) ([]float32, map[string][]float32, error) {
	if slot != 0 && slot != 1 {
		return nil, nil, fmt.Errorf("flux3: slot %d out of range", slot)
	}
	if m.ctxTokens[slot] == 0 {
		return nil, nil, fmt.Errorf("flux3: no text in slot %d", slot)
	}
	I := h * w
	if I > MaxImageTokens {
		return nil, nil, fmt.Errorf("flux3: %d patches exceeds MaxImageTokens", I)
	}
	if len(latent) != I*ditInChannels {
		return nil, nil, fmt.Errorf("flux3: latent len %d != %d", len(latent), I*ditInChannels)
	}

	if m.flowMap {
		m.EarlyDispatches++
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
	if m.flowMap {
		if err := m.k.Write(int(m.tembNext), timestepEmbedding(next)); err != nil {
			return nil, nil, err
		}
	}
	if err := m.k.Write(int(m.tembZero), timestepEmbedding(0.0)); err != nil {
		return nil, nil, err
	}

	ropeK := [4]int{h, w, R, T}
	if m.ropeTable == nil || m.ropeKey != ropeK {
		imgIDs := ImageIDs(h, w, 0)
		var allIDs []float32
		allIDs = append(allIDs, imgIDs...)
		if R > 0 {
			allIDs = append(allIDs, m.refIDs[slot]...)
		}
		allIDs = append(allIDs, TextIDs(T)...)
		m.ropeKey = ropeK
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
	if err := progs[0].Run(); err != nil {
		return nil, nil, err
	}
	if recordWps {
		if emb, err := m.k.Read(int(m.embImgBuf), I*ditWidth); err == nil {
			wps["wp.emb_img"] = emb
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

	// prog 1: block 0
	if err := progs[1].Run(); err != nil {
		return nil, nil, err
	}
	if recordWps {
		if s0, err := m.k.Read(int(m.x), total*ditWidth); err == nil {
			wps["wp.single_0"] = s0
		}
	}

	// progs 2..5: blocks 1..13
	for i := 2; i <= 5; i++ {
		if err := progs[i].Run(); err != nil {
			return nil, nil, err
		}
	}
	if recordWps {
		if s13, err := m.k.Read(int(m.x), total*ditWidth); err == nil {
			wps["wp.single_13"] = s13
		}
	}

	// progs 6..10: blocks 14..27
	for i := 6; i <= 10; i++ {
		if err := progs[i].Run(); err != nil {
			return nil, nil, err
		}
	}
	if recordWps {
		if s27, err := m.k.Read(int(m.x), total*ditWidth); err == nil {
			wps["wp.single_27"] = s27
		}
	}

	// prog 11: tail (final layer)
	if err := progs[11].Run(); err != nil {
		return nil, nil, err
	}

	rawOut, err := m.k.Read(int(m.imgOut), I*ditInChannels)
	if err != nil {
		return nil, nil, err
	}
	unpacked := unpackLatent(rawOut, h, w)
	return unpacked, wps, nil
}
