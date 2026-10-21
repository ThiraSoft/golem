package vk

// YuE2's flow matching half on the Krea 2 machine: the two kernels its blocks
// need beside the products and the norms, and the key and value cache its
// attention reads. yue2/ lays the network out.
//
// Every query of the latent block sees the AR prefix and the block itself,
// with no mask: arLen keys the AR half wrote once, then the block's own,
// rewritten at every evaluation. The cache holds both, every layer, in the
// layout shaders/krea2_flash.comp reads with GLOBAL (the keys [head][np][128],
// the values [head][128][np], fp16), so the prefix is uploaded once and the
// attention reads it where it lies: no evaluation copies it.

import (
	_ "embed"
	"fmt"
	"unsafe"
)

//go:generate glslc -O --target-env=vulkan1.1 -fshader-stage=compute shaders/yue2_qkrope.comp -o shaders/yue2_qkrope.spv
//go:generate glslc -O --target-env=vulkan1.1 -fshader-stage=compute shaders/yue2_kvimport.comp -o shaders/yue2_kvimport.spv
//go:generate glslc -O --target-env=vulkan1.1 -fshader-stage=compute shaders/yue2_axpy.comp -o shaders/yue2_axpy.spv

//go:embed shaders/yue2_qkrope.spv
var yue2QKRopeSPIRV []byte

//go:embed shaders/yue2_kvimport.spv
var yue2KVImportSPIRV []byte

//go:embed shaders/yue2_axpy.spv
var yue2AxpySPIRV []byte

// K2Yue2Axpy computes dst = a + s*b, and optionally dst2 = dst.
type K2Yue2Axpy struct {
	Dst, Dst2 uint32
	A, B      uint32
	N         uint32
	S         float32
	HasDst2   uint32
}

func (k *K2) Yue2Axpy(r *Recorder, p K2Yue2Axpy) {
	if p.N == 0 {
		return
	}
	k.dispatch(r, k2Yue2Axpy, -1, int((p.N+255)/256), unsafe.Pointer(&p))
}

// K2Yue2QKRope is the RMS norm of every head of 128 of the queries and the
// keys, times their weights, then the NeoX rotation (i with i + 64) from a
// table of a token's 64 cosines then 64 sines, in place. See
// shaders/yue2_qkrope.comp.
type K2Yue2QKRope struct {
	Q, K, Stride           uint32
	QHeads, KHeads, Tokens uint32
	Table                  uint32
	QWeight, KWeight       uint32
	Eps                    float32
}

func (k *K2) Yue2QKRope(r *Recorder, q K2Yue2QKRope) {
	k.dispatch(r, k2Yue2QKRope, -1, int((q.Tokens*(q.QHeads+q.KHeads)+3)/4), unsafe.Pointer(&q))
}

// K2Yue2Cache is every layer's keys and values in fp16, a buffer of its own
// beside the arena: rows of np keys a head, the AR prefix first, then the
// latent block's.
type K2Yue2Cache struct {
	k     *K2
	buf   *Buffer
	np    uint32
	heads uint32 // the keys' and values' heads

	// stackPipe copies the AR half's stack cache in; see yue2_ar.go.
	stackPipe *Pipeline
}

// The cache's sets are kept among the machine's, under a weight handle no
// tensor has, so that FreeArena and Close drop them with the others.
const k2Yue2CacheHandle = -2

// NewYue2Cache makes a cache of layers layers of heads heads of 128 and up to
// keys keys, zeroed: the attention reads up to the next 32 keys past the
// last, and multiplies them by a probability of zero, which a NaN in a
// buffer nobody wrote would turn into a NaN.
func (k *K2) NewYue2Cache(layers, heads, keys int) (*K2Yue2Cache, error) {
	np := kvRows(uint32(keys))
	bytes := uint64(layers) * 2 * uint64(heads) * 128 * uint64(np) * 2
	if bytes/2 >= 1<<32 {
		return nil, fmt.Errorf("vk: a YuE2 cache of %d keys is past what the attention addresses", keys)
	}
	b, err := k.d.Local(int(bytes), bufferUsageStorage|bufferUsageTransferDst)
	if err != nil {
		return nil, err
	}
	if err := k.d.Submit(func(r *Recorder) { r.Fill(b, 0) }); err != nil {
		b.Close()
		return nil, err
	}
	return &K2Yue2Cache{k: k, buf: b, np: np, heads: uint32(heads)}, nil
}

// Rows is np, the keys a head's row of the cache holds.
func (c *K2Yue2Cache) Rows() int { return int(c.np) }

// Bytes is what the cache holds on the card.
func (c *K2Yue2Cache) Bytes() int { return c.buf.Size() }

// Keys and Values are where layer l's keys and values start, in halves.
func (c *K2Yue2Cache) Keys(l int) uint32 { return uint32(l) * 2 * c.heads * 128 * c.np }
func (c *K2Yue2Cache) Values(l int) uint32 {
	return c.Keys(l) + c.heads*128*c.np
}

// WriteLayer puts layer l's keys and values, both in the cache's layout, a
// layer's 2·heads·128·np halves, onto the card.
func (c *K2Yue2Cache) WriteLayer(l int, halves []uint16) error {
	if len(halves) != int(2*c.heads*128*c.np) {
		return fmt.Errorf("vk: %d halves for a YuE2 cache layer of %d", len(halves), 2*c.heads*128*c.np)
	}
	return c.k.d.CopyInto(c.buf, int(c.Keys(l))*2, unsafe.Slice((*byte)(unsafe.Pointer(&halves[0])), 2*len(halves)))
}

func (c *K2Yue2Cache) set(kern k2Kernel) *Set {
	key := k2SetKey{kern, k2Yue2CacheHandle}
	if s, ok := c.k.sets[key]; ok {
		return s
	}
	s, err := c.k.pipes[kern].NewSet([]*Buffer{c.buf, c.k.arena, c.k.params})
	if err != nil {
		panic(fmt.Sprintf("vk: YuE2 cache set for kernel %d: %v", kern, err))
	}
	c.k.sets[key] = s
	return s
}

type k2Yue2KVImportPush struct {
	k, v, stride  uint32
	heads, tokens uint32
	row0, np      uint32
	kh, vt        uint32
}

// Import records the copy of tokens fresh keys and values, rows of stride
// floats at k and v in the arena (all three multiples of four), into layer
// l's rows row0 onward. See shaders/yue2_kvimport.comp.
func (c *K2Yue2Cache) Import(r *Recorder, l int, k, v, stride, tokens, row0 uint32) {
	if row0+tokens > c.np {
		panic(fmt.Sprintf("vk: %d keys behind %d in a YuE2 cache of %d", tokens, row0, c.np))
	}
	if (k|v|stride)%4 != 0 {
		panic(fmt.Sprintf("vk: YuE2 keys at %d and values at %d, stride %d, not read four at a time", k, v, stride))
	}
	p := k2Yue2KVImportPush{k, v, stride, c.heads, tokens, row0, c.np, c.Keys(l), c.Values(l)}
	r.DispatchColumns(c.set(k2Yue2KVImport), (tokens+63)/64, c.heads, unsafe.Pointer(&p))
	r.Barrier()
}

// Attn records layer l's attention over its first keys keys: a's queries
// and output in the arena, the keys and values the cache's. a.K, a.V and
// their strides are the cache's business and are ignored.
func (c *K2Yue2Cache) Attn(r *Recorder, l int, a K2Attn) {
	if a.HeadDim != 128 || a.Seqs != 1 || a.Keys > c.np || uint32(a.Heads) != c.heads*a.Group {
		panic(fmt.Sprintf("vk: YuE2 attention of %d heads of %d over %d keys", a.Heads, a.HeadDim, a.Keys))
	}
	a.K, a.KStride, a.KSeq = c.Keys(l), 128, 0
	a.V, a.VStride, a.VSeq = c.Values(l), c.np, 0
	a.Tiles = (a.Queries + k2FlashRows - 1) / k2FlashRows
	r.DispatchColumns(c.set(k2FlashG128), a.Tiles, uint32(a.Heads), a.push())
	r.Barrier()
}

// Close gives the cache back to the card.
func (c *K2Yue2Cache) Close() {
	for _, kern := range []k2Kernel{k2Yue2KVImport, k2FlashG128} {
		key := k2SetKey{kern, k2Yue2CacheHandle}
		if s, ok := c.k.sets[key]; ok {
			s.Close()
			delete(c.k.sets, key)
		}
	}
	if c.stackPipe != nil {
		c.stackPipe.Close()
		c.stackPipe = nil
	}
	if c.buf != nil {
		c.buf.Close()
		c.buf = nil
	}
}

// AddWeightRows is AddWeights for a tensor of rows rows of rowBytes the
// caller writes itself: fill(dst, first, n) writes rows first .. first+n-1
// into dst, which is a staging buffer on its way to the card. It returns the
// tensor's handle.
//
// It is for weights converted on the way, YuE2's NAR half widened from Q8_0
// to fp16: written into a slice and then uploaded, the conversion and the
// transfer ran one after the other, and the slices, two and a half
// gigabytes of them, were memory the process faulted in once for nothing.
// Here fill writes the next chunk while the card reads the last one, through
// the device's staging pair and the handshake CopyInto uses.
func (k *K2) AddWeightRows(rows, rowBytes int, fill func(dst []byte, first, n int)) (int, error) {
	if rows < 1 || rowBytes < 1 || rowBytes > stageChunk-16 {
		return 0, fmt.Errorf("vk: %d rows of %d bytes", rows, rowBytes)
	}
	d := k.d
	stage, err := d.staging()
	if err != nil {
		return 0, err
	}
	// The products read sixteen bytes at a time, as addWeights pads for.
	size := (rows*rowBytes + 15) / 16 * 16
	dst, err := d.newBuffer(uint64(size), bufferUsageStorage|bufferUsageTransferDst, memoryDeviceLocal)
	if err != nil {
		return 0, err
	}
	// A chunk leaves room for the last one's padding.
	per := (stageChunk - 16) / rowBytes
	free := make(chan int, len(stage))
	for i := range stage {
		free <- i
	}
	type ready struct{ slot, first, bytes int }
	filled := make(chan ready, 1)
	go func() {
		for first := 0; first < rows; first += per {
			slot := <-free
			n := min(per, rows-first)
			bytes := n * rowBytes
			buf := stage[slot].Bytes()
			fill(buf[:bytes], first, n)
			if first+n == rows {
				// The padding is zeros, as addWeights's copy leaves it.
				clear(buf[bytes : size-first*rowBytes])
				bytes = size - first*rowBytes
			}
			filled <- ready{slot, first, bytes}
		}
		close(filled)
	}()
	var failed error
	for r := range filled {
		if failed == nil {
			failed = d.copyBuffer(stage[r.slot], dst, uint64(r.first*rowBytes), uint64(r.bytes))
		}
		free <- r.slot
	}
	if failed != nil {
		dst.Close()
		return 0, failed
	}
	k.weights = append(k.weights, dst)
	return len(k.weights) - 1, nil
}
