package vk

import (
	"encoding/binary"
	"math"
	"math/rand"
	"sync"
	"testing"
	"unsafe"
)

// yue2WideCase is YuE2's shape on the card for the wide attention's tests:
// sixteen query heads over eight key heads of 128, a pass of 446 positions
// behind 2560 (the last pass of the Guerreros prompt), causal, one tile cut
// short, in the second of two slots of a random fp16 cache.
type yue2WideCase struct {
	d                  *Device
	q                  []float32 // [column][head][128], fp16-exact
	keys, values       []uint16  // the cache, [slot][position][kv head][128]
	qh, ck, cv, wh, sc *Buffer
}

const (
	yue2WideHeads, yue2WideKV, yue2WideHD = 16, 8, 128
	yue2WideCapacity, yue2WideSlots       = 4096, 2
	yue2WideFirst, yue2WideColumns        = 2560, 446
	yue2WideSlot                          = 1
)

func newYue2WideCase(t *testing.T, d *Device) *yue2WideCase {
	t.Helper()
	c := &yue2WideCase{d: d}
	rng := rand.New(rand.NewSource(11))
	halves := yue2WideSlots * yue2WideCapacity * yue2WideKV * yue2WideHD
	cache := func() ([]uint16, []byte) {
		h := make([]uint16, halves)
		b := make([]byte, 2*halves)
		for i := range h {
			h[i] = f32ToF16(float32(rng.NormFloat64()))
			binary.LittleEndian.PutUint16(b[2*i:], h[i])
		}
		return h, b
	}
	var kb, vb []byte
	c.keys, kb = cache()
	c.values, vb = cache()
	c.q = make([]float32, yue2WideColumns*yue2WideHeads*yue2WideHD)
	for i := range c.q {
		c.q[i] = f16ToF32(f32ToF16(float32(rng.NormFloat64())))
	}
	where := make([]uint32, 4*512)
	for col := range yue2WideColumns {
		pos := uint32(yue2WideFirst + col)
		where[4*col+0], where[4*col+1], where[4*col+2], where[4*col+3] = pos, 0, pos, yue2WideSlot
	}
	var err error
	for _, u := range []struct {
		into **Buffer
		data []byte
	}{{&c.ck, kb}, {&c.cv, vb}, {&c.qh, asBytes(c.q)}, {&c.wh, asBytesUint32(where)}} {
		if *u.into, err = d.Upload(u.data); err != nil {
			t.Fatal(err)
		}
	}
	if c.sc, err = d.Local(4, bufferUsageStorage); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		for _, b := range []*Buffer{c.qh, c.ck, c.cv, c.wh, c.sc} {
			b.Close()
		}
	})
	return c
}

// run dispatches spirv over the pass and returns the Q8_0 form it wrote:
// the magnitudes, four to a word, then the scales and corrections.
func (c *yue2WideCase) run(t *testing.T, spirv []byte) ([]uint32, []uint32) {
	t.Helper()
	d := c.d
	blocks := yue2WideHeads * yue2WideHD / 32
	pipe, err := d.newPipeline(spirv, 7, uint32(unsafe.Sizeof(scorePush{})), coopmatWave, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer pipe.Close()
	aq, err := d.Readback(yue2WideColumns*yue2WideHeads*yue2WideHD, bufferUsageStorage)
	if err != nil {
		t.Fatal(err)
	}
	defer aq.Close()
	as, err := d.Readback(yue2WideColumns*2*blocks*4, bufferUsageStorage)
	if err != nil {
		t.Fatal(err)
	}
	defer as.Close()
	set, err := pipe.NewSet([]*Buffer{c.qh, c.ck, c.cv, c.sc, aq, as, c.wh})
	if err != nil {
		t.Fatal(err)
	}
	defer set.Close()
	push := scorePush{
		heads: yue2WideHeads, kvHeads: yue2WideKV, headDim: yue2WideHD, perKV: yue2WideHeads / yue2WideKV,
		capacity: yue2WideCapacity, mask: uint32(ringMask(yue2WideCapacity)),
		scale: float32(1 / math.Sqrt(yue2WideHD)), col0: 0, columns: yue2WideColumns,
	}
	err = d.Submit(func(r *Recorder) {
		r.DispatchColumns(set, yue2WideHeads, (yue2WideColumns+scoreColumns-1)/scoreColumns, unsafe.Pointer(&push))
	})
	if err != nil {
		t.Fatal(err)
	}
	words := func(b *Buffer, n int) []uint32 {
		return append([]uint32(nil), unsafe.Slice((*uint32)(unsafe.Pointer(&b.Bytes()[0])), n)...)
	}
	return words(aq, yue2WideColumns*yue2WideHeads*yue2WideHD/4), words(as, yue2WideColumns*2*blocks)
}

// dequantize reads the Q8_0 form back as floats, [column][head][128].
func (c *yue2WideCase) dequantize(aq, as []uint32) []float32 {
	n := yue2WideHeads * yue2WideHD
	blocks := n / 32
	bytes := unsafe.Slice((*int8)(unsafe.Pointer(&aq[0])), 4*len(aq))
	scales := unsafe.Slice((*float32)(unsafe.Pointer(&as[0])), len(as))
	out := make([]float32, yue2WideColumns*n)
	for col := range yue2WideColumns {
		for i := range n {
			out[col*n+i] = float32(bytes[col*n+i]) * scales[col*2*blocks+i/32]
		}
	}
	return out
}

// reference is the pass's attention in float64 from the same fp16 operands.
func (c *yue2WideCase) reference() []float64 {
	const n = yue2WideHeads * yue2WideHD
	out := make([]float64, yue2WideColumns*n)
	scale := 1 / math.Sqrt(yue2WideHD)
	var wg sync.WaitGroup
	for col := range yue2WideColumns {
		wg.Add(1)
		go func(col int) {
			defer wg.Done()
			last := yue2WideFirst + col
			s := make([]float64, last+1)
			for h := range yue2WideHeads {
				kv := h / (yue2WideHeads / yue2WideKV)
				q := c.q[col*n+h*yue2WideHD:][:yue2WideHD]
				at := func(cache []uint16, pos int) []uint16 {
					e := ((yue2WideSlot*yue2WideCapacity+pos%yue2WideCapacity)*yue2WideKV + kv) * yue2WideHD
					return cache[e : e+yue2WideHD]
				}
				top := math.Inf(-1)
				for j := range s {
					k := at(c.keys, j)
					dot := 0.0
					for i := range k {
						dot += float64(q[i]) * float64(f16ToF32(k[i]))
					}
					s[j] = dot * scale
					top = math.Max(top, s[j])
				}
				sum := 0.0
				for j := range s {
					s[j] = math.Exp(s[j] - top)
					sum += s[j]
				}
				o := out[col*n+h*yue2WideHD:][:yue2WideHD]
				for j := range s {
					v := at(c.values, j)
					for i := range o {
						o[i] += s[j] / sum * float64(f16ToF32(v[i]))
					}
				}
			}
		}(col)
	}
	wg.Wait()
	return out
}

func relativeRMS(got []float32, want []float64) float64 {
	var e, w float64
	for i := range want {
		d := float64(got[i]) - want[i]
		e += d * d
		w += want[i] * want[i]
	}
	return math.Sqrt(e / w)
}

// TestYue2AttnWide holds yue2_attn_wide.comp (the mix on the matrix cores,
// the probabilities in fp16 as ggml's flash attention takes them) to a
// float64 attention of the same operands, beside attn_scores.comp's
// matrix-core binary, which writes the same Q8_0 form: the Q8_0 rounding is
// the whole of attn_scores's distance, and the wide kernel may land no
// further than two percent past it. Measured: 0.5352% and 0.5355%.
func TestYue2AttnWide(t *testing.T) {
	d := open(t)
	// Closed after the case's buffers, which t.Cleanup frees.
	t.Cleanup(d.Close)
	if !d.Coopmat() {
		t.Skip("the wide attention is the matrix cores' form")
	}
	c := newYue2WideCase(t, d)
	want := c.reference()
	scores := relativeRMS(c.dequantize(c.run(t, attnScoresCoopSPIRV)), want)
	wide := relativeRMS(c.dequantize(c.run(t, yue2AttnWideSPIRV)), want)
	t.Logf("against float64: attn_scores %.4f%%, yue2_attn_wide %.4f%%", 100*scores, 100*wide)
	if !(wide <= 1.02*scores) {
		t.Errorf("yue2_attn_wide is %.4f%% from the exact attention, attn_scores %.4f%%", 100*wide, 100*scores)
	}
}
