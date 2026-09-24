package vk

// The split attention against the whole one it replaces for a narrow pass,
// and both against the processor.
//
// The two kernels add the same products in a different order — the split one
// sums a slice's tiles, then the slices — so they agree to a tolerance and not
// to the bit. The processor's answer in float64 is what says which of them is
// wrong if they part.

import (
	"math"
	"math/rand"
	"testing"
	"unsafe"
)

func TestQwenAttnSplitMatchesWhole(t *testing.T) {
	d := open(t)
	defer d.Close()

	const (
		heads, kvHeads, dim = 24, 4, 256
		maxContext, slots   = 4096, 2
		width               = 8
	)
	rng := rand.New(rand.NewSource(7))
	// Multiples of 1/256 below one are exact in a half, so the processor
	// reads the very numbers the cache holds.
	exact := func() float32 { return float32(rng.Intn(512)-256) / 256 }

	cache := func() ([]float32, *Buffer) {
		f := make([]float32, slots*kvHeads*maxContext*dim)
		h := make([]uint16, len(f))
		for i := range f {
			f[i] = exact()
			h[i] = halfOf(f[i])
		}
		b, err := d.Upload(unsafe.Slice((*byte)(unsafe.Pointer(&h[0])), len(h)*2))
		if err != nil {
			t.Fatal(err)
		}
		return f, b
	}
	kf, kb := cache()
	defer kb.Close()
	vf, vb := cache()
	defer vb.Close()

	// Queries four times the keys' size, so that some rows of scores are
	// sharp and the running maximum moves.
	qOut := make([]float32, width*heads*dim)
	for i := range qOut {
		qOut[i] = 4 * exact()
	}
	qIn := make([]float32, width*heads*2*dim)
	for i := range qIn {
		qIn[i] = 2 * exact()
	}
	qOutB, err := d.Upload(asBytes(qOut))
	if err != nil {
		t.Fatal(err)
	}
	defer qOutB.Close()
	qInB, err := d.Upload(asBytes(qIn))
	if err != nil {
		t.Fatal(err)
	}
	defer qInB.Close()
	posB, err := d.Readback(width*4, bufferUsageStorage)
	if err != nil {
		t.Fatal(err)
	}
	defer posB.Close()
	slotB, err := d.Readback(width*4, bufferUsageStorage)
	if err != nil {
		t.Fatal(err)
	}
	defer slotB.Close()
	outB, err := d.Readback(width*heads*dim*4, bufferUsageStorage)
	if err != nil {
		t.Fatal(err)
	}
	defer outB.Close()
	partB, err := d.Readback(width*heads*qAttnSplits*dim*4, bufferUsageStorage)
	if err != nil {
		t.Fatal(err)
	}
	defer partB.Close()
	statsB, err := d.Readback(width*heads*qAttnSplits*2*4, bufferUsageStorage)
	if err != nil {
		t.Fatal(err)
	}
	defer statsB.Close()

	pipe := func(spirv []byte, binds int, push uintptr) *Pipeline {
		p, err := d.NewPipeline(spirv, binds, uint32(push))
		if err != nil {
			t.Fatal(err)
		}
		return p
	}
	whole := pipe(qwenAttnGQASPIRV, 7, unsafe.Sizeof(attnGQAPush{}))
	defer whole.Close()
	split := pipe(qwenAttnSplitSPIRV, 8, unsafe.Sizeof(attnGQAPush{}))
	defer split.Close()
	merge := pipe(qwenAttnMergeSPIRV, 4, unsafe.Sizeof(attnMergePush{}))
	defer merge.Close()
	setWhole, err := whole.NewSet([]*Buffer{qOutB, kb, vb, qInB, outB, posB, slotB})
	if err != nil {
		t.Fatal(err)
	}
	setSplit, err := split.NewSet([]*Buffer{qOutB, kb, vb, qInB, partB, posB, slotB, statsB})
	if err != nil {
		t.Fatal(err)
	}
	setMerge, err := merge.NewSet([]*Buffer{partB, statsB, qInB, outB})
	if err != nil {
		t.Fatal(err)
	}

	// The processor's answer for one column, in float64.
	reference := func(c, pos, slot int) []float64 {
		out := make([]float64, heads*dim)
		scores := make([]float64, pos+1)
		for h := 0; h < heads; h++ {
			kv := h / (heads / kvHeads)
			base := (slot*kvHeads + kv) * maxContext * dim
			top := math.Inf(-1)
			for i := 0; i <= pos; i++ {
				s := 0.0
				for j := 0; j < dim; j++ {
					s += float64(qOut[c*heads*dim+h*dim+j]) * float64(kf[base+i*dim+j])
				}
				scores[i] = s / math.Sqrt(dim)
				top = math.Max(top, scores[i])
			}
			sum := 0.0
			for i := range scores {
				scores[i] = math.Exp(scores[i] - top)
				sum += scores[i]
			}
			for j := 0; j < dim; j++ {
				acc := 0.0
				for i := range scores {
					acc += scores[i] * float64(vf[base+i*dim+j])
				}
				g := float64(qIn[c*heads*2*dim+h*2*dim+dim+j])
				out[h*dim+j] = acc / sum / (1 + math.Exp(-g))
			}
		}
		return out
	}

	type run struct{ first, count, pos, slot int }
	cases := [][]run{
		{{0, 1, 0, 0}},
		{{0, 1, 5, 1}},
		{{0, 1, 31, 0}},
		{{0, 1, 1000, 0}},
		{{0, 1, 4095, 1}},
		{{0, 2, 700, 0}},
		{{0, 3, 17, 1}},
		{{0, 8, 2500, 0}},
		// Two conversations in one pass, one column each, which is what a
		// server generating for two people runs.
		{{0, 1, 3000, 0}, {1, 1, 40, 1}},
	}
	for _, rs := range cases {
		columns := 0
		pos := unsafe.Slice((*uint32)(unsafe.Pointer(&posB.Bytes()[0])), width)
		slotv := unsafe.Slice((*uint32)(unsafe.Pointer(&slotB.Bytes()[0])), width)
		for _, r := range rs {
			for i := 0; i < r.count; i++ {
				pos[r.first+i] = uint32(r.pos + i)
				slotv[r.first+i] = uint32(r.slot)
			}
			columns += r.count
		}
		push := func(r run, splits int) attnGQAPush {
			return attnGQAPush{
				MaxContext: maxContext, HeadsPerKV: heads / kvHeads, Heads: heads,
				Scale: float32(1 / math.Sqrt(dim)), Columns: uint32(r.count), First: uint32(r.first),
				Splits: uint32(splits),
			}
		}
		runWith := func(record func(*Recorder)) []float32 {
			t.Helper()
			prog, err := d.Compile(record)
			if err != nil {
				t.Fatal(err)
			}
			defer prog.Close()
			if err := prog.Run(); err != nil {
				t.Fatal(err)
			}
			return append([]float32(nil), outB.Floats()[:columns*heads*dim]...)
		}
		want := runWith(func(r *Recorder) {
			for _, ru := range rs {
				p := push(ru, 0)
				r.DispatchColumns(setWhole, heads, uint32((ru.count+qAttnTile-1)/qAttnTile), unsafe.Pointer(&p))
			}
		})
		got := runWith(func(r *Recorder) {
			for _, ru := range rs {
				p := push(ru, qAttnSplits)
				tiles := (ru.count + qAttnSplitTile - 1) / qAttnSplitTile
				r.DispatchColumns(setSplit, heads, uint32(tiles*qAttnSplits), unsafe.Pointer(&p))
			}
			r.Barrier()
			m := attnMergePush{Heads: heads, Splits: qAttnSplits}
			r.DispatchColumns(setMerge, heads, uint32(columns), unsafe.Pointer(&m))
		})

		worstPair, worstRef, wholeRef, splitRef := 0.0, 0.0, 0.0, 0.0
		for _, ru := range rs {
			for i := 0; i < ru.count; i++ {
				c := ru.first + i
				ref := reference(c, ru.pos+i, ru.slot)
				for j := range ref {
					w, g := float64(want[c*heads*dim+j]), float64(got[c*heads*dim+j])
					if math.IsNaN(g) {
						t.Fatalf("%v: column %d, element %d is NaN", rs, c, j)
					}
					worstPair = math.Max(worstPair, math.Abs(w-g))
					wholeRef = math.Max(wholeRef, math.Abs(w-ref[j]))
					splitRef = math.Max(splitRef, math.Abs(g-ref[j]))
					worstRef = math.Max(wholeRef, splitRef)
				}
			}
		}
		t.Logf("%v: split against whole %.2g; against float64, whole %.2g and split %.2g", rs, worstPair, wholeRef, splitRef)
		if worstPair > 1e-5 || worstRef > 1e-5 {
			t.Errorf("%v: split against whole %.3g, against float64 %.3g", rs, worstPair, worstRef)
		}
	}
}
