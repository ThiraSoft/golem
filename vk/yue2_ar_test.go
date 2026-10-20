package vk

import (
	"encoding/binary"
	"math"
	"math/rand"
	"testing"
	"unsafe"
)

// TestYue2AttnWideMatchesScores runs yue2_attn_wide.comp and attn_scores.comp
// (its matrix-core binary) on the same pass and holds the Q8_0 forms they
// write, magnitudes, scales and corrections, to bit equality: the fork
// moves which thread mixes what and nothing a row computes. YuE2's shape,
// sixteen query heads over eight key heads of 128, a pass of 446 positions
// behind 2560 (the last pass of the Guerreros prompt), causal, one tile cut
// short, in the second of two slots.
func TestYue2AttnWideMatchesScores(t *testing.T) {
	d := open(t)
	defer d.Close()
	if !d.Coopmat() {
		t.Skip("the wide attention is the matrix cores' form")
	}

	const (
		heads, kvHeads, hd = 16, 8, 128
		capacity           = 4096
		slots              = 2
		first, columns     = 2560, 446
		slot               = 1
	)
	rng := rand.New(rand.NewSource(11))
	cacheHalves := slots * capacity * kvHeads * hd
	cache := func() []byte {
		b := make([]byte, 2*cacheHalves)
		for i := range cacheHalves {
			binary.LittleEndian.PutUint16(b[2*i:], f32ToF16(float32(rng.NormFloat64())))
		}
		return b
	}
	ck, err := d.Upload(cache())
	if err != nil {
		t.Fatal(err)
	}
	defer ck.Close()
	cv, err := d.Upload(cache())
	if err != nil {
		t.Fatal(err)
	}
	defer cv.Close()
	// The queries are fp16-exact, as attn_prepare leaves them.
	q := make([]float32, columns*heads*hd)
	for i := range q {
		q[i] = f16ToF32(f32ToF16(float32(rng.NormFloat64())))
	}
	qh, err := d.Upload(asBytes(q))
	if err != nil {
		t.Fatal(err)
	}
	defer qh.Close()
	where := make([]uint32, 4*512)
	for c := range columns {
		pos := uint32(first + c)
		where[4*c+0], where[4*c+1], where[4*c+2], where[4*c+3] = pos, 0, pos, slot
	}
	wh, err := d.Upload(asBytesUint32(where))
	if err != nil {
		t.Fatal(err)
	}
	defer wh.Close()
	scratch, err := d.Local(4, bufferUsageStorage)
	if err != nil {
		t.Fatal(err)
	}
	defer scratch.Close()

	blocks := heads * hd / 32
	run := func(spirv []byte) ([]uint32, []uint32) {
		t.Helper()
		pipe, err := d.newPipeline(spirv, 7, uint32(unsafe.Sizeof(scorePush{})), coopmatWave, nil)
		if err != nil {
			t.Fatal(err)
		}
		defer pipe.Close()
		aq, err := d.Readback(columns*heads*hd, bufferUsageStorage)
		if err != nil {
			t.Fatal(err)
		}
		defer aq.Close()
		as, err := d.Readback(columns*2*blocks*4, bufferUsageStorage)
		if err != nil {
			t.Fatal(err)
		}
		defer as.Close()
		set, err := pipe.NewSet([]*Buffer{qh, ck, cv, scratch, aq, as, wh})
		if err != nil {
			t.Fatal(err)
		}
		defer set.Close()
		push := scorePush{
			heads: heads, kvHeads: kvHeads, headDim: hd, perKV: heads / kvHeads,
			capacity: capacity, mask: uint32(ringMask(capacity)),
			scale: float32(1 / math.Sqrt(hd)), col0: 0, columns: columns,
		}
		err = d.Submit(func(r *Recorder) {
			r.DispatchColumns(set, heads, (columns+scoreColumns-1)/scoreColumns, unsafe.Pointer(&push))
		})
		if err != nil {
			t.Fatal(err)
		}
		words := func(b *Buffer, n int) []uint32 {
			return append([]uint32(nil), unsafe.Slice((*uint32)(unsafe.Pointer(&b.Bytes()[0])), n)...)
		}
		return words(aq, columns*heads*hd/4), words(as, columns*2*blocks)
	}
	wantQ, wantS := run(attnScoresCoopSPIRV)
	gotQ, gotS := run(yue2AttnWideSPIRV)
	for _, c := range []struct {
		what      string
		got, want []uint32
	}{{"magnitudes", gotQ, wantQ}, {"scales and corrections", gotS, wantS}} {
		written, differ := 0, 0
		for i := range c.want {
			if c.want[i] != 0 {
				written++
			}
			if c.got[i] != c.want[i] {
				if differ < 5 {
					t.Errorf("%s, word %d: %#08x, attn_scores's %#08x", c.what, i, c.got[i], c.want[i])
				}
				differ++
			}
		}
		// A kernel that wrote nothing would match one that wrote nothing.
		if written < len(c.want)/2 {
			t.Fatalf("attn_scores wrote %d of %d words of %s", written, len(c.want), c.what)
		}
		if differ > 0 {
			t.Errorf("%s: %d of %d words differ", c.what, differ, len(c.want))
		}
	}
}
