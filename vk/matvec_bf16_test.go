package vk

// The bfloat16 product against the float one it is compiled from.
//
// They are the same kernel: shaders/matvec_f32.comp with -DBF16 reads a pair of
// weights to a word and widens each by the shift that puts a bfloat's sixteen
// bits in a float's top half. That shift is exact — a bfloat16 and the float it
// widens to are the same number — so the two answers have to agree to the last
// bit, and this test asserts exactly that rather than a tolerance. A tolerance
// would pass on a kernel that read the wrong half of every word.

import (
	"math"
	"math/rand"
	"testing"
	"unsafe"
)

func TestMatVecBF16MatchesFloat(t *testing.T) {
	d := open(t)
	defer d.Close()

	// A shape with an odd row count and a column count that is not a multiple
	// of the eight lanes, so the tail of a row and the last workgroup are both
	// exercised. The columns stay even: a row of a bfloat16 matrix starts on a
	// word, which every projection in a checkpoint satisfies.
	const rows, cols = 517, 634
	rng := rand.New(rand.NewSource(11))

	// Each weight is a bounded float truncated to its top sixteen bits, and the
	// float side is that truncation widened again. Both kernels then see the
	// same number exactly, which is what makes a bit-for-bit assertion the
	// right one. The bound keeps infinities and NaNs out: a row of those says
	// nothing about whether the two kernels read the same bytes, and a NaN
	// would fail the comparison whether or not they did.
	half := make([]uint16, rows*cols)
	wide := make([]float32, rows*cols)
	for i := range half {
		half[i] = uint16(math.Float32bits(rng.Float32()*4-2) >> 16)
		wide[i] = math.Float32frombits(uint32(half[i]) << 16)
	}
	x := make([]float32, cols)
	for i := range x {
		x[i] = rng.Float32()*2 - 1
	}

	wHalf, err := d.Upload(unsafe.Slice((*byte)(unsafe.Pointer(&half[0])), len(half)*2))
	if err != nil {
		t.Fatal(err)
	}
	defer wHalf.Close()
	wWide, err := d.Upload(asBytes(wide))
	if err != nil {
		t.Fatal(err)
	}
	defer wWide.Close()
	xb, err := d.Upload(asBytes(x))
	if err != nil {
		t.Fatal(err)
	}
	defer xb.Close()

	run := func(spirv []byte, w *Buffer) []float32 {
		t.Helper()
		y, err := d.Readback(rows*4, bufferUsageStorage)
		if err != nil {
			t.Fatal(err)
		}
		defer y.Close()
		pipe, err := d.NewPipeline(spirv, 3, uint32(unsafe.Sizeof(matvecKPush{})))
		if err != nil {
			t.Fatal(err)
		}
		defer pipe.Close()
		set, err := pipe.NewSet([]*Buffer{w, xb, y})
		if err != nil {
			t.Fatal(err)
		}
		push := matvecKPush{Dim: rows, FFN: cols}
		// OUTS is sixteen in the kernel, which is what the group count divides
		// the rows by.
		if err := set.Dispatch(uint32((rows+15)/16), unsafe.Pointer(&push)); err != nil {
			t.Fatal(err)
		}
		return append([]float32(nil), y.Floats()[:rows]...)
	}

	want := run(matvecF32SPIRV, wWide)
	got := run(matvecBF16SPIRV, wHalf)
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("row %d: the bfloat16 kernel answers %v where the float one answers %v",
				i, got[i], want[i])
		}
	}
}

// What the bfloat16 kernel costs against the float one, at the same shape.
//
// TestMatVecBF16MatchesFloat above says the two answer the same thing. It says
// nothing about what they cost, and that turned out to be the question:
// qwen35's streamed pass sends half as many bytes in bfloat16 and finishes
// slower — 3m05.8s against 2m26.3s on the 27B — so the kernel gives back more
// than the bus saves.
//
// The suspicion this measures is the inner loop. The float kernel's eight lanes
// read eight consecutive floats. The bfloat16 kernel's eight lanes read
// `w[(rowBase + i) >> 1]`, so lanes nought and one want the same word, two and
// three the next, and so on: eight loads for four words, and a select on top of
// each. Half the bytes, but not half the work, and the weights are in device
// memory during a pass — where there are five hundred gigabytes a second and
// nothing to be saved by reading fewer of them.
func BenchmarkMatVecBF16AgainstFloat(b *testing.B) {
	d := open(b)
	defer d.Close()

	// A block of the 27B's feed forward: 17408 rows of 5120.
	const rows, cols = 17408, 5120
	rng := rand.New(rand.NewSource(3))
	half := make([]uint16, rows*cols)
	wide := make([]float32, rows*cols)
	for i := range half {
		half[i] = uint16(math.Float32bits(rng.Float32()*4-2) >> 16)
		wide[i] = math.Float32frombits(uint32(half[i]) << 16)
	}
	x := make([]float32, cols)
	for i := range x {
		x[i] = rng.Float32()*2 - 1
	}
	xb, err := d.Upload(asBytes(x))
	if err != nil {
		b.Fatal(err)
	}
	defer xb.Close()
	y, err := d.Readback(rows*4, bufferUsageStorage)
	if err != nil {
		b.Fatal(err)
	}
	defer y.Close()

	run := func(b *testing.B, spirv []byte, data []byte) {
		w, err := d.Upload(data)
		if err != nil {
			b.Fatal(err)
		}
		defer w.Close()
		pipe, err := d.NewPipeline(spirv, 3, uint32(unsafe.Sizeof(matvecKPush{})))
		if err != nil {
			b.Fatal(err)
		}
		defer pipe.Close()
		set, err := pipe.NewSet([]*Buffer{w, xb, y})
		if err != nil {
			b.Fatal(err)
		}
		push := matvecKPush{Dim: rows, FFN: cols}
		groups := uint32((rows + 15) / 16)
		const times = 32
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			if err := set.DispatchTimes(groups, unsafe.Pointer(&push), times); err != nil {
				b.Fatal(err)
			}
		}
		b.StopTimer()
		seconds := b.Elapsed().Seconds() / float64(b.N) / times
		b.ReportMetric(seconds*1e6, "us")
		b.ReportMetric(float64(len(data))/seconds/1e9, "GB/s")
		b.ReportMetric(float64(rows)*float64(cols)/seconds/1e9, "Gweight/s")
	}

	b.Run("f32", func(b *testing.B) { run(b, matvecF32SPIRV, asBytes(wide)) })
	b.Run("bf16", func(b *testing.B) {
		run(b, matvecBF16SPIRV, unsafe.Slice((*byte)(unsafe.Pointer(&half[0])), len(half)*2))
	})
}
