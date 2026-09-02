package vk

// The mat-vec against llama.cpp's own, shape by shape.
//
// llama.cpp's Vulkan backend logs the time of every operation it runs
// (GGML_VK_PERF_LOGGER=1), so a token of Qwen3-4B-Q4_K_M comes back as a list
// of matrix products with a microsecond count each. This runs the same shapes
// through this repository's kernel so the two can be read side by side, which
// is the only comparison that says whether a gap is in the product or in
// everything around it.
//
// The matrices are rotated rather than reread. One of them is fourteen
// megabytes and this card has sixty-four of last-level cache: a loop over a
// single matrix measures the cache and ranks kernels on a bandwidth generation
// never sees. Eight of them do not fit and the rotation is what a block-by-block
// walk of a model does anyway.

import (
	"math/rand"
	"testing"
	"time"
	"unsafe"

	"github.com/ThiraSoft/golem/nn"
)

func TestMatVecAgainstLlamaCpp(t *testing.T) {
	d, err := Open()
	if err != nil {
		t.Skip(err)
	}
	defer d.Close()

	for _, s := range []struct {
		rows, cols int
		q          nn.Quant
		what       string
		theirs     float64 // microseconds, from their logger on this machine
	}{
		{9728, 2560, nn.Q4_K, "ffn gate and up", 27.63},
		{2560, 9728, nn.Q4_K, "ffn down", 40.44},
		{2560, 9728, nn.Q6_K, "ffn down, six bits", 52.71},
		{4096, 2560, nn.Q4_K, "attn query", 14.56},
		{2560, 4096, nn.Q4_K, "attn output", 17.17},
		{1024, 2560, nn.Q4_K, "attn key and value", 7.69},
	} {
		t.Run(s.what, func(t *testing.T) {
			ours := matvecMicroseconds(t, d, s.rows, s.cols, s.q)
			bytes := float64(s.rows) * float64(s.cols)
			switch s.q {
			case nn.Q4_K:
				bytes *= 18.0 / 32
			case nn.Q6_K:
				bytes *= 210.0 / 256
			}
			t.Logf("%-20s %8.2f us  %6.0f GB/s   llama.cpp %6.2f us  %+5.1f%%",
				s.what, ours, bytes/ours/1e3, s.theirs, 100*(ours-s.theirs)/s.theirs)
		})
	}
}

// matvecMicroseconds is one product's time, averaged over a stream of them that
// no cache can hold.
func matvecMicroseconds(tb testing.TB, d *Device, rows, cols int, q nn.Quant) float64 {
	tb.Helper()
	const copies = 8
	rng := rand.New(rand.NewSource(int64(rows*31 + cols)))

	x := make([]float32, cols)
	for i := range x {
		x[i] = rng.Float32()*2 - 1
	}
	qs, scales, _ := q80Column(x)
	aq, err := d.Upload(unsafe.Slice((*byte)(unsafe.Pointer(&qs[0])), len(qs)*4))
	if err != nil {
		tb.Fatal(err)
	}
	defer aq.Close()
	as, err := d.Upload(asBytes(scales))
	if err != nil {
		tb.Fatal(err)
	}
	defer as.Close()
	y, err := d.Readback(rows*4, bufferUsageStorage)
	if err != nil {
		tb.Fatal(err)
	}
	defer y.Close()

	products := newQuantProducts(d, d.Coopmat())
	defer products.Close()
	pipe, err := products.get(q)
	if err != nil {
		tb.Fatal(err)
	}

	fileRow := 0
	switch q {
	case nn.Q4_K:
		fileRow = cols / nn.SuperBlock * 144
	case nn.Q6_K:
		fileRow = cols / nn.SuperBlock * 210
	}
	sets := make([]*Set, copies)
	for c := 0; c < copies; c++ {
		data := make([]byte, rows*fileRow)
		rng.Read(data)
		sane(tb, nn.Matrix{Data: data, Quant: q, Rows: rows, Cols: cols}, rng)
		layout, err := quantLayout(q, data, rows, cols)
		if err != nil {
			tb.Fatal(err)
		}
		w, err := d.Upload(layout)
		if err != nil {
			tb.Fatal(err)
		}
		defer w.Close()
		if sets[c], err = pipe.NewSet([]*Buffer{w, aq, as, y}); err != nil {
			tb.Fatal(err)
		}
		defer sets[c].Close()
	}

	push := moePush{dim: uint32(rows), ffn: uint32(cols), used: 1, split: 1}
	groups := groupsOf(rows, matvecOuts)
	run := func(n int) time.Duration {
		start := time.Now()
		if err := d.Submit(func(r *Recorder) {
			for i := 0; i < n; i++ {
				r.Dispatch(sets[i%copies], groups, unsafe.Pointer(&push))
				r.Barrier()
			}
		}); err != nil {
			tb.Fatal(err)
		}
		return time.Since(start)
	}
	run(copies * 4)
	// The best of several rounds, not the mean of one. A submission here
	// occasionally takes ten times what it takes the round before — the queue,
	// the clocks, something outside this kernel — and a mean over a stream that
	// contains one of those measures the interruption. What the shape is being
	// asked is how fast it *can* read a matrix.
	const passes = 400
	best := time.Hour
	for round := 0; round < 5; round++ {
		if took := run(passes); took < best {
			best = took
		}
	}
	return float64(best.Microseconds()) / passes
}
