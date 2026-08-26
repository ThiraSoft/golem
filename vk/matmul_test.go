package vk

// The tiled product against the CPU kernel it is meant to replace for a batch.
//
// On a real tensor rather than a synthetic one, for the reason vk/q6k_test.go
// gives: the weights carry every scale pattern the quantizer produces, and a
// shader that unpacked one nibble the wrong way round would still agree with a
// random matrix on most rows.

import (
	"math"
	"os"
	"testing"

	"github.com/ThiraSoft/golem/nn"
	"github.com/ThiraSoft/golem/tensors"
)

// aQ4_0 opens one Q4_0 matrix out of whichever checkpoint this machine has.
func aQ4_0(tb testing.TB) (*tensors.GGUF, nn.Matrix) {
	return namedQ4_0(tb, "blk.0.ffn_down.weight")
}

func namedQ4_0(tb testing.TB, tensor string) (*tensors.GGUF, nn.Matrix) {
	tb.Helper()
	for _, spec := range []struct{ env, tensor string }{
		{"GOLEM_MODEL_QWEN_Q4", tensor},
		{"GOLEM_MODEL_12B", tensor},
		{"GOLEM_MODEL_26B", tensor},
	} {
		path := os.Getenv(spec.env)
		if path == "" {
			continue
		}
		g, err := tensors.OpenGGUF(path)
		if err != nil {
			continue
		}
		t, err := g.Get(spec.tensor)
		if err != nil || t.DType != "Q4_0" {
			g.Close()
			continue
		}
		return g, nn.Matrix{Data: t.Raw, Quant: nn.Q4_0, Cols: t.Shape[0], Rows: t.Shape[1]}
	}
	tb.Skip("set GOLEM_MODEL_QWEN_Q4, GOLEM_MODEL_12B or GOLEM_MODEL_26B to run the tiled product tests")
	return nil, nn.Matrix{}
}

// columns is a batch of repeatable hidden states, quantized the way the engine
// does. Each column is different: a kernel that read one column for all of
// them would pass a test where they were alike.
func columnsOf(width, n int) *nn.Batch {
	batch := nn.NewBatch(width, n)
	for c := 0; c < n; c++ {
		for i := range batch.F[c] {
			batch.F[c][i] = float32(math.Sin(float64(i)*0.37+float64(c)*1.7)) * float32(1+(i+c)%17) * 0.11
		}
		batch.QuantizeColumnRange(c, 0, width)
	}
	return batch
}

// oneColumn is that batch's column c on its own, in the layout a device buffer
// wants: nn.Batch interleaves its quantized form by block and then by column,
// so a column of it is contiguous only when there is one.
func oneColumn(b *nn.Batch, c int) *nn.Batch {
	one := nn.NewBatch(b.Width, 1)
	copy(one.F[0], b.F[c])
	one.QuantizeColumnRange(0, 0, b.Width)
	return one
}

func TestMatMulMatchesCPU(t *testing.T) {
	// Every width the tiled product is built at, because above BN columns a
	// workgroup answers a slice of the batch rather than all of it, and a
	// dispatch that forgot to count those slices answers a fraction of the
	// columns and leaves the rest zero.
	for _, columns := range []int{32, 64, 128} {
		t.Run("tiled"+itoa(columns), func(t *testing.T) { matMulMatchesCPU(t, columns, false) })
	}
	// The same widths for the cooperative product, and for the same reason:
	// it has its own BN and its own split, and either one left uncounted in
	// the dispatch answers a fraction of the batch and reads as a speed-up.
	//
	// Five hundred and twelve is in the list because it is the width the
	// prompt pass now carries, and because its geometry is not the geometry
	// of the widths below it: BM and BN both a hundred and twenty-eight,
	// answered by a two-by-two grid of waves.
	for _, columns := range []int{32, 64, 128, 256, 512} {
		t.Run("coopmat"+itoa(columns), func(t *testing.T) { matMulMatchesCPU(t, columns, true) })
	}
}

func matMulMatchesCPU(t *testing.T, cols int, coop bool) {
	g, m := aQ4_0(t)
	defer g.Close()
	d := open(t)
	defer d.Close()

	batch := columnsOf(m.Cols, cols)

	// What the CPU makes of it, one product per column.
	want := make([][]float32, cols)
	for c := range want {
		want[c] = make([]float32, m.Rows)
	}
	m.MatVecBatch(batch, want)

	mm, err := NewMatMul(d, m.Data, m.Rows, m.Cols, cols, coop)
	if err != nil {
		t.Fatal(err)
	}
	defer mm.Close()

	for c := 0; c < cols; c++ {
		if err := mm.SetColumn(c, oneColumn(batch, c)); err != nil {
			t.Fatal(err)
		}
	}
	got := make([][]float32, cols)
	for c := range got {
		got[c] = make([]float32, m.Rows)
	}
	if err := mm.Run(got); err != nil {
		t.Fatal(err)
	}

	// The two sum the same products in different orders — this folds the
	// recentring into each block the way ggml does, where nn carries a second
	// accumulator — so the tolerance is the reference tests' rather than a
	// bit-for-bit one.
	var worst float64
	var at, where int
	var scale float64
	for c := range got {
		for i := range got[c] {
			scale = math.Max(scale, math.Abs(float64(want[c][i])))
			if gap := math.Abs(float64(got[c][i] - want[c][i])); gap > worst {
				worst, at, where = gap, c, i
			}
		}
	}
	if worst > 1e-3*scale {
		t.Fatalf("column %d row %d: %v against the CPU's %v, %g of a peak of %g",
			at, where, got[at][where], want[at][where], worst, scale)
	}
	t.Logf("%d rows by %d columns, worst gap %g of a peak of %g", m.Rows, cols, worst, scale)
}

// TestMatMulCoopByIDMatchesCPU is the cooperative product reading a list of
// columns rather than a range of them.
//
// The trap this guards is not the arithmetic, which TestMatMulMatchesCPU
// already covers: it is the indirection. A pair is a column and a slot packed
// together, the answer belongs in the pair's own row of the output, and the
// activation's scales live at a stride that has the slot in it. Any of the
// three read as a plain column index gives an answer that is the right size,
// the right shape, and wrong.
func TestMatMulCoopByIDMatchesCPU(t *testing.T) {
	d := open(t)
	defer d.Close()
	if !d.Coopmat() {
		t.Skip("no cooperative matrices on this device")
	}

	// Two experts, and a list that sends the odd columns to the second one so
	// that a kernel ignoring the list gets a different answer rather than a
	// lucky one.
	const experts, cols, used = 2, 64, 8
	g, m := aQ4_0(t) // rows by cols, one expert's slab
	defer g.Close()

	pairs := make([]uint32, experts*cols)
	counts := []uint32{0, 0}
	for c := 0; c < cols; c++ {
		e, slot := c%experts, c%used
		pair := uint32(c*used + slot)
		pairs[e*cols+int(counts[e])] = pair
		counts[e]++
	}

	const bn = idProductCoopBN
	plan := []uint32{0}
	for e := 0; e < experts; e++ {
		for c := uint32(0); c < counts[e]; c += bn {
			plan = append(plan, uint32(e), c)
		}
	}
	plan[0] = uint32(len(plan)-1) / 2

	stack := append(append([]byte(nil), m.Data...), m.Data...)

	mm, err := NewMatMulByID(d, stack, m.Rows, m.Cols, experts, cols)
	if err != nil {
		t.Fatal(err)
	}
	defer mm.Close()

	mm.SetCounts(counts)
	mm.SetPairs(pairs)
	mm.SetPlan(plan)

	batch := columnsOf(m.Cols, cols)
	for c := 0; c < cols; c++ {
		slot := c % used
		pair := c*used + slot
		if err := mm.SetIDPairColumn(pair, oneColumn(batch, c), used); err != nil {
			t.Fatal(err)
		}
	}

	want := make([][]float32, cols)
	for c := range want {
		want[c] = make([]float32, m.Rows)
	}
	m.MatVecBatch(batch, want)

	got := make([][]float32, cols*used)
	for p := range got {
		got[p] = make([]float32, m.Rows)
	}
	if err := mm.RunByID(got, used); err != nil {
		t.Fatal(err)
	}

	var worst float64
	var scale float64
	for c := 0; c < cols; c++ {
		slot := c % used
		pair := c*used + slot
		for i := 0; i < m.Rows; i++ {
			scale = math.Max(scale, math.Abs(float64(want[c][i])))
			if gap := math.Abs(float64(got[pair][i] - want[c][i])); gap > worst {
				worst = gap
			}
		}
	}
	if worst > 1e-3*scale {
		t.Fatalf("worst gap %g of peak %g", worst, scale)
	}
	t.Logf("TestMatMulCoopByIDMatchesCPU passed: %d rows by %d columns, worst gap %g of peak %g", m.Rows, cols, worst, scale)
}

// BenchmarkMatMul is the tiled product on one matrix, a pass at a time, with
// nothing else in the submission. What it says is the bandwidth the kernel
// reaches: the matrix is read once whatever the width of the pass, so a pass
// that costs the same at thirty-two columns as at one is a pass that has done
// thirty-two times the work for nothing.
func BenchmarkMatMul(b *testing.B) {
	for _, shape := range []struct {
		name   string
		tensor string
	}{
		{"down", "blk.0.ffn_down.weight"}, // 2560 rows by 9728: forty workgroups
		{"gate", "blk.0.ffn_gate.weight"}, // 9728 rows by 2560: a hundred and fifty-two
	} {
		for _, spec := range []struct {
			name    string
			columns int
			coop    bool
		}{{"8", 8, false}, {"32", 32, false}, {"64", 64, false}, {"128", 128, false}, {"256", 256, false}, {"coop32", 32, true}, {"coop64", 64, true}, {"coop128", 128, true}, {"coop256", 256, true}} {
			columns, coop := spec.columns, spec.coop
			b.Run(shape.name+"/"+spec.name, func(b *testing.B) {
				g, m := namedQ4_0(b, shape.tensor)
				defer g.Close()
				d := open(b)
				defer d.Close()

				mm, err := NewMatMul(d, m.Data, m.Rows, m.Cols, columns, coop)
				if err != nil {
					b.Fatal(err)
				}
				defer mm.Close()

				batch := columnsOf(m.Cols, columns)
				for c := 0; c < columns; c++ {
					if err := mm.SetColumn(c, oneColumn(batch, c)); err != nil {
						b.Fatal(err)
					}
				}
				out := make([][]float32, columns)
				for c := range out {
					out[c] = make([]float32, m.Rows)
				}
				if err := mm.Run(out); err != nil {
					b.Fatal(err)
				}
				// Sixty-four passes to a submission, so that the card is not
				// allowed to rest between them.
				const times = 64
				bytes := m.Rows * rowBytesQ4_0(m.Cols)
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					if err := mm.RunTimes(times); err != nil {
						b.Fatal(err)
					}
				}
				b.StopTimer()
				seconds := b.Elapsed().Seconds() / float64(b.N) / times
				b.ReportMetric(float64(bytes)/seconds/1e9, "GB/s")
				b.ReportMetric(seconds*1e6/float64(columns), "us/column")
			})
		}
	}
}

func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	var d []byte
	for n > 0 {
		d = append([]byte{byte('0' + n%10)}, d...)
		n /= 10
	}
	return string(d)
}

// BenchmarkMatMulCold is the same product against a working set the card
// cannot keep. One Q4_0 matrix of this shape is fourteen megabytes and the
// 9070 XT carries sixty-four of last level cache, so BenchmarkMatMul above
// reads its matrix out of cache from the second pass onwards and reports a
// bandwidth no engine will ever see: a real pass walks half a gigabyte of
// weights and every byte of it is cold. Here the same matrix is uploaded
// copies times over and the passes walk the copies in turn, so that by the
// time one comes round again the cache has long since dropped it.
func BenchmarkMatMulCold(b *testing.B) {
	const copies = 24 // 24 x 14 MiB = 340 MiB, five times the cache
	for _, shape := range []struct{ name, tensor string }{
		{"down", "blk.0.ffn_down.weight"},
		{"gate", "blk.0.ffn_gate.weight"},
	} {
		for _, spec := range []struct {
			name    string
			columns int
			coop    bool
		}{{"32", 32, false}, {"64", 64, false}, {"128", 128, false}, {"256", 256, false}, {"coop64", 64, true}, {"coop256", 256, true}} {
			columns, coop := spec.columns, spec.coop
			b.Run(shape.name+"/"+spec.name, func(b *testing.B) {
				g, m := namedQ4_0(b, shape.tensor)
				defer g.Close()
				d := open(b)
				defer d.Close()

				batch := columnsOf(m.Cols, columns)
				mms := make([]*MatMul, copies)
				for k := range mms {
					mm, err := NewMatMul(d, m.Data, m.Rows, m.Cols, columns, coop)
					if err != nil {
						b.Fatal(err)
					}
					defer mm.Close()
					for c := 0; c < columns; c++ {
						if err := mm.SetColumn(c, oneColumn(batch, c)); err != nil {
							b.Fatal(err)
						}
					}
					mms[k] = mm
				}
				out := make([][]float32, columns)
				for c := range out {
					out[c] = make([]float32, m.Rows)
				}
				if err := mms[0].Run(out); err != nil {
					b.Fatal(err)
				}
				for _, mm := range mms[1:] {
					if err := d.Submit(func(r *Recorder) { mm.upload(r) }); err != nil {
						b.Fatal(err)
					}
				}
				bytes := m.Rows * rowBytesQ4_0(m.Cols)
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					if err := d.Submit(func(r *Recorder) {
						for k, mm := range mms {
							if k > 0 {
								r.Barrier()
							}
							mm.pass(r)
						}
					}); err != nil {
						b.Fatal(err)
					}
				}
				b.StopTimer()
				seconds := b.Elapsed().Seconds() / float64(b.N) / copies
				b.ReportMetric(float64(bytes)/seconds/1e9, "GB/s")
				b.ReportMetric(seconds*1e6/float64(columns), "us/column")
			})
		}
	}
}

// BenchmarkMatMulShape is the tiled product over shapes of a constant size, to
// name the asymmetry shaders/matmul_coop.comp ends on: ffn_down's shape, few
// rows over a long shared dimension, reaches half the microseconds a column of
// ffn_gate's, many rows over a short one, on the same bytes and the same
// multiply count. Nothing in the arithmetic distinguishes them, so what is
// wanted is the curve between them rather than the two ends.
//
// The matrices here are synthetic, which vk/q6k_test.go argues against for a
// parity test and which is fine for a clock: an unpacking bug would be as slow
// as a correct unpacking. Cold, twenty-four copies, as BenchmarkMatMulCold.
func BenchmarkMatMulShape(b *testing.B) {
	const columns = 256
	const copies = 12
	for _, shape := range []struct{ rows, cols int }{
		{1920, 30720}, {3840, 15360}, {7680, 7680}, {15360, 3840}, {30720, 1920},
	} {
		rows, cols := shape.rows, shape.cols
		b.Run(itoa(rows)+"x"+itoa(cols), func(b *testing.B) {
			d := open(b)
			defer d.Close()
			data := make([]byte, rows*rowBytesQ4_0(cols))
			for i := range data {
				data[i] = byte(i*31 + i/17)
			}
			batch := columnsOf(cols, columns)
			mms := make([]*MatMul, copies)
			for k := range mms {
				mm, err := NewMatMul(d, data, rows, cols, columns, false)
				if err != nil {
					b.Fatal(err)
				}
				defer mm.Close()
				for c := 0; c < columns; c++ {
					if err := mm.SetColumn(c, oneColumn(batch, c)); err != nil {
						b.Fatal(err)
					}
				}
				mms[k] = mm
			}
			for _, mm := range mms {
				if err := d.Submit(func(r *Recorder) { mm.upload(r) }); err != nil {
					b.Fatal(err)
				}
			}
			bytes := rows * rowBytesQ4_0(cols)
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				if err := d.Submit(func(r *Recorder) {
					for k, mm := range mms {
						if k > 0 {
							r.Barrier()
						}
						mm.pass(r)
					}
				}); err != nil {
					b.Fatal(err)
				}
			}
			b.StopTimer()
			seconds := b.Elapsed().Seconds() / float64(b.N) / copies
			b.ReportMetric(float64(bytes)/seconds/1e9, "GB/s")
			b.ReportMetric(seconds*1e6/float64(columns), "us/column")
		})
	}
}
