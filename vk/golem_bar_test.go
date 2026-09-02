package vk

// The trellis product against the one it has to beat, interleaved.
//
// Q4_0 has two mat-vecs here and only one of them is the bar: matvec_q40.comp
// reads float activations, and matvec.comp reads them quantized to Q8_0 and
// spends one dotPacked4x8AccSatEXT on eight weights. The second is the path
// most of a model's projections take and it is much the faster, so a trellis
// kernel that beats the first and loses to the second still loses the model.
//
// Round-robin and fastest-of-n, for the reason vk/golemtune.go gives: measured
// one after another, these three move together by a third between runs and
// against each other by nothing that can be believed.

import (
	"math/rand"
	"testing"
	"time"
	"unsafe"

	"github.com/ThiraSoft/golem/nn"
)

func TestGolemAgainstQ40(t *testing.T) {
	if testing.Short() {
		t.Skip("times four kernels over a 9728x2560 product")
	}
	const rows, cols = 9728, 2560
	const times, rounds = 128, 6

	d := open(t)
	defer d.Close()
	r := rand.New(rand.NewSource(23))

	type entry struct {
		name   string
		set    *Set
		groups uint32
		push   unsafe.Pointer
		bytes  int
		best   time.Duration
	}
	var runs []*entry

	// The floor of any trellis kernel: the window cut and the table read, with
	// the float multiply, the fused add and the activation all taken out. It is
	// ABLATE 6 of vk/shaders/matvec_t4g.comp, and nothing that decodes a weight
	// from a twelve-bit window can be faster than it.
	{
		data := make([]byte, rows*(nn.Matrix{Quant: nn.T4G, Cols: cols}).RowBytes())
		r.Read(data)
		pipe, err := d.NewPipelineSpec(ablateSPIRV(t, 4, 6), 4,
			uint32(unsafe.Sizeof(golemPush{})), GolemShapes()[1].Spec())
		if err != nil {
			t.Fatal(err)
		}
		defer pipe.Close()
		w, err := d.UploadTail(data, golemReadTail)
		if err != nil {
			t.Fatal(err)
		}
		defer w.Close()
		table, err := d.Upload(golemTable())
		if err != nil {
			t.Fatal(err)
		}
		defer table.Close()
		act, err := d.Host(cols*4, bufferUsageStorage)
		if err != nil {
			t.Fatal(err)
		}
		defer act.Close()
		out, err := d.Readback(rows*4, bufferUsageStorage)
		if err != nil {
			t.Fatal(err)
		}
		defer out.Close()
		set, err := pipe.NewSet([]*Buffer{w, table, act, out})
		if err != nil {
			t.Fatal(err)
		}
		defer set.Close()
		push := golemPush{dim: rows, ffn: cols}
		per := uint32(GolemShapes()[1].Threads / 8)
		runs = append(runs, &entry{
			name: "T4G floor", set: set, groups: (rows + per - 1) / per,
			push: unsafe.Pointer(&push), bytes: len(data), best: time.Hour,
		})
	}

	// The trellis tiers, each built the way vk/golem.go builds a pass of one.
	for _, kind := range []nn.Quant{nn.T3G, nn.T4G} {
		data := make([]byte, rows*(nn.Matrix{Quant: kind, Cols: cols}).RowBytes())
		r.Read(data)
		k, err := NewGolemKernels(d, kind)
		if err != nil {
			t.Fatal(err)
		}
		defer k.Close()
		act, err := d.Host(cols*4, bufferUsageStorage)
		if err != nil {
			t.Fatal(err)
		}
		defer act.Close()
		out, err := d.Readback(rows*4, bufferUsageStorage)
		if err != nil {
			t.Fatal(err)
		}
		defer out.Close()
		m, err := NewGolemMatrixOn(k, data, rows, cols, act, out)
		if err != nil {
			t.Fatal(err)
		}
		defer m.Close()
		push := m.Push(0)
		runs = append(runs, &entry{
			name: kind.String(), set: m.Set(1), groups: m.Groups(1),
			push: unsafe.Pointer(&push), bytes: len(data), best: time.Hour,
		})
	}

	// The two Q4_0 kernels, on a row of the same width.
	q40 := make([]byte, rows*rowBytesQ4_0(cols))
	r.Read(q40)
	w, err := d.Upload(q40)
	if err != nil {
		t.Fatal(err)
	}
	defer w.Close()
	out, err := d.Readback(rows*4, bufferUsageStorage)
	if err != nil {
		t.Fatal(err)
	}
	defer out.Close()

	xf, err := d.Host(cols*4, bufferUsageStorage)
	if err != nil {
		t.Fatal(err)
	}
	defer xf.Close()
	floatPipe, err := d.NewPipeline(matvecQ40SPIRV, 3, uint32(unsafe.Sizeof(matvecKPush{})))
	if err != nil {
		t.Fatal(err)
	}
	defer floatPipe.Close()
	floatSet, err := floatPipe.NewSet([]*Buffer{w, xf, out})
	if err != nil {
		t.Fatal(err)
	}
	defer floatSet.Close()
	fp := matvecKPush{Dim: rows, FFN: cols}
	runs = append(runs, &entry{
		name: "Q4_0 float", set: floatSet, groups: (rows + 15) / 16,
		push: unsafe.Pointer(&fp), bytes: len(q40), best: time.Hour,
	})

	// The activation in its Q8_0 form: a byte a value, then a scale and a
	// correction for every block of thirty-two.
	aq, err := d.Host(cols, bufferUsageStorage)
	if err != nil {
		t.Fatal(err)
	}
	defer aq.Close()
	as, err := d.Host(2*cols/32*4, bufferUsageStorage)
	if err != nil {
		t.Fatal(err)
	}
	defer as.Close()
	dotPipe, err := d.NewPipeline(matvecSPIRV, 4, uint32(unsafe.Sizeof(moePush{})))
	if err != nil {
		t.Fatal(err)
	}
	defer dotPipe.Close()
	dotSet, err := dotPipe.NewSet([]*Buffer{w, aq, as, out})
	if err != nil {
		t.Fatal(err)
	}
	defer dotSet.Close()
	dp := moePush{dim: rows, ffn: cols, used: 1}
	runs = append(runs, &entry{
		name: "Q4_0 dot4", set: dotSet, groups: (rows + 15) / 16,
		push: unsafe.Pointer(&dp), bytes: len(q40), best: time.Hour,
	})

	for round := 0; round <= rounds; round++ {
		for _, e := range runs {
			start := time.Now()
			if err := e.set.DispatchTimes(e.groups, e.push, times); err != nil {
				t.Fatal(err)
			}
			if took := time.Since(start) / times; round > 0 && took < e.best {
				e.best = took
			}
		}
	}
	var bar time.Duration
	for _, e := range runs {
		if e.name == "Q4_0 dot4" {
			bar = e.best
		}
	}
	for _, e := range runs {
		us := float64(e.best.Nanoseconds()) / 1000
		t.Logf("%-11s %6.2f us  %6.1f GB/s  %.2f of the Q4_0 dot4 kernel",
			e.name, us, float64(e.bytes)/e.best.Seconds()/1e9, e.best.Seconds()/bar.Seconds())
	}
}
