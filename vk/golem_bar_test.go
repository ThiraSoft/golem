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
// one after another, these move together by a third between runs. Twenty rounds
// and not six, because six was not enough either — the same code came back at
// 1.53 and at 1.65 of the bar on two runs, and a kernel change worth eight per
// cent cannot be read off an instrument with eight per cent of slack in it.

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
	const times, rounds = 128, 20

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

	// The trellis tiers, each built the way vk/golem.go builds a pass of one,
	// and T4G a second time with the window cut the other way — the only way to
	// read a difference of a tenth on a card that moves by a sixth between
	// processes.
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

	// T4G hashing every weight instead of reading the shared table, and T4G
	// with the shift pair instead of the bitfield extract. Both are shapes this
	// kernel has had, kept side by side so that the few per cent between them
	// stays measured rather than remembered.
	for _, alt := range []struct {
		name   string
		shape  GolemShape
		lanes  int
		phased int
	}{
		{"T4G hash", GolemShape{Threads: 256, Table: false, Prefetch: true}, 8, 0},
		{"T4G lanes16", GolemShape{Threads: 256, Table: true, Prefetch: true}, 16, 0},
		{"T4G phased", GolemShape{Threads: 256, Table: true, Prefetch: true}, 8, 1},
		{"T4G phased/128", GolemShape{Threads: 128, Table: true, Prefetch: true}, 8, 1},
	} {
		data := make([]byte, rows*(nn.Matrix{Quant: nn.T4G, Cols: cols}).RowBytes())
		r.Read(data)
		pipe, err := d.NewPipelineSpec(buildGolemPhased(t, 4, 0, 1, alt.lanes, alt.phased), 4,
			uint32(unsafe.Sizeof(golemPush{})), alt.shape.Spec())
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
		per := uint32(alt.shape.Threads / alt.lanes)
		runs = append(runs, &entry{
			name: alt.name, set: set, groups: (rows + per - 1) / per,
			push: unsafe.Pointer(&push), bytes: len(data), best: time.Hour,
		})
	}
	{
		data := make([]byte, rows*(nn.Matrix{Quant: nn.T4G, Cols: cols}).RowBytes())
		r.Read(data)
		spv := buildGolemSPIRV(t, 4, 0, 0)
		pipe, err := d.NewPipelineSpec(spv, 4, uint32(unsafe.Sizeof(golemPush{})), GolemShapes()[1].Spec())
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
			name: "T4G shifts", set: set, groups: (rows + per - 1) / per,
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

	// The order rotates. Timed in a fixed order, whichever kernel sits fourth
	// comes out three to six per cent ahead of whichever sits third — the same
	// binary, both ways round — and two afternoons went into a conclusion that
	// was that effect and nothing else.
	for round := 0; round <= rounds; round++ {
		order := make([]*entry, len(runs))
		for i := range runs {
			order[i] = runs[(i+round)%len(runs)]
		}
		for _, e := range order {
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
