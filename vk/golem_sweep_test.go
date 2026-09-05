package vk

// Which shape of the trellis matvec this card is fastest at, and whether every
// shape says the same thing.
//
// The timing lives in vk/golemtune.go, where cmd/golemtune can reach it. What
// is here is the correctness half — a shape that is faster and wrong has been
// shipped from a sweep before — and a reporting test that prints the table
// vk/golem.go's default came from.

import (
	"testing"
	"unsafe"

	"github.com/ThiraSoft/golem/internal/heavy"
	"github.com/ThiraSoft/golem/nn"
)

// golemProduct is y = W·x for one shape, for a caller that wants the answer
// rather than the time.
func golemProduct(t *testing.T, d *Device, kind nn.Quant, shape GolemShape, data []byte, rows, cols int) []float32 {
	t.Helper()
	spirv, ok := golemSPIRV(kind)
	if !ok {
		t.Fatalf("no kernel for %s", kind)
	}
	pipe, err := d.NewPipelineSpec(spirv[1], 4, uint32(unsafe.Sizeof(golemPush{})), shape.Spec())
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
	for i := range act.Floats() {
		act.Floats()[i] = float32(i%17) - 8
	}
	set, err := pipe.NewSet([]*Buffer{w, table, act, out})
	if err != nil {
		t.Fatal(err)
	}
	defer set.Close()
	per := uint32(shape.Threads / 8)
	push := golemPush{dim: uint32(rows), ffn: uint32(cols)}
	if err := set.Dispatch((uint32(rows)+per-1)/per, unsafe.Pointer(&push)); err != nil {
		t.Fatal(err)
	}
	return append([]float32(nil), out.Floats()...)
}

// TestGolemBuildsAgree is the correctness half of the sweep. Every shape
// cmd/golemtune is allowed to pick has to answer, bit for bit, what the shipped
// one answers — the shapes differ in how they reach the weights and where the
// codebook's image lives, and in nothing a product can see.
func TestGolemBuildsAgree(t *testing.T) {
	const rows, cols = 256, 512
	d := open(t)
	defer d.Close()
	// Two shapes past the tuned set, because the point is that the kernel is
	// correct at any of them and not only at the six that get timed.
	shapes := append([]GolemShape{{Threads: 64}, {Threads: 512, Table: true, Prefetch: true}}, GolemTuneShapes...)
	for _, kind := range []nn.Quant{nn.T3G, nn.T4G, nn.T5G} {
		// A real encode, because a kernel that reads the stream wrongly reads
		// random bytes just as happily.
		data, _ := t4gMatrixAs(t, rows, cols, kind)
		var want []float32
		for _, shape := range shapes {
			got := golemProduct(t, d, kind, shape, data, rows, cols)
			if want == nil {
				want = got
				continue
			}
			for i := range want {
				if got[i] != want[i] {
					t.Fatalf("%s %v: row %d reads %v against %v", kind, shape, i, got[i], want[i])
				}
			}
		}
	}
}

// TestGolemSweep prints what cmd/golemtune prints, so that the table in
// vk/golem.go can be checked without leaving the test suite. It reports and
// does not assert: the fastest shape is the card's answer, not this
// repository's.
func TestGolemSweep(t *testing.T) {
	heavy.Skip(t, "times every shape of every pass width over a 9728x2560 product")
	d := open(t)
	defer d.Close()
	rows, err := TuneGolem(d, 9728, 2560, 5)
	if err != nil {
		t.Fatal(err)
	}
	for _, r := range rows {
		us := float64(r.Best.Nanoseconds()) / 1000
		t.Logf("columns %d threads %3d table %-5t prefetch %-5t: %7.2f us  %6.2f us/column",
			r.Columns, r.Shape.Threads, r.Shape.Table, r.Shape.Prefetch, us, us/float64(r.Columns))
	}
	t.Logf("GOLEM_MATVEC_SHAPE=%q", GolemShapeSetting(GolemTuneBest(rows)))
}
