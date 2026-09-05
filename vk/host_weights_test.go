package vk

// What a weight costs a kernel when it lives in host memory rather than on the
// card.
//
// This is the measurement the streamed mixture rests on, and it asks a question
// the plan never asked. The plan assumes a missing expert has to be *uploaded*
// before it can be used, which means the host has to know it is missing, which
// means the host has to know the routing — and the routing is on the card, one
// block's worth at a time, inside a program recorded once and re-run per token.
// That is the wall §3 of the findings runs into.
//
// There may be no wall. This card's heap zero is fifteen and a half gibibytes of
// system memory the GPU can address, and Device.Host allocates in it. A shader
// reads a storage buffer the same way wherever it lives; what changes is the
// rate. So an expert pool that does not fit in device memory could simply *stay*
// in host memory and be read across the bus by the kernel that wants it — no
// upload to schedule, no prefetch to get right, no readback, and no host
// decision between two blocks. A cache in device memory then stops being the
// mechanism and becomes an optimisation: a copy of the experts worth copying.
//
// The whole design turns on one number: what a mat-vec reads at out of host
// memory. If it is the bus, the arrangement above costs exactly what
// gemma/expert_cache_test.go's table already prices. If it is much less than the
// bus — because a kernel's access pattern is not a DMA's, and a shader stalling
// on the bus is not the same as a copy engine streaming it — then the upload the
// plan wanted is the only route and this is a dead end worth knowing about.

import (
	"math/rand"
	"testing"
	"time"
	"unsafe"

	"github.com/ThiraSoft/golem/internal/heavy"
)

func TestWeightsInHostMemory(t *testing.T) {
	heavy.Skip(t, "moves a gigabyte through a kernel several times")
	d := open(t)
	defer d.Close()

	// Large enough that the card's sixty-four megabytes of last-level cache
	// cannot hold it, so the rate is the memory's and not the cache's. The
	// columns are a block of the 26B A4B's expert width; the rows are as many
	// as it takes to be a gigabyte.
	const cols = 2816
	const rows = 95000
	weights := make([]float32, rows*cols)
	rng := rand.New(rand.NewSource(7))
	for i := range weights {
		weights[i] = rng.Float32()*2 - 1
	}
	bytes := len(weights) * 4
	x := make([]float32, cols)
	for i := range x {
		x[i] = rng.Float32()*2 - 1
	}

	xb, err := d.Upload(asBytes(x))
	if err != nil {
		t.Fatal(err)
	}
	defer xb.Close()
	y, err := d.Readback(rows*4, bufferUsageStorage)
	if err != nil {
		t.Fatal(err)
	}
	defer y.Close()
	pipe, err := d.NewPipeline(matvecF32SPIRV, 3, uint32(unsafe.Sizeof(matvecKPush{})))
	if err != nil {
		t.Fatal(err)
	}
	defer pipe.Close()

	// The same product over the same numbers, from the two places a buffer can
	// live. The answers are compared as well as the rates: a kernel reading
	// host memory is not a different kernel, and if it were the rate would not
	// mean anything.
	measure := func(name string, w *Buffer) []float32 {
		t.Helper()
		set, err := pipe.NewSet([]*Buffer{w, xb, y})
		if err != nil {
			t.Fatal(err)
		}
		push := matvecKPush{Dim: rows, FFN: cols}
		groups := uint32((rows + 15) / 16) // OUTS is sixteen in the kernel
		const passes = 4
		if err := set.Dispatch(groups, unsafe.Pointer(&push)); err != nil {
			t.Fatal(err) // warm
		}
		start := time.Now()
		if err := set.DispatchTimes(groups, unsafe.Pointer(&push), passes); err != nil {
			t.Fatal(err)
		}
		seconds := time.Since(start).Seconds() / passes
		t.Logf("%-24s %6.1f GB/s  %8.2f ms a pass", name, float64(bytes)/seconds/1e9, seconds*1e3)
		return append([]float32(nil), y.Floats()[:rows]...)
	}

	local, err := d.Upload(asBytes(weights))
	if err != nil {
		t.Fatal(err)
	}
	fromCard := measure("device memory", local)
	local.Close()

	// Host memory the GPU addresses: what an expert pool larger than the card
	// would live in. It is written once from this side and read by the kernel
	// across the bus for every pass.
	host, err := d.Host(bytes, bufferUsageStorage)
	if err != nil {
		t.Skipf("no host-visible allocation of %d bytes: %v", bytes, err)
	}
	defer host.Close()
	copy(host.Bytes(), asBytes(weights))
	fromHost := measure("host memory, over the bus", host)

	for i := range fromCard {
		if fromCard[i] != fromHost[i] {
			t.Fatalf("row %d answers %v from the card and %v from host memory",
				i, fromCard[i], fromHost[i])
		}
	}
	if name, rate := linkRate(); name != "" {
		t.Logf("the bus is %s, about %.1f GB/s of payload; the copy engine reaches 6.7", name, rate/1e9)
	}
}
