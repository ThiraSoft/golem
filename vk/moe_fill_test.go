package vk

// Fetching an expert into the slot the admission gave it.
//
// The copy is device-side on purpose. The host does not know the routing — it
// is picked on the card, one block at a time, inside a program recorded once —
// so anything the host had to decide would cost a readback between every pair
// of blocks, which is the arrangement vk/stack.go exists to avoid. A kernel
// reading the pool and writing the cache needs none of that.
//
// It costs nothing over reading the pool directly. A miss crosses the bus once
// either way: 3.3 MB at 6.3 GB/s, half a millisecond. What it buys is every
// later token that wants the same expert, which then reads device memory at
// five hundred gigabytes a second instead of six.

import (
	"testing"
	"unsafe"
)

// TestFillCopiesTheAdmittedExperts holds the one thing this kernel does: after
// it, the slot named by each fetch holds the expert named beside it, and no
// other slot has moved.
func TestFillCopiesTheAdmittedExperts(t *testing.T) {
	d := open(t)
	defer d.Close()

	// Four experts of eight words, a cache of two slots.
	const experts, words, slots = 4, 8, 2
	pool := make([]int32, experts*words)
	for e := 0; e < experts; e++ {
		for w := 0; w < words; w++ {
			pool[e*words+w] = int32(100*e + w)
		}
	}
	host := func(vals []int32) *Buffer {
		b, err := d.Host(len(vals)*4, bufferUsageStorage)
		if err != nil {
			t.Fatal(err)
		}
		copy(ints32(b, len(vals)), vals)
		return b
	}
	src := host(pool)
	defer src.Close()
	dst := host(make([]int32, slots*words))
	defer dst.Close()
	// Expert two into slot one, expert nought into slot nought.
	fl := host([]int32{2, 2, 1, 0, 0})
	defer fl.Close()

	pipe, err := d.NewPipeline(moeFillSPIRV, 3, uint32(unsafe.Sizeof(fillPush{})))
	if err != nil {
		t.Fatal(err)
	}
	defer pipe.Close()
	set, err := pipe.NewSet([]*Buffer{src, dst, fl})
	if err != nil {
		t.Fatal(err)
	}
	// Sized for the worst case — every fetch the list could hold — because a
	// recorded program cannot ask how many there turned out to be. The
	// workgroups past the count leave immediately.
	const groups = 1
	push := fillPush{Words: words, Groups: groups}
	if err := set.Dispatch(slots*groups, unsafe.Pointer(&push)); err != nil {
		t.Fatal(err)
	}

	got := ints32(dst, slots*words)
	for w := 0; w < words; w++ {
		if want := int32(0*100 + w); got[0*words+w] != want {
			t.Fatalf("slot nought word %d is %d, want %d — expert nought was fetched into it", w, got[0*words+w], want)
		}
		if want := int32(2*100 + w); got[1*words+w] != want {
			t.Fatalf("slot one word %d is %d, want %d — expert two was fetched into it", w, got[1*words+w], want)
		}
	}
}

// TestFillStopsAtTheCount is the half that a recorded program makes necessary:
// the dispatch is sized for every fetch the list could hold, and a run with
// fewer must leave the rest of the cache alone.
func TestFillStopsAtTheCount(t *testing.T) {
	d := open(t)
	defer d.Close()

	const experts, words, slots = 4, 8, 2
	pool := make([]int32, experts*words)
	for i := range pool {
		pool[i] = int32(i + 1)
	}
	host := func(vals []int32) *Buffer {
		b, err := d.Host(len(vals)*4, bufferUsageStorage)
		if err != nil {
			t.Fatal(err)
		}
		copy(ints32(b, len(vals)), vals)
		return b
	}
	src := host(pool)
	defer src.Close()
	kept := make([]int32, slots*words)
	for i := range kept {
		kept[i] = -7
	}
	dst := host(kept)
	defer dst.Close()
	// One fetch, into slot nought. Slot one must not be touched.
	fl := host([]int32{1, 3, 0, 0, 0})
	defer fl.Close()

	pipe, err := d.NewPipeline(moeFillSPIRV, 3, uint32(unsafe.Sizeof(fillPush{})))
	if err != nil {
		t.Fatal(err)
	}
	defer pipe.Close()
	set, err := pipe.NewSet([]*Buffer{src, dst, fl})
	if err != nil {
		t.Fatal(err)
	}
	push := fillPush{Words: words, Groups: 1}
	if err := set.Dispatch(slots, unsafe.Pointer(&push)); err != nil {
		t.Fatal(err)
	}
	got := ints32(dst, slots*words)
	for w := 0; w < words; w++ {
		if want := pool[3*words+w]; got[w] != want {
			t.Fatalf("slot nought word %d is %d, want %d", w, got[w], want)
		}
		if got[words+w] != -7 {
			t.Fatalf("slot one word %d is %d, and nothing asked for it to change", w, got[words+w])
		}
	}
}
