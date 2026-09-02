package vk

// The admission kernel, which is what turns an expert's identifier into the
// place in device memory where a copy of it lives.
//
// A cache means the two expert kernels can no longer read `id * stride`: expert
// 114 is in whatever slot it was given. Rather than teach those two kernels
// about residency — they are the hottest in the engine — the identifiers are
// rewritten between the pick and the product, in place, by this one workgroup.
// After it, `ids` holds slots and everything downstream is unchanged.
//
// It also says what has to be fetched: a miss takes a slot from whichever
// expert went longest unused, and the pair goes on a list the fill kernel reads.

import (
	"testing"
	"unsafe"
)

// An admitTable is the residence of one block's experts, laid out as the kernel
// reads it: where each expert is, what each slot holds, when each slot was last
// wanted, and a clock to answer that last question with.
type admitTable struct {
	experts, slots int
	data           []int32
}

func newAdmitTable(experts, slots int) *admitTable {
	t := &admitTable{experts: experts, slots: slots, data: make([]int32, experts+2*slots+1)}
	for i := range t.data {
		t.data[i] = -1
	}
	for i := experts + slots; i < len(t.data); i++ {
		t.data[i] = 0 // the recencies and the clock start at zero, not at -1
	}
	return t
}

func (t *admitTable) slotOf(e int) int32   { return t.data[e] }
func (t *admitTable) expertIn(s int) int32 { return t.data[t.experts+s] }

// runAdmit puts ids and the table on the card, runs the kernel, and reads all
// three back.
//
// Host-visible buffers throughout: these are a few hundred bytes, the kernel
// reads and writes them the same way it would device memory, and it saves a
// staging copy in each direction for a test that is about arithmetic and not
// about the bus.
func runAdmit(t *testing.T, d *Device, tb *admitTable, used, columns int, ids []int32) (slots []int32, fills []int32) {
	t.Helper()
	n := used * columns
	if len(ids) != n {
		t.Fatalf("%d identifiers for %d columns of %d", len(ids), columns, used)
	}
	host := func(vals []int32) *Buffer {
		b, err := d.Host(len(vals)*4, bufferUsageStorage)
		if err != nil {
			t.Fatal(err)
		}
		copy(ints32(b, len(vals)), vals)
		return b
	}
	idBuf := host(ids)
	defer idBuf.Close()
	tbBuf := host(tb.data)
	defer tbBuf.Close()
	// One count, then a pair per possible miss.
	flBuf := host(make([]int32, 1+2*n))
	defer flBuf.Close()

	pipe, err := d.NewPipeline(moeAdmitSPIRV, 3, uint32(unsafe.Sizeof(admitPush{})))
	if err != nil {
		t.Fatal(err)
	}
	defer pipe.Close()
	set, err := pipe.NewSet([]*Buffer{idBuf, tbBuf, flBuf})
	if err != nil {
		t.Fatal(err)
	}
	push := admitPush{Experts: uint32(tb.experts), Slots: uint32(tb.slots),
		Used: uint32(used), Columns: uint32(columns)}
	if err := set.Dispatch(1, unsafe.Pointer(&push)); err != nil {
		t.Fatal(err)
	}
	copy(tb.data, ints32(tbBuf, len(tb.data)))
	return append([]int32(nil), ints32(idBuf, n)...), append([]int32(nil), ints32(flBuf, 1+2*n)...)
}

// ints32 reads a host-visible buffer as signed words.
func ints32(b *Buffer, n int) []int32 {
	return unsafe.Slice((*int32)(unsafe.Pointer(&b.Bytes()[0])), n)
}

// TestAdmitResolvesSlots is the whole contract in one run: a first sight of an
// expert takes a slot and goes on the fill list, a second sight of it inside the
// same pass costs nothing, and the identifiers come back as slots.
func TestAdmitResolvesSlots(t *testing.T) {
	d := open(t)
	defer d.Close()

	tb := newAdmitTable(8, 3)
	// Two columns of two: expert 5, expert 5, then expert 3, expert 5.
	ids, fills := runAdmit(t, d, tb, 2, 2, []int32{5, 5, 3, 5})

	want := []int32{0, 0, 1, 0}
	for i := range want {
		if ids[i] != want[i] {
			t.Fatalf("identifier %d resolved to slot %d, want %d (all of them: %v)", i, ids[i], want[i], ids)
		}
	}
	if fills[0] != 2 {
		t.Fatalf("%d experts to fetch, want 2 — five and three, each once", fills[0])
	}
	got := [][2]int32{{fills[1], fills[2]}, {fills[3], fills[4]}}
	want2 := [][2]int32{{5, 0}, {3, 1}}
	for i := range want2 {
		if got[i] != want2[i] {
			t.Fatalf("fetch %d is expert %d into slot %d, want expert %d into slot %d",
				i, got[i][0], got[i][1], want2[i][0], want2[i][1])
		}
	}
	if tb.slotOf(5) != 0 || tb.slotOf(3) != 1 || tb.expertIn(0) != 5 || tb.expertIn(1) != 3 {
		t.Fatalf("the table does not hold what was admitted: five is at %d, three at %d; slot nought holds %d and slot one holds %d",
			tb.slotOf(5), tb.slotOf(3), tb.expertIn(0), tb.expertIn(1))
	}
}

// TestAdmitEvictsTheLeastRecentlyWanted holds the replacement policy. Three
// experts through two slots: the third takes the place of whichever of the
// first two went longest unused, which is the first.
func TestAdmitEvictsTheLeastRecentlyWanted(t *testing.T) {
	d := open(t)
	defer d.Close()

	tb := newAdmitTable(8, 2)
	ids, fills := runAdmit(t, d, tb, 3, 1, []int32{1, 2, 3})

	if ids[0] != 0 || ids[1] != 1 {
		t.Fatalf("the first two took slots %d and %d, want nought and one", ids[0], ids[1])
	}
	if ids[2] != 0 {
		t.Fatalf("the third took slot %d, want nought — one was wanted longer ago than two", ids[2])
	}
	if fills[0] != 3 {
		t.Fatalf("%d experts to fetch, want three: none of them was resident", fills[0])
	}
	if tb.slotOf(1) != -1 {
		t.Fatalf("expert one still claims slot %d after being evicted", tb.slotOf(1))
	}
	if tb.expertIn(0) != 3 {
		t.Fatalf("slot nought holds expert %d, want three", tb.expertIn(0))
	}
}
