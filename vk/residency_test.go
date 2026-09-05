package vk

import (
	"testing"
	"unsafe"
)

// How much host memory a submission may reach at once.
//
// It is the number that decides how large a mixture this card can hold beside
// it. An expert pool that does not fit in device memory lives in system memory
// the card addresses, and every submission of a token reaches every block's
// pool — so the question is not how much can be allocated but how much can be
// made resident at once, and those are not the same number.
//
// **Allocating is not submitting.** Thirty gibibytes of host-visible buffers
// allocate here without complaint, because the driver allocates lazily. The
// first submission that names sixteen of them fails with "Not enough memory for
// command submission", which is the heap's own size — 15.63 GiB, half this
// machine's RAM, and amdgpu's gttsize is what sets it.
//
// So the 26B A4B in Q4_0 holds its 12.85 GB of experts here and runs; the same
// model in Q8_0 wants 24 GB and cannot, on this machine, without a boot
// parameter or a pool read from the file instead.
func TestResidencyCap(t *testing.T) {
	d := open(t)
	defer d.Close()
	pipe, err := d.NewPipeline(moeFillSPIRV, 3, uint32(unsafe.Sizeof(fillPush{})))
	if err != nil {
		t.Fatal(err)
	}
	defer pipe.Close()
	fl, _ := d.Host(1024, bufferUsageStorage)
	defer fl.Close()
	dst, _ := d.Host(1<<20, bufferUsageStorage)
	defer dst.Close()

	var kept []*Buffer
	var sets []*Set
	defer func() {
		for _, s := range sets {
			s.Close()
		}
		for _, b := range kept {
			b.Close()
		}
	}()
	push := fillPush{Words: 1, Groups: 1}
	for i := 1; i <= 26; i++ {
		b, err := d.Host(1<<30, bufferUsageStorage)
		if err != nil {
			t.Logf("allocation refusée à %d GiB: %v", i, err)
			return
		}
		kept = append(kept, b)
		s, err := pipe.NewSet([]*Buffer{b, dst, fl})
		if err != nil {
			t.Logf("descripteur refusé à %d GiB: %v", i, err)
			return
		}
		sets = append(sets, s)
		// Every buffer so far in one submission, which is what a token does.
		if err := d.Submit(func(r *Recorder) {
			for _, one := range sets {
				r.Dispatch(one, 1, unsafe.Pointer(&push))
			}
		}); err != nil {
			t.Logf("SOUMISSION refusée avec %d GiB résidents: %v", i, err)
			return
		}
		t.Logf("%d GiB soumis", i)
	}
}
