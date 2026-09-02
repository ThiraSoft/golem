package vk

// What the upload path costs, route by route, against the link the card
// negotiates.
//
// The streamed-mixture work rests on one number: how fast a weight crosses the
// bus. The plan recorded 6-7 GB/s for every route the repository has, against a
// link of 32 GT/s on sixteen lanes — about 63 GB/s of payload. An eight-fold
// gap is not a property of PCIe, so it is a property of this path, and this
// file exists to say which part of it.
//
// Every route moves the same payload into an allocation it did not make, timed
// over several passes with the best kept. **Allocating is not transferring**:
// a gigabyte of host-visible memory costs a quarter of a million page faults on
// first touch, and a first attempt at this table put that inside the clock and
// reported a third of the truth for the routes that allocate most.
//
// The interesting rows are the two that take the shipped path apart: "stage
// only" writes the host buffer and never submits, "dma only" submits a staging
// buffer that is already full. The shipped path does them one after the other,
// so what it can reach is their harmonic sum and nothing better.

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
	"unsafe"
)

// uploadBytes is what every route moves. A gigabyte is large enough that one
// submission's overhead does not decide the answer, and small enough to sit in
// this machine's RAM several times over.
const uploadBytes = 1 << 30

// uploadPasses is how many times each route moves it. The best is kept: a route
// is being asked what it can do, and a scheduler that took the machine away for
// a millisecond is not an answer about the bus.
const uploadPasses = 5

// linkRate walks from the card up to the root port and reports the narrowest
// hop, as payload bytes a second.
//
// **Reading the card's own link is how this repository came to believe it had a
// 63 GB/s bus.** The card here sits behind a switch: its own port trains at
// 32 GT/s on sixteen lanes and says so in sysfs, while the hop that actually
// reaches the processor — the root port — trains at 8 GT/s on eight. A transfer
// crosses every hop, so the bus is the slowest of them and not the nearest.
//
// A lane carries its transfer rate under a line code: 128b/130b at 8 GT/s and
// above, 8b/10b below, which is what the ratio here chooses between.
func linkRate() (string, float64) {
	paths, _ := filepath.Glob("/sys/class/drm/card*/device/current_link_speed")
	worst, name := 0.0, ""
	for _, p := range paths {
		dir, err := filepath.EvalSymlinks(filepath.Dir(p))
		if err != nil {
			continue
		}
		// Up the tree until the path leaves the PCI hierarchy. Each step is one
		// hop, and every one of them carries the transfer.
		for strings.Contains(dir, "/pci") {
			gts, lanes, ok := linkAt(dir)
			if ok {
				code := 128.0 / 130.0
				if gts < 8 {
					code = 8.0 / 10.0
				}
				if rate := gts * 1e9 / 8 * code * lanes; worst == 0 || rate < worst {
					worst = rate
					name = fmt.Sprintf("%s at %g GT/s x%g", filepath.Base(dir), gts, lanes)
				}
			}
			up := filepath.Dir(dir)
			if up == dir {
				break
			}
			dir = up
		}
	}
	return name, worst
}

// linkAt reads one device's negotiated speed and width.
func linkAt(dir string) (gts, lanes float64, ok bool) {
	speed, err := os.ReadFile(filepath.Join(dir, "current_link_speed"))
	if err != nil {
		return 0, 0, false
	}
	width, err := os.ReadFile(filepath.Join(dir, "current_link_width"))
	if err != nil {
		return 0, 0, false
	}
	if _, err := fmt.Sscanf(strings.TrimSpace(string(speed)), "%g", &gts); err != nil {
		return 0, 0, false
	}
	if _, err := fmt.Sscanf(strings.TrimSpace(string(width)), "%g", &lanes); err != nil {
		return 0, 0, false
	}
	return gts, lanes, lanes > 0
}

// TestUploadRoutes measures every way this repository has of putting a byte on
// the card, and prints them beside the link.
func TestUploadRoutes(t *testing.T) {
	if testing.Short() {
		t.Skip("moves tens of gigabytes across the bus")
	}
	d := open(t)
	defer d.Close()

	src := make([]byte, uploadBytes)
	for i := range src {
		src[i] = byte(i)
	}

	// Every allocation any route wants, made once and touched once, so that no
	// clock below contains a page fault or a vkAllocateMemory.
	dst, err := d.newBuffer(uploadBytes, bufferUsageStorage|bufferUsageTransferDst, memoryDeviceLocal)
	if err != nil {
		t.Fatal(err)
	}
	defer dst.Close()

	staging := map[string]*Buffer{}
	for name, props := range map[string]uint32{
		"write-combined": memoryHostVisible | memoryHostCoherent,
		"cached":         memoryHostVisible | memoryHostCoherent | memoryHostCached,
		"second":         memoryHostVisible | memoryHostCoherent,
	} {
		b, err := d.newBuffer(uploadBytes, bufferUsageTransferSrc, props)
		if err != nil {
			t.Fatalf("%s staging: %v", name, err)
		}
		defer b.Close()
		clear(b.Bytes()) // first touch, outside every clock below
		staging[name] = b
	}

	// Resizable BAR: device memory with an address on this side. The card
	// offers it here as memory type three; a card without it fails this
	// allocation and the row is left out rather than faked.
	bar, err := d.newBuffer(uploadBytes, bufferUsageStorage, memoryDeviceLocal|memoryHostVisible|memoryHostCoherent)
	if err != nil {
		t.Logf("no host-visible device memory: %v", err)
	} else {
		defer bar.Close()
		clear(bar.Bytes())
	}

	type route struct {
		name string
		rate float64
	}
	var out []route
	measure := func(name string, run func() error) {
		t.Helper()
		best := 0.0
		for i := 0; i < uploadPasses; i++ {
			start := time.Now()
			if err := run(); err != nil {
				t.Errorf("%s: %v", name, err)
				return
			}
			if rate := float64(uploadBytes) / time.Since(start).Seconds() / 1e9; rate > best {
				best = rate
			}
		}
		out = append(out, route{name, best})
	}

	// What this machine's memory does when the card is not involved at all. It
	// is the ceiling on any route that stages, and the number the write-combined
	// rows are to be read against.
	host := make([]byte, uploadBytes)
	clear(host)
	measure("host to host memcpy", func() error { copy(host, src); return nil })

	// What the repository actually does with a tensor, into a destination it did
	// not have to allocate. It is the row the others explain.
	measure("CopyInto, as shipped", func() error { return d.CopyInto(dst, 0, src) })

	// The path as it stood before: a copy and a submission alternating through
	// one staging buffer.
	wc := staging["write-combined"]
	measure("stage only, 64 MiB", func() error { return stageOnly(src, wc, 64<<20) })
	measure("dma only, 64 MiB", func() error { return dmaOnly(d, wc, dst, uploadBytes, 64<<20) })
	measure("stage then dma, 64 MiB", func() error { return stageThenDMA(d, src, wc, dst, 64<<20) })

	// The same alternation at other widths. If the submission is what costs,
	// these move; if the copy is, they do not.
	for _, chunk := range []int{8 << 20, 256 << 20, uploadBytes} {
		name := fmt.Sprintf("stage then dma, %d MiB", chunk>>20)
		measure(name, func() error { return stageThenDMA(d, src, wc, dst, chunk) })
	}

	// The alternation taken out: the copy of chunk i+1 runs on this side while
	// the card is still fetching chunk i. Two staging buffers and no extra
	// Vulkan object — the copy is a memcpy and the wait is a queue wait, and
	// they are already on different sides of the machine.
	for _, chunk := range []int{4 << 20, 16 << 20, 64 << 20, 256 << 20} {
		name := fmt.Sprintf("staged ahead, %d MiB", chunk>>20)
		measure(name, func() error { return stageAhead(d, src, wc, staging["second"], dst, chunk) })
	}

	// Staging through cached host memory instead: writing it is an ordinary
	// store rather than a write-combine, and reading it back is the card's
	// problem rather than this side's.
	cached := staging["cached"]
	measure("cached stage only", func() error { return stageOnly(src, cached, 64<<20) })
	measure("cached dma only", func() error { return dmaOnly(d, cached, dst, uploadBytes, 64<<20) })

	// One whole-payload copy command, staging already full: what the copy
	// engine does with no alternation at all.
	measure("dma only, one submit", func() error { return dmaOnly(d, wc, dst, uploadBytes, uploadBytes) })

	// The same copy command with the bus taken out of it: device memory to
	// device memory. It says whether 6.6 GB/s is what the copy engine costs or
	// what the bus costs, and those are not the same finding.
	local, err := d.newBuffer(uploadBytes, bufferUsageStorage|bufferUsageTransferSrc, memoryDeviceLocal)
	if err != nil {
		t.Fatal(err)
	}
	defer local.Close()
	measure("dma, VRAM to VRAM", func() error { return dmaOnly(d, local, dst, uploadBytes, 64<<20) })

	// No staging and no copy on this side at all: the source pointer is handed
	// to the card, and the copy engine reads the process's own pages.
	if align := d.ImportAlignment(); align == 0 {
		t.Log("no host-pointer import on this device")
	} else {
		room := make([]byte, uploadBytes+int(align))
		off := (align - uint64(uintptr(unsafe.Pointer(&room[0])))%align) % align
		aligned := room[off : off+uploadBytes]
		copy(aligned, src)
		imported, err := d.Import(unsafe.Pointer(&aligned[0]), uploadBytes, bufferUsageTransferSrc)
		if err != nil {
			t.Errorf("import: %v", err)
		} else {
			defer imported.Close()
			measure("imported, 64 MiB", func() error { return dmaOnly(d, imported, dst, uploadBytes, 64<<20) })
			measure("imported, one submit", func() error { return dmaOnly(d, imported, dst, uploadBytes, uploadBytes) })

			// The same bytes imported as sixteen allocations rather than one,
			// each submitted whole. It separates the two things the row above
			// confounds: what a submission costs, and what referencing a large
			// imported allocation costs in each of them.
			const piece = 64 << 20
			var pieces []*Buffer
			for off := 0; off+piece <= uploadBytes; off += piece {
				b, err := d.Import(unsafe.Pointer(&aligned[off]), piece, bufferUsageTransferSrc)
				if err != nil {
					t.Errorf("import piece: %v", err)
					break
				}
				defer b.Close()
				pieces = append(pieces, b)
			}
			measure("imported in pieces, 64 MiB", func() error {
				for i, b := range pieces {
					if err := dmaOnly(d, b, dst, piece, piece); err != nil {
						return err
					}
					_ = i
				}
				return nil
			})
		}
	}

	// No staging and no copy engine: the bytes cross as stores.
	if bar != nil {
		measure("rebar, stores into VRAM", func() error { copy(bar.Bytes(), src); return nil })
	}

	if name, rate := linkRate(); name != "" {
		t.Logf("link: %s, about %.1f GB/s of payload", name, rate/1e9)
	}
	for _, r := range out {
		t.Logf("%-28s %6.2f GB/s", r.name, r.rate)
	}
}

// stageAhead is the shipped loop with the two stages overlapped: while the card
// copies out of one staging buffer, this side fills the other.
//
// Nothing here is asynchronous on the card. The submission still blocks, and it
// is the host's memcpy that moves off the critical path, which is the half that
// is free to move: the card is idle for exactly as long as this side is copying.
func stageAhead(d *Device, data []byte, a, b, dst *Buffer, chunk int) error {
	buf := [2]*Buffer{a, b}
	type ready struct{ slot, off int }
	// free carries the slots this side may write. A slot goes back into it only
	// once the card has finished reading it — without that the producer runs a
	// buffer ahead into one still in flight, which measured 4.74 GB/s against
	// the 6.70 of the copy alone and was overwriting the bytes being sent.
	free := make(chan int, 2)
	free <- 0
	free <- 1
	filled := make(chan ready, 1)
	go func() {
		for off := 0; off < len(data); off += chunk {
			slot := <-free
			n := min(chunk, len(data)-off)
			copy(buf[slot].Bytes()[:n], data[off:off+n])
			filled <- ready{slot, off}
		}
		close(filled)
	}()
	for r := range filled {
		n := min(chunk, len(data)-r.off)
		src := buf[r.slot]
		if err := d.run(func(cb commandBuffer) {
			region := bufferCopy{srcOffset: 0, dstOffset: uint64(r.off), size: uint64(n)}
			vkCmdCopyBuffer(cb, src.handle, dst.handle, 1, &region)
		}); err != nil {
			return err
		}
		free <- r.slot
	}
	return nil
}

// stageOnly writes the payload into the staging buffer a chunk at a time and
// never submits, which is the host's half of the shipped path.
func stageOnly(data []byte, host *Buffer, chunk int) error {
	into := host.Bytes()
	for off := 0; off < len(data); off += chunk {
		n := min(chunk, len(data)-off)
		copy(into[:n], data[off:off+n])
	}
	return nil
}

// dmaOnly submits copies out of a staging buffer whose contents it does not
// write, which is the card's half. What the buffer holds does not change what
// the copy engine costs.
func dmaOnly(d *Device, host, dst *Buffer, total, chunk int) error {
	for off := 0; off < total; off += chunk {
		n := min(chunk, total-off)
		if err := d.run(func(cb commandBuffer) {
			region := bufferCopy{srcOffset: 0, dstOffset: uint64(off), size: uint64(n)}
			vkCmdCopyBuffer(cb, host.handle, dst.handle, 1, &region)
		}); err != nil {
			return err
		}
	}
	return nil
}

// stageThenDMA is UploadTail's loop with the allocation lifted out: copy a
// chunk in, submit it, wait, take the next.
func stageThenDMA(d *Device, data []byte, host, dst *Buffer, chunk int) error {
	into := host.Bytes()
	for off := 0; off < len(data); off += chunk {
		n := min(chunk, len(data)-off)
		copy(into[:n], data[off:off+n])
		if err := d.run(func(cb commandBuffer) {
			region := bufferCopy{srcOffset: 0, dstOffset: uint64(off), size: uint64(n)}
			vkCmdCopyBuffer(cb, host.handle, dst.handle, 1, &region)
		}); err != nil {
			return err
		}
	}
	return nil
}
