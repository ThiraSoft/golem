package vk

import (
	"testing"
	"unsafe"
)

// TestMemoryPropertiesLayout holds the Go structure to the C one it stands for.
//
// It exists because it was wrong. A filler before memoryHeaps, added to force
// an alignment that was already there, pushed the array eight bytes along, and
// every heap was read from the middle of its neighbour: sizes of zero, flags of
// 0xfb000000. Nothing caught it for as long as nothing read a heap — the only
// reader of this structure was memoryTypeFor, and the types sit before the
// heaps — so the fault surfaced only when a caller asked the card how large it
// is, got zero, and sized a window against it.
//
// A structure this side declares and the driver fills has no compiler holding
// the two together. This is the join.
func TestMemoryPropertiesLayout(t *testing.T) {
	var p physicalDeviceMemoryProperties
	for _, c := range []struct {
		what string
		got  uintptr
		want uintptr
	}{
		{"memoryTypes", unsafe.Offsetof(p.memoryTypes), 4},
		{"memoryHeapCount", unsafe.Offsetof(p.memoryHeapCount), 260},
		{"memoryHeaps", unsafe.Offsetof(p.memoryHeaps), 264},
		{"the whole of it", unsafe.Sizeof(p), 520},
	} {
		if c.got != c.want {
			t.Errorf("%s is at %d and Vulkan puts it at %d", c.what, c.got, c.want)
		}
	}
}

// TestDeviceLocalBytesIsPlausible asks the card what it holds. A driver that
// answered zero would size a working set to nothing and be taken for a small
// card rather than for a misread structure, which is exactly what happened.
func TestDeviceLocalBytesIsPlausible(t *testing.T) {
	d := open(t)
	defer d.Close()
	n := d.DeviceLocalBytes()
	if n < 128<<20 {
		t.Fatalf("the card reports %d bytes of device-local memory, which no card has", n)
	}
}
