//go:build arm64

package nn

import "testing"

// The instructions the kernels need and Go's assembler will not spell, checked
// against arithmetic done by hand. An encoding typo assembles happily and means
// something else, so each one is held to a result only the intended instruction
// produces.
//
// Every lane here is a different value, and the four inputs of a lane are not
// interchangeable: a wrong lane order, a wrong element size, or the unsigned
// instruction where the signed one belongs all give a different answer.

func TestSDOTEncoding(t *testing.T) {
	if !dotprod {
		t.Skip("no FEAT_DotProd on this machine")
	}
	var acc [4]int32
	var a, b [16]int8
	for i := range a {
		a[i] = int8(i + 1) // 1..16
		b[i] = 2
	}
	sdotProbe(&acc[0], &a[0], &b[0])
	// Lane k accumulates a[4k..4k+3] against 2.
	want := [4]int32{20, 52, 84, 116}
	if acc != want {
		t.Errorf("SDOT gave %v, want %v", acc, want)
	}
}

func TestSDOTIsSigned(t *testing.T) {
	if !dotprod {
		t.Skip("no FEAT_DotProd on this machine")
	}
	var acc [4]int32
	var a, b [16]int8
	for i := range a {
		a[i] = -1
		b[i] = 1
	}
	sdotProbe(&acc[0], &a[0], &b[0])
	// Signed: four products of -1 per lane. UDOT would read -1 as 255 and give
	// 1020, which is what this test exists to catch — the two encodings are one
	// bit apart.
	want := [4]int32{-4, -4, -4, -4}
	if acc != want {
		t.Errorf("SDOT gave %v, want %v (1020 means the unsigned encoding)", acc, want)
	}
}

func TestSDOTAccumulates(t *testing.T) {
	if !dotprod {
		t.Skip("no FEAT_DotProd on this machine")
	}
	acc := [4]int32{100, 200, 300, 400}
	var a, b [16]int8
	for i := range a {
		a[i] = 1
		b[i] = 1
	}
	sdotProbe(&acc[0], &a[0], &b[0])
	// It adds into the destination rather than replacing it, which is the
	// property the kernels' inner loops are built on.
	want := [4]int32{104, 204, 304, 404}
	if acc != want {
		t.Errorf("SDOT gave %v, want %v", acc, want)
	}
}

func TestUDOTEncoding(t *testing.T) {
	if !dotprod {
		t.Skip("no FEAT_DotProd on this machine")
	}
	var acc [4]uint32
	var a, b [16]uint8
	for i := range a {
		a[i] = 255
		b[i] = 2
	}
	udotProbe(&acc[0], &a[0], &b[0])
	// Unsigned: 255 stays 255. The signed encoding would read it as -1.
	want := [4]uint32{2040, 2040, 2040, 2040}
	if acc != want {
		t.Errorf("UDOT gave %v, want %v", acc, want)
	}
}

// TestNoDotProdSkipsRatherThanTraps records what happens on an ARMv8.0 part:
// the tests above skip, and nothing in the package has executed an SDOT. Run
// this file under QEMU_CPU=cortex-a53 and a SIGILL means a kernel reached for
// the instruction without checking dotprod first.
func TestNoDotProdSkipsRatherThanTraps(t *testing.T) {
	if dotprod {
		t.Skip("this machine has FEAT_DotProd")
	}
	t.Log("no FEAT_DotProd: SDOT paths must be unreachable")
}

func sdotProbe(acc *int32, a, b *int8)

func udotProbe(acc *uint32, a, b *uint8)
