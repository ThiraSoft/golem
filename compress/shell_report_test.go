package compress

import (
	"math"
	"testing"
)

// What a kernel decoding on the fly would have to hold.
//
// This is the constraint that picks the lattice, and it is not visible from the
// error curves at all. A code is an index into the shell, and expanding it is a
// table lookup: 2·d bytes an entry, and the table has to sit somewhere a
// workgroup can reach. E8 only fits one at about two bits a weight, which is
// well below where this model still answers; at the rate that works its shell
// holds thirteen million points, two hundred and sixteen mebibytes. D4 spends
// the same bits on a shell of four thousand — thirty-one kibibytes, which is
// shared memory. And the table is the lattice, so one table serves the whole
// model rather than one codebook per tensor.
func TestDecodeTableFitsSharedMemory(t *testing.T) {
	const sharedMemory = 32 << 10 // what a workgroup can be given on RDNA

	d4 := shellSize(LatD4, 40)
	if got := d4 * LatD4.Dim() * 2; got > sharedMemory {
		t.Errorf("D4 at r²=40: %d points, table of %d bytes, over the %d a workgroup has",
			d4, got, sharedMemory)
	}
	if bits := math.Log2(float64(d4)); bits > 12 {
		t.Errorf("D4 at r²=40 needs %.2f bits an index, over the 12 the format spends", bits)
	}

	// And the shape of the problem, so the next reader does not have to
	// rediscover why the lattice is not E8.
	for _, c := range []struct {
		l Lattice
		r float32
	}{{LatE8, 10}, {LatE8, 42}, {LatD4, 40}, {LatD4, 160}} {
		n := shellSize(c.l, c.r)
		bits := math.Log2(float64(n))
		t.Logf("%s r²=%-4g %10d points  %5.2f bits  %.3f bpw  table %9.1f KiB",
			c.l, c.r, n, bits, bits/float64(c.l.Dim()), float64(n*c.l.Dim()*2)/1024)
	}
}
