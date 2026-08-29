package nn

import (
	"encoding/binary"
	"math"
	"math/rand"
	"testing"
)

// The shell has to be the shell: 3961 points of even coordinate sum within a
// squared radius of forty, which is 24·Σσ_odd(n) by D4's theta series. A point
// miscounted is a rate misreported and a code that means two things.
func TestD4ShellIsTheLattice(t *testing.T) {
	if got := D4Points(); got != 3961 {
		t.Fatalf("%d points, want 3961", got)
	}
	if bits := math.Log2(float64(D4Points())); bits > D4Bits {
		t.Fatalf("%.2f bits an index, over the %d the format spends", bits, D4Bits)
	}
	if got := D4Points() * 4 * 2; got > 32<<10 {
		t.Fatalf("decode table of %d bytes, over a workgroup's shared memory", got)
	}
	seen := map[[4]int8]bool{}
	for i := 0; i < D4Points(); i++ {
		p := D4Point(uint16(i))
		if seen[p] {
			t.Fatalf("point %v appears twice", p)
		}
		seen[p] = true
		sum := int(p[0]) + int(p[1]) + int(p[2]) + int(p[3])
		if sum&1 != 0 {
			t.Fatalf("point %v has odd coordinate sum", p)
		}
		n2 := int(p[0])*int(p[0]) + int(p[1])*int(p[1]) + int(p[2])*int(p[2]) + int(p[3])*int(p[3])
		if n2 > D4Radius {
			t.Fatalf("point %v has squared norm %d, past the shell", p, n2)
		}
		if c, ok := D4Code(p); !ok || int(c) != i {
			t.Fatalf("point %v codes back to %d, not %d", p, c, i)
		}
	}
}

// Twelve-bit codes packed two to three bytes, and read back.
func TestD4CodePacking(t *testing.T) {
	r := rand.New(rand.NewSource(7))
	codes := make([]uint16, 16)
	buf := make([]byte, 24)
	for trial := 0; trial < 500; trial++ {
		for i := range codes {
			codes[i] = uint16(r.Intn(D4Points()))
		}
		PutD4Codes(buf, codes)
		for i, want := range codes {
			if got := d4CodeAt(buf, i); got != want {
				t.Fatalf("code %d came back %d, want %d", i, got, want)
			}
		}
	}
}

// A block written by hand and read by the dequantiser: the scale multiplies the
// point, and the four coordinates of a code land on four consecutive weights.
func TestD4GBlockRoundTrip(t *testing.T) {
	r := rand.New(rand.NewSource(3))
	block := make([]byte, d4BlockBytes)
	scale := float32(0.0125)
	binary.LittleEndian.PutUint16(block[0:2], floatToHalf(scale))
	codes := make([]uint16, 16)
	for i := range codes {
		codes[i] = uint16(r.Intn(D4Points()))
	}
	PutD4Codes(block[2:], codes)

	out := make([]float32, D4Block)
	DequantizeD4G(block, D4Block, out)
	s := halfToFloat(floatToHalf(scale))
	for i, c := range codes {
		p := D4Point(c)
		for j := 0; j < 4; j++ {
			want := float32(p[j]) * s
			if got := out[i*4+j]; got != want {
				t.Fatalf("weight %d is %g, want %g", i*4+j, got, want)
			}
		}
	}
}

// The rotation has to be its own inverse up to the sign vector it is folded
// with, or the weights and the activations disagree about which space they are
// in and the product is quietly wrong.
func TestPrepareD4GIsOrthogonal(t *testing.T) {
	const n, group = 256, 128
	r := rand.New(rand.NewSource(11))
	x := make([]float32, n)
	pre := make([]float32, n)
	for i := range x {
		x[i] = r.Float32()*2 - 1
		pre[i] = 1
		if r.Intn(2) == 0 {
			pre[i] = -1
		}
	}
	before := make([]float32, n)
	copy(before, x)

	PrepareD4G(x, pre, group)
	// A norm the transform must preserve, group by group.
	for base := 0; base < n; base += group {
		var a, b float64
		for i := base; i < base+group; i++ {
			a += float64(before[i]) * float64(before[i])
			b += float64(x[i]) * float64(x[i])
		}
		if math.Abs(a-b) > 1e-3*a {
			t.Fatalf("group at %d: norm² %g became %g", base, a, b)
		}
	}
}

// More than one block, so the split planes are actually exercised: with a
// single block the two layouts coincide and a reader that still thinks the
// scale sits next to its codes would pass.
func TestD4GPlanesAreSplit(t *testing.T) {
	const n = D4Block * 5
	r := rand.New(rand.NewSource(21))
	row := make([]byte, n/D4Block*d4BlockBytes)
	scales, codes := D4Planes(row, n)
	if len(scales) != 10 || len(codes) != 120 {
		t.Fatalf("planes are %d and %d bytes, want 10 and 120", len(scales), len(codes))
	}
	want := make([]float32, n)
	for b := 0; b < 5; b++ {
		step := float32(b+1) * 0.01
		binary.LittleEndian.PutUint16(scales[b*2:], floatToHalf(step))
		cs := make([]uint16, 16)
		for i := range cs {
			cs[i] = uint16(r.Intn(D4Points()))
		}
		PutD4Codes(codes[b*24:], cs)
		s := halfToFloat(floatToHalf(step))
		for i, c := range cs {
			p := D4Point(c)
			for j := 0; j < 4; j++ {
				want[b*D4Block+i*4+j] = float32(p[j]) * s
			}
		}
	}
	got := make([]float32, n)
	DequantizeD4G(row, n, got)
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("weight %d is %g, want %g", i, got[i], want[i])
		}
	}
}
