package nn

import "testing"

// The table a kernel indexes by coordinates has to give the same answer the
// map does, in both directions and for the points that are not there at all.
func TestD4InverseTableAgreesWithTheMap(t *testing.T) {
	for _, bits := range []int{D4Bits, D4Bits16} {
		lim, side, table := D4InverseTable(bits)
		if got := side * side * side * side; got != len(table) {
			t.Fatalf("%d bits: a table of %d for a side of %d", bits, len(table), side)
		}
		coded := 0
		for a := -lim; a <= lim; a++ {
			for b := -lim; b <= lim; b++ {
				for c := -lim; c <= lim; c++ {
					for d := -lim; d <= lim; d++ {
						pt := [4]int8{int8(a), int8(b), int8(c), int8(d)}
						at := 0
						for _, v := range pt {
							at = at*side + int(v) + lim
						}
						want, ok := D4CodeN(pt, bits)
						got := table[at]
						if !ok {
							if got != D4NoCode {
								t.Fatalf("%d bits: %v is not in the tier and the table says %d", bits, pt, got)
							}
							continue
						}
						coded++
						if got != uint32(want) {
							t.Fatalf("%d bits: %v is %d and the table says %d", bits, pt, want, got)
						}
					}
				}
			}
		}
		if coded != 1<<bits {
			t.Errorf("%d bits: the table holds %d of the tier's %d points", bits, coded, 1<<bits)
		}
	}
}

// Every point of the tier fits inside the box the table is cut to — which is
// what lets a coordinate past it be refused without a lookup.
func TestEveryPointOfATierFitsTheBox(t *testing.T) {
	for _, bits := range []int{D4Bits, D4Bits16} {
		lim, _, _ := D4InverseTable(bits)
		for i := 0; i < 1<<bits; i++ {
			for _, v := range D4Point(uint16(i)) {
				if int(v) > lim || int(v) < -lim {
					t.Fatalf("%d bits: point %d has a coordinate of %d, past %d", bits, i, v, lim)
				}
			}
		}
	}
}

func TestD4StepTableIsWhatD4StepReturns(t *testing.T) {
	tab := D4StepTable()
	if len(tab) != 256 {
		t.Fatalf("%d steps", len(tab))
	}
	for c := 0; c < 256; c++ {
		if tab[c] != D4Step(byte(c)) {
			t.Fatalf("step %d is %v in the table and %v from D4Step", c, tab[c], D4Step(byte(c)))
		}
	}
}
