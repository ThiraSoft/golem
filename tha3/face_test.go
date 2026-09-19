package tha3

import "testing"

func TestFaceMorpher(t *testing.T) {
	for _, run := range runs {
		t.Run(run, func(t *testing.T) {
			f := loadFixtures(t, run)
			w := openNetwork(t, netFaceMorpher)
			m := newFaceMorpher(w)
			if err := w.err(); err != nil {
				t.Fatal(err)
			}
			n := netFaceMorpher
			m.forward(f.tensor(t, n+".in.0"), f.pose(t, n+".in.1"), f.checker(t, n))
		})
	}
}
