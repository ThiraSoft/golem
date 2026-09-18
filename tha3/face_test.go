package tha3

import "testing"

func TestFaceMorpher(t *testing.T) {
	for _, run := range runs {
		t.Run(run, func(t *testing.T) {
			f := loadFixtures(t, run)
			m := newFaceMorpher(openNetwork(t, netFaceMorpher))
			n := netFaceMorpher
			m.forward(f.tensor(t, n+".in.0"), f.pose(t, n+".in.1"), f.checker(t, n))
		})
	}
}
