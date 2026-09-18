package tha3

import "testing"

func TestEyebrowDecomposer(t *testing.T) {
	f := loadFixtures(t, "neutral")
	d := newEyebrowDecomposer(openNetwork(t, netEyebrowDecomposer))
	d.forward(f.tensor(t, netEyebrowDecomposer+".in.0"), f.checker(t, netEyebrowDecomposer))
}

func TestEyebrowCombiner(t *testing.T) {
	for _, run := range runs {
		t.Run(run, func(t *testing.T) {
			f := loadFixtures(t, run)
			c := newEyebrowCombiner(openNetwork(t, netEyebrowCombiner))
			n := netEyebrowCombiner
			c.forward(f.tensor(t, n+".in.0"), f.tensor(t, n+".in.1"), f.pose(t, n+".in.2"), f.checker(t, n))
		})
	}
}
