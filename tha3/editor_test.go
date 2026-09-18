package tha3

import "testing"

func TestEditor(t *testing.T) {
	for _, run := range runs {
		t.Run(run, func(t *testing.T) {
			f := loadFixtures(t, run)
			w := openNetwork(t, netEditor)
			e := newEditor(w)
			if err := w.err(); err != nil {
				t.Fatal(err)
			}
			n := netEditor
			e.forward(f.tensor(t, n+".in.0"), f.tensor(t, n+".in.1"), f.tensor(t, n+".in.2"),
				f.pose(t, n+".in.3"), f.checker(t, n))
		})
	}
}
