package tha3

import (
	"testing"

	"github.com/ThiraSoft/golem/internal/kyutai/reference"
)

// End to end: the recorded picture and pose in, every network's inputs and
// outputs compared on the way, the frame last.
func TestPoser(t *testing.T) {
	for _, run := range runs {
		t.Run(run, func(t *testing.T) {
			f := loadFixtures(t, run)
			p, err := Open(Dir())
			if err != nil {
				t.Skipf("weights not found (%v): see ref/tha3/README.md", err)
			}
			defer p.Close()
			p.trace = f.checker(t, "")
			if err := p.SetImage(f.tensor(t, "image")); err != nil {
				t.Fatal(err)
			}
			var pose [NumParams]float32
			copy(pose[:], f.Pose)
			frame, err := p.Pose(pose)
			if err != nil {
				t.Fatal(err)
			}
			reference.Compare(t, "frame", frame.Data, f.tensor(t, netEditor+".out.0").Data, tolerance)
		})
	}
}

func TestPoseBeforeImage(t *testing.T) {
	p, err := Open(Dir())
	if err != nil {
		t.Skipf("weights not found (%v)", err)
	}
	defer p.Close()
	if _, err := p.Pose([NumParams]float32{}); err == nil {
		t.Fatal("Pose without a picture did not fail")
	}
}
