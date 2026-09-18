package tha3

import (
	"errors"
	"io/fs"
	"testing"

	"github.com/ThiraSoft/golem/internal/kyutai/reference"
	"github.com/ThiraSoft/golem/vk"
)

func openPoserOrSkip(t *testing.T) *Poser {
	t.Helper()
	p, err := Open(Dir())
	if errors.Is(err, fs.ErrNotExist) {
		t.Skipf("weights not found (%v): see ref/tha3/README.md", err)
	}
	if err != nil {
		t.Fatal(err)
	}
	return p
}

// The card against the processor, which is itself held to PyTorch: every
// waypoint the processor's tracer emits, on every recorded pose, in the
// order it was emitted, so the first one out of tolerance says where.
func TestCardMatchesCPU(t *testing.T) {
	dev, err := vk.Open()
	if err != nil {
		t.Skipf("no Vulkan device: %v", err)
	}
	dev.Close()
	cpu := openPoserOrSkip(t)
	defer cpu.Close()
	card := openPoserOrSkip(t)
	defer card.Close()
	if err := card.useVulkan(true); err != nil {
		t.Fatal(err)
	}
	for _, run := range runs {
		t.Run(run, func(t *testing.T) {
			f := loadFixtures(t, run)
			var names []string
			want := map[string]Tensor{}
			cpu.trace = func(name string, x Tensor) {
				names = append(names, name)
				want[name] = x.Clone()
			}
			img := f.tensor(t, "image")
			if err := cpu.SetImage(img); err != nil {
				t.Fatal(err)
			}
			if err := card.SetImage(img); err != nil {
				t.Fatal(err)
			}
			var pose [NumParams]float32
			copy(pose[:], f.Pose)
			cpuFrame, err := cpu.Pose(pose)
			if err != nil {
				t.Fatal(err)
			}
			cardFrame, err := card.Pose(pose)
			if err != nil {
				t.Fatal(err)
			}
			for _, name := range names {
				got := card.gpu.run.Waypoint(name)
				if got == nil {
					t.Fatalf("%s: the card traced nothing under that name", name)
				}
				reference.Compare(t, name, got, want[name].Data, tolerance)
				if t.Failed() {
					t.FailNow()
				}
			}
			reference.Compare(t, "frame", cardFrame.Data, cpuFrame.Data, tolerance)
		})
	}
}
