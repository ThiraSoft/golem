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

// The plan UseVulkan builds has no traces, so its tensors live shorter and
// share more of the arena than in the traced plan TestCardMatchesCPU checks.
// This runs the shipped plan: the picture set before the card is taken, two
// poses a picture, SetImage again for every run, frames compared.
func TestCardShippedPlan(t *testing.T) {
	dev, err := vk.Open()
	if err != nil {
		t.Skipf("no Vulkan device: %v", err)
	}
	dev.Close()
	cpu := openPoserOrSkip(t)
	defer cpu.Close()
	card := openPoserOrSkip(t)
	defer card.Close()
	for i, run := range runs {
		f := loadFixtures(t, run)
		img := f.tensor(t, "image")
		if err := cpu.SetImage(img); err != nil {
			t.Fatal(err)
		}
		if err := card.SetImage(img); err != nil {
			t.Fatal(err)
		}
		if i == 0 {
			if err := card.UseVulkan(); err != nil {
				t.Fatal(err)
			}
		}
		var first, second [NumParams]float32
		copy(first[:], f.Pose)
		second[MouthOoo], second[HeadX] = 0.7, -0.4
		for _, pose := range [][NumParams]float32{first, second} {
			want, err := cpu.Pose(pose)
			if err != nil {
				t.Fatal(err)
			}
			got, err := card.Pose(pose)
			if err != nil {
				t.Fatal(err)
			}
			reference.Compare(t, run+" frame", got.Data, want.Data, tolerance)
		}
	}
}
