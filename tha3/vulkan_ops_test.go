package tha3

import (
	"math/rand"
	"testing"

	"github.com/ThiraSoft/golem/internal/kyutai/reference"
	"github.com/ThiraSoft/golem/vk"
)

// cardTolerance is for one operation on the card against the same operation
// on the processor: the card fuses multiply-adds and sums in another order,
// and nothing else should differ.
const cardTolerance = 1e-4

// onCard describes a graph over the given inputs, runs it once as a pose
// pass, and returns its one output. It skips when there is no device.
func onCard(t *testing.T, pose []float32, inputs []Tensor, build func(g *vk.THA3Graph, in []vk.THA3Tensor) vk.THA3Tensor) Tensor {
	t.Helper()
	dev, err := vk.Open()
	if err != nil {
		t.Skipf("no Vulkan device: %v", err)
	}
	defer dev.Close()
	g := vk.NewTHA3Graph()
	in := make([]vk.THA3Tensor, len(inputs))
	for i, x := range inputs {
		in[i] = g.Input(x.C, x.H, x.W)
	}
	g.SetPhase(vk.THA3PosePhase)
	out := build(g, in)
	g.Output(out)
	run, err := g.Build(dev, false)
	if err != nil {
		t.Fatal(err)
	}
	defer run.Close()
	for i, x := range inputs {
		if err := run.Write(in[i], x.Data); err != nil {
			t.Fatal(err)
		}
	}
	var p [NumParams]float32
	copy(p[:], pose)
	run.SetPose(p[:])
	if _, err := run.RunPose(); err != nil {
		t.Fatal(err)
	}
	return Tensor{C: out.C, H: out.H, W: out.W, Data: run.Output(0)}
}

func checkCard(t *testing.T, name string, got, want Tensor) {
	t.Helper()
	if got.C != want.C || got.H != want.H || got.W != want.W {
		t.Fatalf("%s: %dx%dx%d, want %dx%dx%d", name, got.C, got.H, got.W, want.C, want.H, want.W)
	}
	reference.Compare(t, name, got.Data, want.Data, cardTolerance)
}

// unitTensor is uniform in [0, 1), what an alpha holds.
func unitTensor(r *rand.Rand, c, h, w int) Tensor {
	t := NewTensor(c, h, w)
	for i := range t.Data {
		t.Data[i] = r.Float32()
	}
	return t
}

func TestCardCopies(t *testing.T) {
	r := rand.New(rand.NewSource(10))
	x := randomTensor(r, 3, 10, 12)
	y := randomTensor(r, 3, 4, 5)
	z := randomTensor(r, 2, 10, 12)
	checkCard(t, "crop", onCard(t, nil, []Tensor{x}, func(g *vk.THA3Graph, in []vk.THA3Tensor) vk.THA3Tensor {
		return g.Crop(in[0], 2, 3, 4, 5)
	}), x.Crop(2, 3, 4, 5))
	pasted := x.Clone()
	pasted.Paste(y, 5, 6)
	checkCard(t, "paste", onCard(t, nil, []Tensor{x, y}, func(g *vk.THA3Graph, in []vk.THA3Tensor) vk.THA3Tensor {
		return g.Paste(in[0], in[1], 5, 6)
	}), pasted)
	checkCard(t, "concat", onCard(t, nil, []Tensor{x, z}, func(g *vk.THA3Graph, in []vk.THA3Tensor) vk.THA3Tensor {
		return g.Concat(in[0], in[1])
	}), Concat(x, z))
	checkCard(t, "upsample", onCard(t, nil, []Tensor{y}, func(g *vk.THA3Graph, in []vk.THA3Tensor) vk.THA3Tensor {
		return g.Upsample2(in[0])
	}), UpsampleNearest2(y))
	checkCard(t, "channels", onCard(t, nil, []Tensor{x}, func(g *vk.THA3Graph, in []vk.THA3Tensor) vk.THA3Tensor {
		return g.Crop(in[0].Channels(1, 3), 0, 0, 10, 12)
	}), x.Channels(1, 3).Clone())
}

func TestCardPose(t *testing.T) {
	r := rand.New(rand.NewSource(11))
	pose := random(r, NumParams)
	got := onCard(t, pose, nil, func(g *vk.THA3Graph, _ []vk.THA3Tensor) vk.THA3Tensor {
		return g.PoseSlice(12, 27, 5, 6)
	})
	checkCard(t, "pose", got, Broadcast(pose[12:39], 5, 6))
}

func TestCardBlends(t *testing.T) {
	r := rand.New(rand.NewSource(12))
	image, change := randomTensor(r, 4, 9, 11), randomTensor(r, 4, 9, 11)
	alpha1, alpha4 := unitTensor(r, 1, 9, 11), unitTensor(r, 4, 9, 11)
	for _, a := range []Tensor{alpha1, alpha4} {
		got := onCard(t, nil, []Tensor{a, change, image}, func(g *vk.THA3Graph, in []vk.THA3Tensor) vk.THA3Tensor {
			return g.ColorChange(in[0], in[1], in[2])
		})
		checkCard(t, "color change", got, applyColorChange(a, change, image))
	}
	got := onCard(t, nil, []Tensor{alpha1, change, image}, func(g *vk.THA3Graph, in []vk.THA3Tensor) vk.THA3Tensor {
		return g.RGBChange(in[0], in[1], in[2])
	})
	checkCard(t, "rgb change", got, applyRGBChange(alpha1, change, image))
	half := change.Channels(3, 4).Clone()
	for i, v := range half.Data {
		half.Data[i] = (v + 1) / 2
	}
	got = onCard(t, nil, []Tensor{change, image}, func(g *vk.THA3Graph, in []vk.THA3Tensor) vk.THA3Tensor {
		return g.RGBHalfAlpha(in[0], in[1])
	})
	checkCard(t, "half alpha", got, applyRGBChange(half, change, image))
	got = onCard(t, nil, []Tensor{alpha1, change, image}, func(g *vk.THA3Graph, in []vk.THA3Tensor) vk.THA3Tensor {
		return g.Sharpen(in[0], in[1], in[2], 0.6)
	})
	checkCard(t, "sharpen", got, applySharpen(alpha1, change, image, 0.6))
	got = onCard(t, nil, []Tensor{alpha1}, func(g *vk.THA3Graph, in []vk.THA3Tensor) vk.THA3Tensor {
		return g.Steepen(in[0], steepenBy(0.6))
	})
	checkCard(t, "steepen", got, steepen(alpha1, 0.6))
}

func TestCardResize(t *testing.T) {
	r := rand.New(rand.NewSource(13))
	x := randomTensor(r, 3, 64, 48)
	for _, size := range [][2]int{{32, 24}, {100, 90}} {
		got := onCard(t, nil, []Tensor{x}, func(g *vk.THA3Graph, in []vk.THA3Tensor) vk.THA3Tensor {
			return g.Resize(in[0], size[0], size[1])
		})
		checkCard(t, "resize", got, ResizeBilinear(x, size[0], size[1]))
	}
}

func TestCardWarp(t *testing.T) {
	r := rand.New(rand.NewSource(14))
	image := randomTensor(r, 4, 20, 24)
	grid, add := NewTensor(2, 20, 24), NewTensor(2, 20, 24)
	for i := range grid.Data {
		grid.Data[i] = (r.Float32()*2 - 1) * 0.3
		add.Data[i] = (r.Float32()*2 - 1) * 0.1
	}
	// Far outside the picture, so the border clamp is exercised.
	grid.Data[0], grid.Data[7], grid.Data[500] = 5, -5, 3
	got := onCard(t, nil, []Tensor{image, grid}, func(g *vk.THA3Graph, in []vk.THA3Tensor) vk.THA3Tensor {
		return g.Warp(in[0], in[1])
	})
	checkCard(t, "warp", got, applyGridChange(grid, image))
	sum := grid.Clone()
	for i, v := range add.Data {
		sum.Data[i] += v
	}
	got = onCard(t, nil, []Tensor{image, grid, add}, func(g *vk.THA3Graph, in []vk.THA3Tensor) vk.THA3Tensor {
		return g.WarpAdd(in[0], in[1], in[2])
	})
	checkCard(t, "warp add", got, applyGridChange(sum, image))
}
