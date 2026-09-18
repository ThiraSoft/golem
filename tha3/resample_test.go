package tha3

import (
	"math"
	"testing"
)

func TestInstanceNorm(t *testing.T) {
	x := Tensor{C: 2, H: 1, W: 4, Data: []float32{1, 2, 3, 4, 5, 5, 5, 5}}
	InstanceNorm{Weight: []float32{2, 1}, Bias: []float32{1, 3}}.Apply(x)
	// Channel 0: mean 2.5, biased variance 1.25.
	s := 2 / math.Sqrt(1.25+1e-5)
	want := []float32{float32(1 - 1.5*s), float32(1 - 0.5*s), float32(1 + 0.5*s), float32(1 + 1.5*s), 3, 3, 3, 3}
	closeTo(t, "instance norm", x.Data, want, 1e-5)
}

// PyTorch: F.interpolate(torch.tensor([[[[0.,1.],[2.,3.]]]]), size=(4,4),
// mode="bilinear", align_corners=False).
func TestResizeBilinearUp(t *testing.T) {
	x := Tensor{C: 1, H: 2, W: 2, Data: []float32{0, 1, 2, 3}}
	got := ResizeBilinear(x, 4, 4)
	want := []float32{
		0, 0.25, 0.75, 1,
		0.5, 0.75, 1.25, 1.5,
		1.5, 1.75, 2.25, 2.5,
		2, 2.25, 2.75, 3,
	}
	closeTo(t, "bilinear up", got.Data, want, 1e-6)
}

// Halving without antialias averages each 2×2 block, since every output
// centre falls exactly between four inputs.
func TestResizeBilinearDown(t *testing.T) {
	x := seq(1, 4, 4)
	got := ResizeBilinear(x, 2, 2)
	closeTo(t, "bilinear down", got.Data, []float32{2.5, 4.5, 10.5, 12.5}, 1e-6)
}

func TestUpsampleNearest2(t *testing.T) {
	got := UpsampleNearest2(Tensor{C: 1, H: 1, W: 2, Data: []float32{1, 2}})
	closeTo(t, "nearest", got.Data, []float32{1, 1, 2, 2, 1, 1, 2, 2}, 0)
}

func TestGridChangeIdentity(t *testing.T) {
	x := seq(2, 5, 6)
	got := applyGridChange(NewTensor(2, 5, 6), x)
	closeTo(t, "identity", got.Data, x.Data, 1e-5)
}

// An offset of one pixel to the right, 2/W in normalized units, reads each
// pixel's right neighbour; the last column reads past the border, which
// clamps to the last column.
func TestGridChangeShift(t *testing.T) {
	x := seq(1, 2, 4)
	change := NewTensor(2, 2, 4)
	for i := range change.Plane(0) {
		change.Plane(0)[i] = 2.0 / 4
	}
	got := applyGridChange(change, x)
	closeTo(t, "shift", got.Data, []float32{1, 2, 3, 3, 5, 6, 7, 7}, 1e-5)
}
