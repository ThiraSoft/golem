package tha3

import (
	"math"

	"github.com/ThiraSoft/golem/nn"
)

// instanceNormEps is PyTorch's default for InstanceNorm2d, which every block
// here uses.
const instanceNormEps = 1e-5

// InstanceNorm is InstanceNorm2d(affine=True) without running statistics:
// each channel of each picture is brought to mean 0 and variance 1 over its
// own pixels, then scaled and shifted.
type InstanceNorm struct {
	Weight, Bias []float32
}

// Apply normalizes x in place. The statistics are summed in float64: the
// bottleneck planes are small, the upper ones are 512×512, and a float32 sum
// over a quarter million values drifts further than PyTorch's does.
func (n InstanceNorm) Apply(x Tensor) {
	size := x.H * x.W
	nn.InParallel(x.C, x.C*size*3, func(start, end int) {
		for c := start; c < end; c++ {
			p := x.Plane(c)
			var mean float64
			for _, v := range p {
				mean += float64(v)
			}
			mean /= float64(size)
			var variance float64
			for _, v := range p {
				d := float64(v) - mean
				variance += d * d
			}
			variance /= float64(size)
			scale := float32(1/math.Sqrt(variance+instanceNormEps)) * n.Weight[c]
			shift := n.Bias[c] - float32(mean)*scale
			for i, v := range p {
				p[i] = v*scale + shift
			}
		}
	})
}
