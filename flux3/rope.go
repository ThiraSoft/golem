package flux3

import (
	"math"
)

// Rotary position embeddings for FLUX 3 Image Klein.
// EmbedND uses 4 axes (t, row, col, l) of 32 dimensions each, theta 10000,
// and image coordinates scaled by 0.5.

const (
	RopeTheta   = 10000.0
	RopeHeadDim = 128
	RopeAxes    = 4
	RopeAxisDim = 32
)

// ImageIDs generates 4D position IDs (t, row, col, l) for an image of h × w patches.
// Reference images are distinguished by negative indices on the l axis (-1 - i).
func ImageIDs(h, w, index int) []float32 {
	ids := make([]float32, h*w*4)
	idx := 0
	for r := 0; r < h; r++ {
		for c := 0; c < w; c++ {
			ids[idx+0] = 0.0
			ids[idx+1] = float32(r) * 0.5
			ids[idx+2] = float32(c) * 0.5
			ids[idx+3] = float32(index)
			idx += 4
		}
	}
	return ids
}

// RefIDs generates 4D position IDs for reference image shapes at l = -1 - i.
func RefIDs(shapes [][2]int) []float32 {
	var ids []float32
	for i, s := range shapes {
		ids = append(ids, ImageIDs(s[0], s[1], -1-i)...)
	}
	return ids
}

// TextIDs generates 4D position IDs for text tokens (0, 0, 0, i).
func TextIDs(length int) []float32 {
	ids := make([]float32, length*4)
	for i := 0; i < length; i++ {
		ids[i*4+0] = 0.0
		ids[i*4+1] = 0.0
		ids[i*4+2] = 0.0
		ids[i*4+3] = float32(i)
	}
	return ids
}

// EmbedPE computes the [1, N, 1, 64, 2, 2] position embeddings matching ComfyUI's
// EmbedND / Flux3.rope. For each token and each of the 64 pairs, it produces a 2x2
// rotation matrix [[cos, -sin], [sin, cos]], formatted as 4 consecutive floats.
func EmbedPE(ids []float32) []float32 {
	tokens := len(ids) / 4
	out := make([]float32, tokens*64*4)
	for tok := 0; tok < tokens; tok++ {
		base := tok * 256
		pairIdx := 0
		for axis := 0; axis < RopeAxes; axis++ {
			pos := float64(ids[tok*4+axis])
			for i := 0; i < RopeAxisDim/2; i++ {
				scale := float64(2*i) / float64(RopeAxisDim)
				omega := 1.0 / math.Pow(RopeTheta, scale)
				arg := pos * omega
				cos := float32(math.Cos(arg))
				sin := float32(math.Sin(arg))

				m := base + pairIdx*4
				out[m+0] = cos
				out[m+1] = -sin
				out[m+2] = sin
				out[m+3] = cos
				pairIdx++
			}
		}
	}
	return out
}

// RopeTable computes the [N, 128] table for vk.K2Rope: for each token, 64 cosines
// followed by 64 sines.
func RopeTable(ids []float32) []float32 {
	tokens := len(ids) / 4
	out := make([]float32, tokens*RopeHeadDim)
	for tok := 0; tok < tokens; tok++ {
		base := tok * RopeHeadDim
		pairIdx := 0
		for axis := 0; axis < RopeAxes; axis++ {
			pos := float64(ids[tok*4+axis])
			for i := 0; i < RopeAxisDim/2; i++ {
				scale := float64(2*i) / float64(RopeAxisDim)
				omega := 1.0 / math.Pow(RopeTheta, scale)
				arg := pos * omega
				out[base+pairIdx] = float32(math.Cos(arg))
				out[base+64+pairIdx] = float32(math.Sin(arg))
				pairIdx++
			}
		}
	}
	return out
}
