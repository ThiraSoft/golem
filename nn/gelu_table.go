package nn

// GELU as ggml's CPU backend computes it, which is not GELU.
//
// ggml does not evaluate the formula: it rounds the input to fp16, uses the
// sixteen bits as an index into a table of 65536 precomputed fp16 results, and
// widens the answer back to float32. Two roundings, so about three decimal
// digits of agreement with the closed form — and a gap of that size on the feed
// forward output is a gap in which a real mistake would be invisible.
//
// The table is built once, on first use: 65536 entries of four bytes, a
// quarter of a megabyte, against a model of three gigabytes.

import (
	"math"
	"sync"
)

var (
	geluOnce  sync.Once
	geluTable [1 << 16]float32
)

// geluTanh is the approximation ggml tabulates.
func geluTanh(x float64) float64 {
	const sqrt2OverPi = 0.79788456080286535587989211986876
	const coefA = 0.044715
	return 0.5 * x * (1 + math.Tanh(sqrt2OverPi*x*(1+coefA*x*x)))
}

func buildGELUTable() {
	for i := 0; i < 1<<16; i++ {
		in := halfToFloat(uint16(i))
		out := float32(geluTanh(float64(in)))
		geluTable[i] = halfToFloat(floatToHalf(out))
	}
}

// GELUTable applies ggml's GELU in place.
func GELUTable(x []float32) {
	geluOnce.Do(buildGELUTable)
	for i, v := range x {
		switch {
		case v <= -10:
			x[i] = 0
		case v >= 10:
			// left alone: GELU(x) is x to well beyond fp16's precision here
		default:
			x[i] = geluTable[floatToHalf(v)]
		}
	}
}
