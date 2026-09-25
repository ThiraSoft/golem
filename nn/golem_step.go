package nn

// The step grid every golem tier shares: 128 weights coded as one path, and a
// step code a sixty-four of them.
//
// A step code is eight bits naming a power of two, a sixteenth of an octave
// apart — the spacing the scheme has always used, and a coarser one costs a
// whole point of perplexity that 0.03 dB of weight error could not see. A step
// is the block's RMS itself, fitted by least squares after the path is chosen,
// and the grid sits two octaves higher and wider at the top than a lattice's
// would, for the reason the measurement gave: a unit-variance source clipped at
// the lattice's ceiling reconstructed at 5.5 dB instead of 22.6.

import (
	"math"
	"sync/atomic"
)

const (
	// GolemSeq is how many weights are coded as one path. It is the effective
	// dimension of the codebook, and it is set by what the encoder's
	// backpointers fit in rather than by anything the decoder cares about.
	GolemSeq = 128
	// GolemBlock is how many weights share one step code.
	GolemBlock = 64
	// golemStepsPerSeq is how many step codes a sequence carries.
	golemStepsPerSeq = GolemSeq / GolemBlock
)

// golemStepBias places the 256 codes on the exponent axis. The grid is
// 2^((c-224)/16), which runs from 6.1e-5 to 3.83 — sixteen octaves at four
// percent a step.
//
// Where it sits is measured, not chosen. The block RMS of every matrix of
// Qwen3-0.6B, rotated, spans 0.0011 to 0.27, eight octaves; the salience scales
// columns by at most 24 either way before the rotation mixes them, and the
// remaining headroom is for that. A block that lands on either end is counted
// and said out loud rather than clipped quietly — see GolemStepCode — because a
// step that saturates is exactly the failure this format has already produced
// once, at 5.5 dB on a source whose RMS was one.
const golemStepBias = 224

var golemSteps [256]float32

func init() {
	for c := range golemSteps {
		golemSteps[c] = float32(math.Exp2((float64(c) - golemStepBias) / 16))
	}
}

// GolemStepClipped counts the blocks whose step landed on an end of the grid.
// Zero on every model this has been pointed at; anything else means the window
// is in the wrong place for that checkpoint, and the file is worse than the
// codec it was written with.
var GolemStepClipped atomic.Int64

// GolemStepCode is the code nearest a step. A step at or below zero is the
// smallest code: it belongs to a block that is all but zero, and no path
// through the trellis reaches further down than the grid does.
func GolemStepCode(v float32) byte {
	if !(v > 0) {
		return 0
	}
	c := math.Round(math.Log2(float64(v))*16 + golemStepBias)
	if c < 0 {
		GolemStepClipped.Add(1)
		c = 0
	}
	if c > 255 {
		GolemStepClipped.Add(1)
		c = 255
	}
	return byte(c)
}

// GolemStep expands a step code.
func GolemStep(c byte) float32 { return golemSteps[c] }

// GolemStepBias is where the grid sits, for a kernel that expands a code with an
// exp2 rather than a table.
const GolemStepBias = golemStepBias
