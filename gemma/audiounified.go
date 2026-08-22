package gemma

// The other projector: no encoder at all.
//
// gemma4ua cuts the waveform into frames of 640 samples — forty milliseconds
// at sixteen kilohertz — RMS-normalizes each frame across its own 640 values,
// and projects it once into the model's width. Every block those rows go
// through afterwards is the language model's, which is why there is nothing
// else in this file. The 12B listens this way; E2B has the conformer.
//
// The last frame is padded with silence rather than dropped: a recording is
// rarely a whole number of frames long, and the alternative is losing up to
// forty milliseconds off the end of every one of them.

import "github.com/ThiraSoft/golem/nn"

func (a *AudioTower) encodeUnified(samples []float32) [][]float32 {
	cfg := a.Cfg
	frame := cfg.FrameSize
	if len(samples) == 0 {
		return nil
	}
	n := (len(samples) + frame - 1) / frame

	framed := make([]float32, n*frame)
	copy(framed, samples)
	nn.InParallel(n, n*frame, func(first, last int) {
		for p := first; p < last; p++ {
			nn.RMSNormPlain(framed[p*frame:(p+1)*frame], nil, cfg.Eps)
		}
	})

	tmp := make([]float32, n*frame)
	out := make([]float32, n*cfg.ProjDim)
	a.W.MMProj.Apply(framed, tmp, out, n)

	rows := make([][]float32, n)
	for p := range rows {
		rows[p] = out[p*cfg.ProjDim : (p+1)*cfg.ProjDim]
	}
	return rows
}
