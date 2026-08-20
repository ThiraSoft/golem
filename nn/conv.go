package nn

// One-dimensional convolutions, in streaming mode.
//
// The audio decoder never sees the whole utterance: it receives one frame of
// latents at a time and must produce the corresponding samples, with no
// discontinuity at the seam. Each convolution therefore keeps the little
// context the next one will need — the last input samples for a direct
// convolution, the not-yet-complete output tail for a transposed one. That
// state, and only that state, is what distinguishes these convolutions from
// ordinary ones.
//
// Signals are laid out channel by channel: `x[c*T+t]`, as in PyTorch.

// Conv1d is a causal convolution in streaming mode.
type Conv1d struct {
	Weights  []float32 // Outputs x Inputs x Kernel
	Bias     []float32
	Inputs   int
	Outputs  int
	Kernel   int
	Stride   int
	Dilation int
}

// ConvState holds a convolution's context between two calls, and the buffers
// the call works in.
//
// The shapes are fixed for the life of a decoder — the same number of steps
// reaches the same layer on every frame — so allocating them per call bought
// nothing but six megabytes of zeroing and of garbage per frame. The zeroing is
// the smaller half of the bill: the collector's background marking takes a
// processor away from a pool whose eight workers then all wait, at the next
// barrier, for the one that lost it. The buffers grow on demand and never
// shrink.
type ConvState struct {
	previous []float32 // Inputs x (effective kernel - stride)
	keep     int

	input  []float32 // Inputs x (keep + steps)
	output []float32 // Outputs x outputs
	window []float32 // outputs x (Inputs x Kernel), for the gathered form
}

// grow returns exactly n floats out of buf, reusing its storage when it is
// large enough. The contents are not cleared: every caller below writes each
// element before reading it.
func grow(buf []float32, n int) []float32 {
	if cap(buf) >= n {
		return buf[:n]
	}
	return make([]float32, n)
}

func (c Conv1d) effectiveKernel() int { return (c.Kernel-1)*c.Dilation + 1 }

// NewState allocates the context, initialized to zero — the constant padding
// the decoder uses.
func (c Conv1d) NewState() *ConvState {
	keep := c.effectiveKernel() - c.Stride
	if keep < 0 {
		keep = 0
	}
	return &ConvState{previous: make([]float32, c.Inputs*keep), keep: keep}
}

// Apply processes `steps` positions and returns the output, channel by channel.
func (c Conv1d) Apply(x []float32, steps int, state *ConvState) ([]float32, int) {
	total := state.keep + steps
	state.input = grow(state.input, c.Inputs*total)
	input := state.input
	for e := 0; e < c.Inputs; e++ {
		copy(input[e*total:], state.previous[e*state.keep:(e+1)*state.keep])
		copy(input[e*total+state.keep:], x[e*steps:(e+1)*steps])
	}

	outputs := (total-c.effectiveKernel())/c.Stride + 1
	if outputs < 0 {
		outputs = 0
	}
	state.output = grow(state.output, c.Outputs*outputs)
	y := state.output

	if c.gathered(outputs) {
		c.applyGathered(input, y, total, outputs, state)
	} else {
		c.applyAxpy(input, y, total, outputs)
	}
	c.keepTail(input, total, state)
	return y, outputs
}

// applyAxpy sweeps the output with unit stride for a fixed coefficient: it is
// the only arrangement where both arrays are read contiguously. The variants
// that accumulate in a register position by position look thriftier, but
// scatter the input reads over as many cache lines as there are channels, and
// cost a fifth more.
func (c Conv1d) applyAxpy(input, y []float32, total, outputs int) {
	InParallel(c.Outputs, c.Outputs*c.Inputs*c.Kernel*outputs, func(start, end int) {
		for s := start; s < end; s++ {
			var bias float32
			if c.Bias != nil {
				bias = c.Bias[s]
			}
			row := y[s*outputs : (s+1)*outputs]
			block := c.Weights[s*c.Inputs*c.Kernel : (s+1)*c.Inputs*c.Kernel]
			for i := range row {
				row[i] = bias
			}
			for e := 0; e < c.Inputs; e++ {
				kernel := block[e*c.Kernel : (e+1)*c.Kernel]
				channel := input[e*total : (e+1)*total]
				for k, weight := range kernel {
					if weight == 0 {
						continue
					}
					offset := k * c.Dilation
					if c.Stride == 1 {
						Axpy(row, channel[offset:], weight)
						continue
					}
					for i := range row {
						row[i] += weight * channel[i*c.Stride+offset]
					}
				}
			}
		}
	})
}

// keepTail sets aside exactly what the next window will re-read.
func (c Conv1d) keepTail(input []float32, total int, state *ConvState) {
	for e := 0; e < c.Inputs; e++ {
		copy(state.previous[e*state.keep:(e+1)*state.keep], input[(e+1)*total-state.keep:(e+1)*total])
	}
}

// gathered says which of the two arrangements to use.
//
// The axpy form above makes one call per (output channel, input channel, tap)
// and sweeps the output row inside it, so it costs Inputs*Kernel calls per
// output channel and the row length is what amortizes them. The decoder has
// layers where that row is sixteen floats long and Inputs*Kernel is three
// thousand five hundred: the 512-channel input convolution makes 1.8 million
// calls per frame to multiply-add sixteen values each, and spends more time
// entering and leaving the kernel than inside it.
//
// The other arrangement gathers the window each output position reads into one
// contiguous vector — the usual im2col — after which the convolution is a dot
// product of that vector against a row of weights, which is already contiguous.
// It costs one call per (output channel, output position) instead, and pays for
// it with one pass to build the window and a horizontal reduction at the end of
// each dot product.
//
// So the question is only whether the output row is long enough to amortize an
// axpy. Measured on the decoder's own shapes, the crossover is a row of a few
// tens of positions: at sixteen the gathered form is six times faster, at
// ninety-six it is already ten percent slower, and at four hundred and eighty
// it is not close. Thirty-two is the round number inside that gap, and it puts
// the two layers that need it — the 512-channel input convolution and the
// projection — on one side and everything nearer the audio rate on the other.
const shortRow = 32

func (c Conv1d) gathered(outputs int) bool {
	return outputs > 0 && outputs <= shortRow && outputs < c.Inputs*c.Kernel
}

// applyGathered computes the output through the gathered window described
// above. The window buffer is built once for the whole layer and read by every
// output channel, which is why it is not built inside the parallel section.
func (c Conv1d) applyGathered(input, y []float32, total, outputs int, state *ConvState) {
	width := c.Inputs * c.Kernel
	state.window = grow(state.window, outputs*width)
	col := state.window
	for e := 0; e < c.Inputs; e++ {
		channel := input[e*total : (e+1)*total]
		if c.Dilation == 1 {
			for i := 0; i < outputs; i++ {
				copy(col[i*width+e*c.Kernel:i*width+(e+1)*c.Kernel], channel[i*c.Stride:])
			}
			continue
		}
		for i := 0; i < outputs; i++ {
			tap := col[i*width+e*c.Kernel:]
			for k := 0; k < c.Kernel; k++ {
				tap[k] = channel[i*c.Stride+k*c.Dilation]
			}
		}
	}

	InParallel(c.Outputs, c.Outputs*width*outputs, func(start, end int) {
		for s := start; s < end; s++ {
			var bias float32
			if c.Bias != nil {
				bias = c.Bias[s]
			}
			row := y[s*outputs : (s+1)*outputs]
			block := c.Weights[s*width : (s+1)*width]
			for i := range row {
				row[i] = bias + DotF32(block, col[i*width:(i+1)*width])
			}
		}
	})
}

// ConvTranspose1d widens the signal: it is the one that climbs from the frame
// rate back up to the sample rate.
type ConvTranspose1d struct {
	Weights []float32 // Inputs x (Outputs/Groups) x Kernel
	Bias    []float32
	Inputs  int
	Outputs int
	Kernel  int
	Stride  int
	Groups  int

	// Packed is the same weights in the order the gathered form reads them:
	// output channel, then phase, then input channel, then tap. Prepare fills
	// it; when it is nil the polyphase axpy form below is used instead.
	Packed []float32
}

// taps is how many weights of one phase land on one output position: the depth
// of the small correlation a transposed convolution becomes once it is written
// phase by phase.
func (c ConvTranspose1d) taps() int { return (c.Kernel + c.Stride - 1) / c.Stride }

// Prepare builds the packed weights the gathered form needs, if that form is
// the one this shape wants for a frame of `steps` positions. It is called once,
// when the layer is loaded, by a caller that knows how many positions will
// reach it.
//
// Written phase by phase, a transposed convolution of stride S is S ordinary
// convolutions of depth ceil(K/S) whose outputs interleave. That is the whole
// content of the rearrangement: phase p of output channel o reads the taps
// p, p+S, p+2S, ... of every input channel, and once those are contiguous the
// phase is a dot product like any other. The weights are only permuted — for
// every shape in the decoder K is exactly 2S, so not one float is padding.
func (c *ConvTranspose1d) Prepare(steps int) {
	if !c.gathered(steps) {
		return
	}
	outputsPerGroup := c.Outputs / c.Groups
	inputsPerGroup := c.Inputs / c.Groups
	m := c.taps()
	width := inputsPerGroup * m
	packed := make([]float32, c.Outputs*c.Stride*width)
	for s := 0; s < c.Outputs; s++ {
		group := s / outputsPerGroup
		for phase := 0; phase < c.Stride; phase++ {
			block := packed[(s*c.Stride+phase)*width:]
			for i := 0; i < inputsPerGroup; i++ {
				e := group*inputsPerGroup + i
				base := (e*outputsPerGroup + s%outputsPerGroup) * c.Kernel
				for t := 0; t < m; t++ {
					if k := phase + t*c.Stride; k < c.Kernel {
						block[i*m+t] = c.Weights[base+k]
					}
				}
			}
		}
	}
	c.Packed = packed
}

// windowLimit is how large the gathered window is allowed to get, in floats.
//
// Every output channel reads the whole window, so the form is only worth
// anything while the window stays in the second-level cache; past that each
// worker streams a quarter of a megabyte per channel out of the shared one and
// the arrangement loses to the axpy form it replaced. Measured on the two
// expansions that sit either side of the line: the second, whose window is
// fifty thousand floats, goes from 650 to 400 microseconds; the third, whose
// window is a hundred and twenty thousand, goes from 430 to 845.
const windowLimit = 64 << 10

// gathered says whether this shape wants the packed form.
//
// Two things have to hold. The window has to fit in cache, which is what
// windowLimit says. And the dot product has to run over enough input channels
// to be worth entering: a grouped convolution with one channel per group — the
// sixteen-fold upsampling, where every channel is expanded on its own — would
// be trading one call per tap for one call per pair of taps, and it keeps the
// axpy form.
func (c ConvTranspose1d) gathered(steps int) bool {
	if steps <= 0 || c.Stride <= 0 || c.Groups <= 0 ||
		c.Outputs%c.Groups != 0 || c.Inputs%c.Groups != 0 ||
		c.Inputs/c.Groups < shortRow {
		return false
	}
	return (steps+c.taps())*c.Inputs*c.taps() <= windowLimit
}

// interleave writes one output row from the `stride` phase buffers that were
// computed separately. The output is written contiguously and the phases are
// read as that many sequential streams, which the prefetcher follows without
// trouble. The bias is added here, on the one pass that touches every value.
func interleave(row, buffer []float32, nj, stride int, bias float32) {
	for j := 0; j < nj; j++ {
		block := row[j*stride:]
		if len(block) > stride {
			block = block[:stride]
		}
		for phase := range block {
			block[phase] = buffer[phase*nj+j] + bias
		}
	}
}

// ConvTState holds the output tail that the next call must complete: the last
// positions written are not final yet, since the following inputs will add to
// them.
type ConvTState struct {
	partial []float32 // Outputs x (Kernel - Stride)
	keep    int

	// The raw output, before the previous call's tail is folded into it, and
	// the positions that have become final. Kept for the same reason as
	// Conv1d's buffers: the shapes never change, so neither should the
	// allocation.
	raw    []float32
	out    []float32
	window []float32 // nj x (Inputs x taps), for the gathered form
}

func (c ConvTranspose1d) NewState() *ConvTState {
	keep := c.Kernel - c.Stride
	if keep < 0 {
		keep = 0
	}
	return &ConvTState{partial: make([]float32, c.Outputs*keep), keep: keep}
}

// Apply processes `steps` positions and returns the positions that have become
// final.
func (c ConvTranspose1d) Apply(x []float32, steps int, state *ConvTState) ([]float32, int) {
	raw := (steps-1)*c.Stride + c.Kernel
	state.raw = grow(state.raw, c.Outputs*raw)
	y := state.raw

	outputsPerGroup := c.Outputs / c.Groups
	inputsPerGroup := c.Inputs / c.Groups

	// Polyphase form, written as separate phases. Output position o only
	// receives the weights whose index is congruent to o modulo the stride;
	// within one phase, o = phase + j*Stride and the contribution of channel[t]
	// falls at j = t + k/Stride. By laying each phase out in its own contiguous
	// buffer, the inner loop becomes a plain dst += a*src again — vectorizable —
	// instead of one write every Stride floats. The final interleave costs only
	// one pass over the output, against one per weight before.
	nj := (raw + c.Stride - 1) / c.Stride

	// The window every phase reads: for output index j, the inputs at
	// j, j-1, ... j-(taps-1), zero where that falls outside the frame. It is
	// built once for the whole layer, like Conv1d's.
	var col []float32
	var colWidth, taps int
	if c.Packed != nil && c.gathered(steps) {
		taps = c.taps()
		colWidth = c.Inputs * taps
		state.window = grow(state.window, nj*colWidth)
		col = state.window
		clear(col)
		for e := 0; e < c.Inputs; e++ {
			channel := x[e*steps : (e+1)*steps]
			for j := 0; j < nj; j++ {
				slot := col[j*colWidth+e*taps:]
				for t := 0; t < taps; t++ {
					if u := j - t; u >= 0 && u < steps {
						slot[t] = channel[u]
					}
				}
			}
		}
	}

	InParallel(c.Outputs, c.Outputs*inputsPerGroup*c.Kernel*steps, func(start, end int) {
		buffer := make([]float32, c.Stride*nj)
		for s := start; s < end; s++ {
			group := s / outputsPerGroup
			row := y[s*raw : (s+1)*raw]
			var bias float32
			if c.Bias != nil {
				bias = c.Bias[s]
			}
			if col != nil {
				// One dot product per (phase, output position), against the
				// taps of that phase laid out contiguously by Prepare.
				block := c.Packed[s*c.Stride*inputsPerGroup*taps:]
				window := group * inputsPerGroup * taps
				for phase := 0; phase < c.Stride; phase++ {
					w := block[phase*inputsPerGroup*taps:][:inputsPerGroup*taps]
					dst := buffer[phase*nj : (phase+1)*nj]
					for j := range dst {
						dst[j] = DotF32(w, col[j*colWidth+window:])
					}
				}
				interleave(row, buffer, nj, c.Stride, bias)
				continue
			}
			// Cleared to zero rather than to the bias: that is the pattern the
			// compiler recognizes and replaces with a vectorized memory clear.
			// The bias is added during the interleave, which passes over every
			// value anyway.
			clear(buffer)
			for e := group * inputsPerGroup; e < (group+1)*inputsPerGroup; e++ {
				base := (e*outputsPerGroup + s%outputsPerGroup) * c.Kernel
				kernel := c.Weights[base : base+c.Kernel]
				channel := x[e*steps : (e+1)*steps]
				// phase and offset follow k by simple increment. Obtaining them
				// through k%Stride and k/Stride cost two integer divisions per
				// coefficient — on a channel a few positions long, more
				// expensive than the multiply-accumulate itself. offset stays
				// bounded by (Kernel-1)/Stride, and nj is
				// steps-1+ceil(Kernel/Stride): the target slice always contains
				// the whole channel, so no guard is needed.
				phase, offset := 0, 0
				for _, weight := range kernel {
					if weight != 0 {
						// A channel of one or two positions — the upsampling
						// sees exactly one — is not worth entering a kernel
						// for: the call is the whole cost.
						if len(channel) < 8 {
							dst := buffer[phase*nj+offset:]
							for i, v := range channel {
								dst[i] += weight * v
							}
						} else {
							AxpyFull(buffer[phase*nj+offset:], channel, weight)
						}
					}
					if phase++; phase == c.Stride {
						phase, offset = 0, offset+1
					}
				}
			}
			interleave(row, buffer, nj, c.Stride, bias)
		}
	})

	if state.keep == 0 {
		return y, raw
	}

	// The first `keep` positions receive the tail left by the previous call;
	// the last `keep` become the new tail. The bias must count only once: we
	// subtract it from what we set aside.
	final := raw - state.keep
	state.out = grow(state.out, c.Outputs*final)
	out := state.out
	for s := 0; s < c.Outputs; s++ {
		row := y[s*raw : (s+1)*raw]
		tail := state.partial[s*state.keep : (s+1)*state.keep]
		for i := 0; i < state.keep && i < raw; i++ {
			row[i] += tail[i]
		}
		var bias float32
		if c.Bias != nil {
			bias = c.Bias[s]
		}
		for i := 0; i < state.keep; i++ {
			tail[i] = row[final+i] - bias
		}
		copy(out[s*final:(s+1)*final], row[:final])
	}
	return out, final
}
