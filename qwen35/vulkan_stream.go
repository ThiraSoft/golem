package qwen35

// Calibrating a checkpoint larger than the card.
//
// A conversion has to measure what every matrix is fed, and the honest thing to
// measure is the checkpoint being converted — which is BF16, four bytes a
// weight once widened, and fifty-four gigabytes for Qwen3.8-27B against a card
// that holds sixteen. The stack in vulkan.go uploads the whole trunk and cannot
// answer that.
//
// It does not have to. A block is a pure function of the hidden states that
// enter it: give it every position at once and it needs nothing of the blocks
// after it, and nothing of the blocks before it beyond the states they already
// produced. So the model can go past the card a window at a time — upload a
// window, push all the positions through it, keep the states it produced, drop
// it — and the host only ever holds one array of hidden states, which for two
// thousand positions of a five-thousand-wide model is forty megabytes.
//
// This pays because the pass is wide. A weight that crosses the bus serves one
// multiply per column, so streaming costs the bus divided by the batch: at one
// column it is the whole cost and generation would run at half a token a
// second, and at two thousand it is a hundredth of the arithmetic and free. The
// break-even is the ratio of the two bandwidths — device memory against the bus,
// which is about a hundred here. A calibration is on the right side of it; token
// generation never is, which is why this is a converter's path and not an
// engine's.

import (
	"fmt"
	"math"
	"runtime"
	"sync"

	"github.com/ThiraSoft/golem/nn"
	"github.com/ThiraSoft/golem/vk"
)

// wideOf is a matrix in the form the window's kernels read, which is the
// checkpoint's own form when the card has a kernel for it and float32
// otherwise. Every quantized form goes through nn.Matrix.Row, which is what
// every other reader of a checkpoint uses, so there is no second decoder to
// disagree with it.
//
// **The bfloat16 case returns the checkpoint's bytes and touches nothing.**
// Widening every matrix used to read the 27B's 54.8 GB and write and send
// 109.6 — the shift that puts a bfloat's bits in a float's top half, done on
// this side sixty times a window. shaders/matvec_f32.comp does the same shift
// now, so the host allocates nothing and the bus carries half.
func wideOf(w nn.Matrix, form vk.WideForm) []byte {
	if w.Quant == nn.F32 || (form == vk.WideBF16 && w.Quant == nn.BF16) {
		return w.Data
	}
	out := make([]float32, w.Rows*w.Cols)
	spread(w.Rows, func(lo, hi int) {
		for r := lo; r < hi; r++ {
			w.Row(r, out[r*w.Cols:(r+1)*w.Cols])
		}
	})
	return f32Bytes(out)
}

// wideForm is the form every projection of the trunk will be uploaded in.
//
// It is bfloat16 only when every one of them already is. The kernel is chosen
// once for the whole pipeline, so a trunk that mixed the two would have to
// widen the odd matrix out to a kernel that reads pairs of weights to a word —
// there is no such thing, and the answer is to widen all of them instead. The
// decay projections are excluded because they are float32 in every checkpoint
// and vk keeps the float kernel for them whatever the rest is.
func (m *Model) wideForm() vk.WideForm {
	if wideF32Only {
		return vk.WideF32
	}
	for i := range m.W.Blocks[:m.trunk()] {
		bw := &m.W.Blocks[i]
		for _, w := range []nn.Matrix{bw.Gate, bw.Up, bw.Down, bw.Q, bw.K, bw.V, bw.O,
			bw.QKV, bw.AttnGate, bw.SSMOut} {
			if w.Rows*w.Cols > 0 && w.Quant != nn.BF16 {
				return vk.WideF32
			}
		}
	}
	return vk.WideBF16
}

// wideF32Only makes wideForm answer WideF32 whatever the checkpoint holds,
// which is the path every window took before the card had a bfloat16 kernel:
// the host widens each matrix and sends twice its size.
//
// It is here for one test. The two forms compute the same numbers — widening a
// bfloat16 to a float is a shift, and the kernel does the shift the host used
// to — so the only way to hold the newer path to the older one is to run both,
// and the only way to run the older one is to ask for it.
var wideF32Only bool

// wideBytes is how many bytes a weight occupies in the given form.
func wideBytes(form vk.WideForm) int {
	if form == vk.WideBF16 {
		return 2
	}
	return 4
}

func spread(n int, fn func(lo, hi int)) {
	p := runtime.GOMAXPROCS(0)
	if p > n {
		p = n
	}
	if p < 2 {
		fn(0, n)
		return
	}
	var wg sync.WaitGroup
	chunk := (n + p - 1) / p
	for lo := 0; lo < n; lo += chunk {
		hi := min(lo+chunk, n)
		wg.Add(1)
		go func(lo, hi int) { defer wg.Done(); fn(lo, hi) }(lo, hi)
	}
	wg.Wait()
}

func f32Bytes(v []float32) []byte {
	out := make([]byte, len(v)*4)
	spread(len(v), func(lo, hi int) {
		for i := lo; i < hi; i++ {
			b := math.Float32bits(v[i])
			out[i*4+0] = byte(b)
			out[i*4+1] = byte(b >> 8)
			out[i*4+2] = byte(b >> 16)
			out[i*4+3] = byte(b >> 24)
		}
	})
	return out
}

// streamShape is the geometry of the whole trunk, whatever window is resident.
// It is the model's and not a window's: a window of delta nets alone would
// otherwise leave the attention side of it at zero and build a pipeline that
// cannot record the next window.
func (m *Model) streamShape(ctx int) vk.QwenShape {
	cfg := m.Cfg
	s := vk.QwenShape{
		// A window never drafts: it is a converter's path over a wide batch,
		// and there is no prediction block in it to refuse a column.
		Snapshots:    false,
		Dim:          cfg.Dim,
		FFN:          cfg.Blocks[0].FFN,
		MaxContext:   ctx,
		Eps:          cfg.Eps,
		RoPESections: [4]int(cfg.RoPESections),
		Float:        m.wideForm(),
	}
	for _, bc := range cfg.Blocks[:m.trunk()] {
		if bc.Type == BlockFullAttn && s.Heads == 0 {
			s.Heads, s.KVHeads = bc.Heads, bc.KVHeads
			s.HeadDim, s.RoPEDims = bc.HeadDim, bc.RoPEDims
			s.RoPEBase = float32(bc.RoPEBase)
		}
		if bc.Type == BlockSSM && s.Rank == 0 {
			s.ConvDim = bc.SSMGroupCount*bc.SSMStateSize*2 + bc.SSMInnerSize
			s.Inner, s.Rank = bc.SSMInnerSize, bc.SSMTimeStepRank
			s.StateSize, s.Groups = bc.SSMStateSize, bc.SSMGroupCount
		}
	}
	return s
}

// streamShare is how much of the card a window may occupy.
//
// It is a share of what is *free*, not of the heap's size, wherever the driver
// will say — vk's DeviceLocalFree, which is VK_EXT_memory_budget. The
// difference is not academic on this machine: two thirds of the heap is 10.6
// GiB, and what is actually free moves between 14.9 and 15.6 GiB depending on
// the desktop and on what the previous window has not finished releasing. The
// same constant therefore meant seventy-one per cent of the card on one run and
// sixty-eight on another, which is how a test that holds for an hour loses the
// device on the next run for no reason it can name.
//
// The rest is the margin: the driver's own allocations, and the fragmentation
// between one window's buffers and the next's. Three quarters of the heap left
// it at its ceiling and lost the device; two thirds of what is free is a
// tighter claim and a safer one.
const streamShare = 2.0 / 3.0

// buildWindow uploads blocks from `from` onward, in the form wideForm chooses,
// and stops when the
// next one would put the pipeline past the budget. It answers the pipeline and
// the block after the last one it took, so the caller walks the trunk by asking
// rather than by arithmetic.
//
// The budget is the card's, asked of the driver. A window is at least one block
// whatever it says: a model whose single block does not fit has no streamed
// answer either, and failing on the submission with the real number in hand is
// a better error than refusing with a computed one.
func (m *Model) buildWindow(from, ctx, want int) (*vk.QwenPipeline, int, error) {
	d, err := m.device()
	if err != nil {
		return nil, 0, err
	}
	trunk := m.trunk()
	// What the driver says is left, and the heap's size only when it will not
	// say. A window is sized against the card it is about to be built on rather
	// than against the card's specification.
	room := d.DeviceLocalFree()
	if room == 0 {
		room = d.DeviceLocalBytes()
	}
	budget := uint64(float64(room) * streamShare)
	shape := m.streamShape(ctx)
	form := shape.Float
	pipe, err := vk.NewQwenPipeline(d, shape)
	if err != nil {
		return nil, 0, fmt.Errorf("qwen35: cannot create the window's pipeline: %w", err)
	}
	var attnNorms, ffnNorms [][]float32
	var isSSM []bool
	to := from
	for i := from; i < trunk; i++ {
		if i > from {
			// Whether the block after this one still fits, judged on what the
			// pipeline actually holds and what a block of this model weighs.
			if want > 0 && i-from >= want {
				break
			}
			if want == 0 && pipe.DeviceBytes()+uint64(m.StreamBlockBytes()) > budget {
				break
			}
		}
		bc := m.Cfg.Blocks[i]
		bw := &m.W.Blocks[i]
		attnNorms = append(attnNorms, bw.AttnNorm)
		ffnNorms = append(ffnNorms, bw.FFNNorm)
		isSSM = append(isSSM, bc.Type != BlockFullAttn)

		if err := pipe.AddFFNBlock(vk.QwenFFNData{
			Gate: wideOf(bw.Gate, form), Up: wideOf(bw.Up, form), Down: wideOf(bw.Down, form),
		}); err != nil {
			pipe.Close()
			return nil, 0, fmt.Errorf("qwen35: block %d feed forward: %w", i, err)
		}
		// The window's own numbering, not the model's: the pipeline files a
		// block at the index it is given and record walks its blocks from zero.
		// What the model calls this block is carried by SetBlockWindow, which
		// is what the calibration files its sites under.
		at := i - from
		if bc.Type == BlockFullAttn {
			err = pipe.AddAttnBlock(at, vk.QwenAttnData{
				WQ: wideOf(bw.Q, form), WK: wideOf(bw.K, form), WV: wideOf(bw.V, form), WO: wideOf(bw.O, form),
				QNorm: bw.QNorm, KNorm: bw.KNorm,
			})
		} else {
			err = pipe.AddSSMBlock(at, vk.QwenSSMData{
				WQKV:  wideOf(bw.QKV, form),
				WGate: wideOf(bw.AttnGate, form),
				// The decay's two, always float32 on the card. See the comment
				// beside setAlpha in vk/qwen_pipeline.go.
				WAlpha:     wideOf(bw.SSMAlpha, vk.WideF32),
				WBeta:      wideOf(bw.SSMBeta, vk.WideF32),
				WOut:       wideOf(bw.SSMOut, form),
				Out:        nn.F32,
				ConvWeight: bw.Conv1D,
				SSMA:       bw.SSMA,
				SSMDtBias:  bw.SSMDtBias,
				SSMNorm:    bw.SSMNorm,
			})
		}
		if err != nil {
			pipe.Close()
			return nil, 0, fmt.Errorf("qwen35: block %d mixer: %w", i, err)
		}
		to = i + 1
	}
	if to == from {
		pipe.Close()
		return nil, 0, fmt.Errorf("qwen35: no block was taken at %d", from)
	}
	if err := pipe.SetNorms(attnNorms, ffnNorms, m.W.OutputNorm, isSSM); err != nil {
		pipe.Close()
		return nil, 0, fmt.Errorf("qwen35: window %d-%d norms: %w", from, to, err)
	}
	return pipe, to, nil
}

// CalibrateStreamed measures every site of the model with `window` blocks on
// the card at a time, reading the checkpoint as it is rather than a quantized
// twin of it. It answers the per-site sums and how many rows they were taken
// over, which is what VulkanCalibrationSums answers for a resident stack.
//
// runs are independent contexts — a corpus cut into windows, each its own
// conversation — and every one of them goes through a resident window before
// the next window is uploaded. The other order would upload the model once per
// run, which is the whole cost of this path paid again for nothing.
//
// The states between model windows live on the host, because they are the only
// thing a later block reads. Two thousand positions of a five-thousand-wide
// model is forty megabytes of them.
func (m *Model) CalibrateStreamed(runs [][]int32, window, ctx int) (map[string][]float32, int, error) {
	trunk := m.trunk()
	if trunk == 0 {
		return nil, 0, fmt.Errorf("qwen35: the model has no blocks to stream")
	}
	rows := 0
	for _, ids := range runs {
		if len(ids) > ctx {
			return nil, 0, fmt.Errorf("qwen35: a run of %d positions past the %d these windows are built for", len(ids), ctx)
		}
		rows += len(ids)
	}
	if rows == 0 {
		return nil, 0, fmt.Errorf("qwen35: nothing to calibrate on")
	}

	// The embedding, on the host, which is where it already happens: the card's
	// stack has always been given hidden states rather than tokens.
	dim := m.Cfg.Dim
	xs := make([][][]float32, len(runs))
	for r, ids := range runs {
		xs[r] = make([][]float32, len(ids))
		for i, id := range ids {
			xs[r][i] = make([]float32, dim)
			m.W.TokenEmbd.Row(int(id), xs[r][i])
		}
	}

	sums := map[string][]float32{}
	for from := 0; from < trunk; {
		pipe, to, err := m.buildWindow(from, ctx, window)
		if err != nil {
			return nil, 0, err
		}
		pipe.SetBlockWindow(from, to == trunk)
		if err := pipe.StartCalibration(); err != nil {
			pipe.Close()
			return nil, 0, fmt.Errorf("qwen35: window %d-%d: %w", from, to, err)
		}
		for _, run := range xs {
			// Each run is its own context: the caches are written before they
			// are read, so only the delta nets' state has to be forgotten.
			if err := pipe.ResetState(); err != nil {
				pipe.Close()
				return nil, 0, err
			}
			// Every position in order, in passes as wide as the pipeline has a
			// binary for. The order is the delta net's business: its state
			// walks the positions, so a window sees them exactly once and
			// exactly in sequence, which is what a resident stack does too.
			for t := 0; t < len(run); {
				n := pipe.WidthFor(len(run) - t)
				at := make([]vk.QwenPlace, n)
				for c := 0; c < n; c++ {
					p := t + c
					at[c] = vk.QwenPlace{Pos: p, T: p, H: p, W: p}
				}
				if _, err := pipe.ForwardPlaces(run[t:t+n], at); err != nil {
					pipe.Close()
					return nil, 0, fmt.Errorf("qwen35: window %d-%d at position %d: %w", from, to, t, err)
				}
				// What this window made is what the next one reads: the state
				// before the output norm, because that norm belongs to the end
				// of the model and this is the middle of it.
				for c := 0; c < n; c++ {
					copy(run[t+c], pipe.HiddenColumn(c))
				}
				t += n
			}
		}
		pipe.CountCalibration(rows)
		got, _, err := pipe.CalibrationSums()
		if err != nil {
			pipe.Close()
			return nil, 0, err
		}
		for k, v := range got {
			sums[k] = append([]float32(nil), v...)
		}
		fmt.Printf("  blocks %d-%d, %d MiB on the card\n", from, to, pipe.DeviceBytes()>>20)
		pipe.Close()
		from = to
	}
	return sums, rows, nil
}

// FloatWeights says the checkpoint keeps its projections in the form it was
// trained in, which is not an inference form: four bytes a weight, or two, on a
// card that reads a quantized one at half of one. Such a checkpoint reaches the
// card only through CalibrateStreamed, a window at a time.
//
// A bfloat16 one now goes up as it is — shaders/matvec_f32.comp has a -DBF16
// build and vk.WideBF16 selects it — where it used to be widened to float32 on
// this side. A float32 one is still sent as it is, and every quantized form
// still reaches this path widened.
func (m *Model) FloatWeights() bool {
	q := m.W.Blocks[0].Down.Quant
	return q == nn.BF16 || q == nn.F32
}

// StreamBlockBytes is what one block of this model occupies on the card in the
// form it is uploaded in — two bytes a weight for a bfloat16 checkpoint and four
// for anything else — which is what decides how many of them a window may hold.
func (m *Model) StreamBlockBytes() int {
	bw := &m.W.Blocks[0]
	n := 0
	for _, w := range []nn.Matrix{bw.Gate, bw.Up, bw.Down, bw.Q, bw.K, bw.V, bw.O,
		bw.QKV, bw.AttnGate, bw.SSMAlpha, bw.SSMBeta, bw.SSMOut} {
		n += w.Rows * w.Cols
	}
	// The widest block of the two kinds, not the first one: a trunk alternates
	// them and a window sized on the smaller would not hold the larger.
	if len(m.W.Blocks) > 1 {
		other := &m.W.Blocks[1]
		m2 := 0
		for _, w := range []nn.Matrix{other.Gate, other.Up, other.Down, other.Q, other.K, other.V, other.O,
			other.QKV, other.AttnGate, other.SSMAlpha, other.SSMBeta, other.SSMOut} {
			m2 += w.Rows * w.Cols
		}
		n = max(n, m2)
	}
	return n * wideBytes(m.wideForm())
}

// ForwardStreamed runs every position of every run through the model a window
// of blocks at a time and answers what the head reads: the state after the
// output norm, one row a position, in the order they were given.
//
// It is CalibrateStreamed without the accumulators and with the last window's
// answer kept instead of thrown away. Both exist because the same mechanism
// serves two questions — what the matrices are fed, and what the model says —
// and only the second needs the states to come back.
//
// The head is not here. It is one matrix and the host already owns it: Logits
// reads the same state a resident stack hands back, so a caller writes the same
// line whichever path produced it.
func (m *Model) ForwardStreamed(runs [][]int32, window, ctx int) ([][][]float32, error) {
	trunk := m.trunk()
	if trunk == 0 {
		return nil, fmt.Errorf("qwen35: the model has no blocks to stream")
	}
	for _, ids := range runs {
		if len(ids) > ctx {
			return nil, fmt.Errorf("qwen35: a run of %d positions past the %d these windows are built for", len(ids), ctx)
		}
	}

	dim := m.Cfg.Dim
	xs := make([][][]float32, len(runs))
	out := make([][][]float32, len(runs))
	for r, ids := range runs {
		xs[r] = make([][]float32, len(ids))
		out[r] = make([][]float32, len(ids))
		for i, id := range ids {
			xs[r][i] = make([]float32, dim)
			m.W.TokenEmbd.Row(int(id), xs[r][i])
		}
	}

	for from := 0; from < trunk; {
		pipe, to, err := m.buildWindow(from, ctx, window)
		if err != nil {
			return nil, err
		}
		last := to == trunk
		for r, run := range xs {
			if err := pipe.ResetState(); err != nil {
				pipe.Close()
				return nil, err
			}
			for t := 0; t < len(run); {
				n := pipe.WidthFor(len(run) - t)
				at := make([]vk.QwenPlace, n)
				for c := 0; c < n; c++ {
					p := t + c
					at[c] = vk.QwenPlace{Pos: p, T: p, H: p, W: p}
				}
				hs, err := pipe.ForwardPlaces(run[t:t+n], at)
				if err != nil {
					pipe.Close()
					return nil, fmt.Errorf("qwen35: window %d-%d at position %d: %w", from, to, t, err)
				}
				for c := 0; c < n; c++ {
					if last {
						// What the head reads, which is the only thing the last
						// window is asked for.
						out[r][t+c] = append([]float32(nil), hs[c]...)
					} else {
						// What the next window reads: the state before the
						// output norm, because that norm belongs to the end of
						// the model and this is the middle of it.
						copy(run[t+c], pipe.HiddenColumn(c))
					}
				}
				t += n
			}
		}
		pipe.Close()
		from = to
	}
	return out, nil
}
