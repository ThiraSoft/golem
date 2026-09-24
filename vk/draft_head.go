package vk

import (
	_ "embed"
	"fmt"
	"unsafe"

	"github.com/ThiraSoft/golem/nn"
)

// The logit head read on the card behind the prediction block and behind a
// speculative pass, with the peak of what it writes found there too.
//
// A guess used to be the block on the card, its hidden state read back, the
// model's head run as a matrix of its own against it, a quarter of a million
// logits read back, and their peak found on the processor: on Bonsai 2 that
// was 1.8 milliseconds a guess, of which the block itself was 0.7. Here the
// head reads the block's output where it lies and one integer comes back.
//
// Two heads may be carried. The verifying one is the model's own, every row of
// it, because what it answers is the model's greedy token and nothing else will
// do. The guessing one may be fewer rows: a guess outside them is a guess not
// made, which costs acceptance and never correctness, and on a vocabulary of a
// quarter of a million the head is most of what a guess reads. ids says which
// token each of its rows is.

//go:generate glslc -O --target-env=vulkan1.1 -fshader-stage=compute shaders/argmax.comp -o shaders/argmax.spv
//go:embed shaders/argmax.spv
var argmaxSPIRV []byte

type draftHead struct {
	rows int
	// ids names the token of each row, and is nil for the model's whole head.
	ids []int32

	weights, signs, logits, best *Buffer
	pipeArg                      *Pipeline
	setQuant, setProd, setArg    *Set
	// gm is the head as a .golem matrix, read through act once prep has put
	// the hidden states into its form; nil for a head of llama.cpp's formats.
	gm   *GolemMatrix
	act  *Buffer
	prep *PrepareGolem

	// pass is DraftTokenAt's recording, and peaks PeaksOf's, one per width.
	pass  *Program
	peaks map[int]*Program
}

// peakColumns is the widest pass PeaksOf answers.
const peakColumns = 16

// AddDraftHead gives the pipeline a head in one of llama.cpp's formats. data
// is the matrix as the file holds it, rows by cols in format q; pre is the
// sign vector a Prism head rotates its input by, nil for a plain one. With ids
// nil it is the model's whole head and verifies as well as guesses; with ids it
// is a guessing head of those tokens only.
func (p *QwenPipeline) AddDraftHead(data []byte, q nn.Quant, rows, cols int, pre []float32, ids []int32) error {
	if err := p.checkHead(rows, cols, ids); err != nil {
		return err
	}
	pipe, err := p.quants.get(q)
	if err != nil {
		return err
	}
	layout, err := quantLayout(q, data, rows, cols)
	if err != nil {
		return err
	}
	h := &draftHead{rows: rows, ids: ids}
	fail := func(err error) error {
		h.close()
		return err
	}
	if h.weights, err = p.d.Upload(layout); err != nil {
		return fail(err)
	}
	if err := p.headBuffers(h, peakColumns); err != nil {
		return fail(err)
	}
	if pre != nil {
		if h.signs, err = p.upload(asBytes(pre)); err != nil {
			return fail(err)
		}
		if h.setQuant, err = p.rotPipes.Bind(p.stage, h.signs, p.normedQ, p.normedS); err != nil {
			return fail(err)
		}
	} else if h.setQuant, err = p.pipeQuant.NewSet([]*Buffer{p.stage, p.normedQ, p.normedS}); err != nil {
		return fail(err)
	}
	if h.setProd, err = pipe.NewSet([]*Buffer{h.weights, p.normedQ, p.normedS, h.logits}); err != nil {
		return fail(err)
	}
	p.install(h)
	return nil
}

// AddGolemDraftHead is AddDraftHead for the model's own .golem head, whose
// weights stay where they are and are read from here as well, by that head's
// kernels — five bits where the blocks are three or four. pre is the head's
// vector.
func (p *QwenPipeline) AddGolemDraftHead(head *GolemHead, pre []float32) error {
	return p.addGolemHead(head.k, head.m.weights, nil, head.rows, head.cols, pre, nil)
}

// AddGolemGuessHead is a guessing head of a .golem model: data holds the rows
// of the tokens ids names, in the head's format, which kernels k read.
func (p *QwenPipeline) AddGolemGuessHead(k *GolemKernels, data []byte, rows, cols int, pre []float32, ids []int32) error {
	if ids == nil {
		return fmt.Errorf("vk: a guessing head names its tokens")
	}
	return p.addGolemHead(k, nil, data, rows, cols, pre, ids)
}

func (p *QwenPipeline) addGolemHead(k *GolemKernels, weights *Buffer, data []byte, rows, cols int, pre []float32, ids []int32) error {
	if err := p.checkHead(rows, cols, ids); err != nil {
		return err
	}
	h := &draftHead{rows: rows, ids: ids}
	var err error
	fail := func(err error) error {
		h.close()
		return err
	}
	width := peakColumns
	if ids != nil {
		width = 1
	}
	if h.act, err = p.local(cols * 4 * width); err != nil {
		return fail(err)
	}
	if err := p.headBuffers(h, width); err != nil {
		return fail(err)
	}
	if h.prep, err = NewPrepareGolem(p.d, h.act, pre, prepareGolemGroup); err != nil {
		return fail(err)
	}
	if weights != nil {
		h.gm, err = newGolemMatrixShared(k, weights, rows, cols, h.act, h.logits)
	} else {
		h.gm, err = NewGolemMatrixOn(k, data, rows, cols, h.act, h.logits)
	}
	if err != nil {
		return fail(err)
	}
	p.install(h)
	return nil
}

func (p *QwenPipeline) checkHead(rows, cols int, ids []int32) error {
	if p.mtp == nil {
		return fmt.Errorf("vk: a draft head needs the prediction block")
	}
	if cols != p.shape.Dim {
		return fmt.Errorf("vk: a draft head reads %d inputs, the block writes %d", cols, p.shape.Dim)
	}
	if ids != nil && len(ids) != rows {
		return fmt.Errorf("vk: %d rows named by %d ids", rows, len(ids))
	}
	return nil
}

// headBuffers are the logits of width columns, their peaks, and the kernel
// that finds them.
func (p *QwenPipeline) headBuffers(h *draftHead, width int) error {
	var err error
	if h.logits, err = p.local(h.rows * 4 * width); err != nil {
		return err
	}
	if h.best, err = p.d.Readback(4*width, bufferUsageStorage); err != nil {
		return err
	}
	if h.pipeArg, err = p.d.NewPipeline(argmaxSPIRV, 2, 4); err != nil {
		return err
	}
	h.setArg, err = h.pipeArg.NewSet([]*Buffer{h.logits, h.best})
	return err
}

// install puts a head where it belongs: the whole head verifies, and guesses
// too unless a guessing head is also carried.
func (p *QwenPipeline) install(h *draftHead) {
	old := &p.draft
	if h.ids != nil {
		old = &p.guess
	}
	if *old != nil {
		(*old).close()
	}
	*old = h
}

// HasDraftHead says whether DraftTokenAt can answer.
func (p *QwenPipeline) HasDraftHead() bool { return p.draft != nil || p.guess != nil }

// DraftTokenAt runs the prediction block and a head behind it, and answers the
// token at the head's peak and the block's output-normed state, which the next
// guess of a chain reads as its hidden half.
func (p *QwenPipeline) DraftTokenAt(eh []float32, at QwenPlace) (int32, []float32, error) {
	s := p.shape
	h := p.guess
	if h == nil {
		h = p.draft
	}
	if h == nil {
		return 0, nil, fmt.Errorf("vk: the pipeline carries no draft head")
	}
	if len(eh) != s.Dim*2 {
		return 0, nil, fmt.Errorf("vk: the prediction block takes %d floats, given %d", s.Dim*2, len(eh))
	}
	if at.Pos >= s.MaxContext {
		return 0, nil, fmt.Errorf("vk: position %d is past the %d the pipeline was built for", at.Pos, s.MaxContext)
	}
	copy(p.mtp.ehIn.Floats(), eh)
	p.setPlace(at)
	if h.pass == nil {
		prog, err := p.d.Compile(func(r *Recorder) {
			p.recordMTP(r)
			p.recordHead(r, h, 1)
		})
		if err != nil {
			return 0, nil, err
		}
		h.pass = prog
	}
	if err := h.pass.Run(); err != nil {
		return 0, nil, err
	}
	row := *(*uint32)(unsafe.Pointer(&h.best.Bytes()[0]))
	tok := int32(row)
	if h.ids != nil {
		tok = h.ids[row]
	}
	return tok, p.stage.Floats()[:s.Dim], nil
}

// PeaksOf is the token at the head's peak for each of the last pass's first
// columns, found where the pass left its output-normed states: the model's
// greedy answer at every column of a speculative pass, with one integer a
// column read back instead of a row of logits.
func (p *QwenPipeline) PeaksOf(columns int) ([]int32, error) {
	h := p.draft
	if h == nil {
		return nil, fmt.Errorf("vk: the pipeline carries no whole head")
	}
	if columns < 1 || columns > peakColumns {
		return nil, fmt.Errorf("vk: peaks of one to %d columns, given %d", peakColumns, columns)
	}
	prog, ok := h.peaks[columns]
	if !ok {
		var err error
		if prog, err = p.d.Compile(func(r *Recorder) { p.recordHead(r, h, columns) }); err != nil {
			return nil, err
		}
		if h.peaks == nil {
			h.peaks = map[int]*Program{}
		}
		h.peaks[columns] = prog
	}
	if err := prog.Run(); err != nil {
		return nil, err
	}
	best := unsafe.Slice((*uint32)(unsafe.Pointer(&h.best.Bytes()[0])), columns)
	out := make([]int32, columns)
	for c := range out {
		out[c] = int32(best[c])
	}
	return out, nil
}

// recordHead is a head over the first columns of stage, and the peak of each.
func (p *QwenPipeline) recordHead(r *Recorder, h *draftHead, columns int) {
	s := p.shape
	if h.gm != nil {
		r.Copy(h.act, 0, p.stage, s.Dim*4*columns)
		r.Barrier()
		p.prepare(r, h.prep, columns)
		r.Barrier()
		p.golemProduct(r, h.gm, columns)
	} else {
		if h.signs != nil {
			p.rotate(r, h.setQuant, s.Dim, columns)
		} else {
			quant := swigluPush{N: uint32(s.Dim), Columns: uint32(columns)}
			r.Dispatch(h.setQuant, uint32((s.Dim/quantBlock*columns+255)/256), unsafe.Pointer(&quant))
			r.Barrier()
		}
		push := moePush{dim: uint32(h.rows), ffn: uint32(s.Dim), used: 1, split: 1}
		for at := 0; at < columns; {
			w := max(h.setProd.widest(columns-at), 1)
			if w < columns-at {
				if c := h.setProd.cover(columns-at, peakColumns-at); c > 0 {
					w = c
				}
			}
			push.col = uint32(at)
			if w == 1 {
				r.Dispatch(h.setProd, groupsOf(h.rows, matvecOuts), unsafe.Pointer(&push))
			} else {
				r.DispatchWide(h.setProd, w, groupsOf(h.rows, matvecOuts), unsafe.Pointer(&push))
			}
			at += w
		}
	}
	r.Barrier()
	n := uint32(h.rows)
	r.DispatchColumns(h.setArg, 1, uint32(columns), unsafe.Pointer(&n))
	r.Barrier()
}

func (h *draftHead) close() {
	if h.pass != nil {
		h.pass.Close()
	}
	for _, prog := range h.peaks {
		prog.Close()
	}
	if h.gm != nil {
		h.gm.Close()
	}
	if h.prep != nil {
		h.prep.Close()
	}
	for _, s := range []*Set{h.setQuant, h.setProd, h.setArg} {
		if s != nil {
			s.Close()
		}
	}
	if h.pipeArg != nil {
		h.pipeArg.Close()
	}
	for _, b := range []*Buffer{h.weights, h.signs, h.logits, h.best, h.act} {
		if b != nil {
			b.Close()
		}
	}
}
