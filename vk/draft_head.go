package vk

import (
	_ "embed"
	"fmt"
	"unsafe"

	"github.com/ThiraSoft/golem/nn"
)

// The prediction block's own logit head, and the peak of what it writes, laid
// down behind the block in one recording.
//
// A guess used to be the block on the card, its hidden state read back, the
// model's head run as a matrix of its own against it, a quarter of a million
// logits read back, and their peak found on the processor: on Bonsai 2 that
// was 1.8 milliseconds a guess, of which the block itself was 0.7. Here the
// head reads the block's output where it lies and one integer comes back.
//
// The head may be the model's or fewer rows of it: ids says which token each
// row is, and nil means row i is token i.

//go:generate glslc -O --target-env=vulkan1.1 -fshader-stage=compute shaders/argmax.comp -o shaders/argmax.spv
//go:embed shaders/argmax.spv
var argmaxSPIRV []byte

type draftHead struct {
	rows int
	ids  []int32

	weights, signs, logits, best *Buffer
	pipeArg                      *Pipeline
	setQuant, setProd, setArg    *Set
	pass                         *Program
	// peaks are PeaksOf's recordings, one per width.
	peaks map[int]*Program
}

// peakColumns is the widest pass PeaksOf answers.
const peakColumns = 16

// AddDraftHead gives the prediction block a head of its own. data is the
// matrix as the file holds it, rows by cols in format q; pre is the sign vector
// a Prism head rotates its input by, nil for a plain one.
func (p *QwenPipeline) AddDraftHead(data []byte, q nn.Quant, rows, cols int, pre []float32, ids []int32) error {
	if p.mtp == nil {
		return fmt.Errorf("vk: a draft head needs the prediction block")
	}
	if cols != p.shape.Dim {
		return fmt.Errorf("vk: a draft head reads %d inputs, the block writes %d", cols, p.shape.Dim)
	}
	if ids != nil && len(ids) != rows {
		return fmt.Errorf("vk: %d rows named by %d ids", rows, len(ids))
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
	if h.weights, err = p.d.Upload(layout); err != nil {
		return err
	}
	if h.logits, err = p.local(rows * 4 * peakColumns); err != nil {
		return err
	}
	if h.best, err = p.d.Readback(4*peakColumns, bufferUsageStorage); err != nil {
		return err
	}
	if pre != nil {
		if h.signs, err = p.upload(asBytes(pre)); err != nil {
			return err
		}
		if h.setQuant, err = p.rotPipes.Bind(p.stage, h.signs, p.normedQ, p.normedS); err != nil {
			return err
		}
	} else if h.setQuant, err = p.pipeQuant.NewSet([]*Buffer{p.stage, p.normedQ, p.normedS}); err != nil {
		return err
	}
	if h.setProd, err = pipe.NewSet([]*Buffer{h.weights, p.normedQ, p.normedS, h.logits}); err != nil {
		return err
	}
	if h.pipeArg, err = p.d.NewPipeline(argmaxSPIRV, 2, 4); err != nil {
		return err
	}
	if h.setArg, err = h.pipeArg.NewSet([]*Buffer{h.logits, h.best}); err != nil {
		return err
	}
	p.draft = h
	return nil
}

// HasDraftHead says whether DraftTokenAt can answer.
func (p *QwenPipeline) HasDraftHead() bool { return p.draft != nil }

// DraftTokenAt runs the prediction block and its head, and answers the token
// at the head's peak and the block's output-normed state, which the next
// guess of a chain reads as its hidden half.
func (p *QwenPipeline) DraftTokenAt(eh []float32, at QwenPlace) (int32, []float32, error) {
	s := p.shape
	h := p.draft
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
			p.recordDraftHead(r)
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

func (p *QwenPipeline) recordDraftHead(r *Recorder) {
	s := p.shape
	h := p.draft
	if h.signs != nil {
		p.rotate(r, h.setQuant, s.Dim, 1)
	} else {
		quant := swigluPush{N: uint32(s.Dim), Columns: 1}
		r.Dispatch(h.setQuant, uint32((s.Dim/quantBlock+255)/256), unsafe.Pointer(&quant))
		r.Barrier()
	}
	push := moePush{dim: uint32(h.rows), ffn: uint32(s.Dim), used: 1, split: 1}
	r.Dispatch(h.setProd, groupsOf(h.rows, matvecOuts), unsafe.Pointer(&push))
	r.Barrier()
	n := uint32(h.rows)
	r.Dispatch(h.setArg, 1, unsafe.Pointer(&n))
	r.Barrier()
}

// PeaksOf is the token at the head's peak for each of the last pass's first
// columns, found where the pass left its output-normed states: the model's
// greedy answer at every column of a speculative pass, with one integer a
// column read back instead of a row of logits. It is only the model's own
// answer when the draft head is the model's whole head, which ids nil says.
func (p *QwenPipeline) PeaksOf(columns int) ([]int32, error) {
	h := p.draft
	if h == nil || h.ids != nil {
		return nil, fmt.Errorf("vk: the pipeline carries no full draft head")
	}
	if columns < 1 || columns > peakColumns {
		return nil, fmt.Errorf("vk: peaks of one to %d columns, given %d", peakColumns, columns)
	}
	prog, ok := h.peaks[columns]
	if !ok {
		var err error
		if prog, err = p.d.Compile(func(r *Recorder) { p.recordPeaks(r, columns) }); err != nil {
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
	best := unsafe.Slice((*uint32)(unsafe.Pointer(&h.best.Bytes()[0])), peakColumns)
	out := make([]int32, columns)
	for c := range out {
		out[c] = int32(best[c])
	}
	return out, nil
}

func (p *QwenPipeline) recordPeaks(r *Recorder, columns int) {
	s := p.shape
	h := p.draft
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
	for _, s := range []*Set{h.setQuant, h.setProd, h.setArg} {
		if s != nil {
			s.Close()
		}
	}
	if h.pipeArg != nil {
		h.pipeArg.Close()
	}
	for _, b := range []*Buffer{h.weights, h.signs, h.logits, h.best} {
		if b != nil {
			b.Close()
		}
	}
}
