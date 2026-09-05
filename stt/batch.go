package stt

// Several microphones stepping together.
//
// One stream saturates this processor: the aggregate throughput of concurrent
// transcriptions is flat from one client to eight, because the trunk's fifty-
// four megabytes of weights a layer do not fit in any cache and every stream
// reads them again from main memory. The arithmetic is not the wall — the
// traffic is.
//
// So the streams are stepped together and the weights are read once for all of
// them, which is what nn.MatVecBatch is: the row is the outer loop and the
// batch the inner one. It is the same change that lets a prompt be read faster
// than an answer is written, applied to a different reason for having more than
// one column.
//
// What batches and what does not. The four matrices of a block do, and so does
// the logit head. Attention does not: each stream has its own cache and its own
// window, so N streams are N attentions — they are spread over the pool by
// stream and head together rather than folded into one product. The Mimi codec
// and the split quantiser do not either, and they are what the ceiling becomes
// once the products stop being it.

import (
	"context"
	"errors"
	"fmt"
	"math"
	"strings"
	"sync"
	"time"

	"github.com/ThiraSoft/golem/nn"
)

// BatchScratch is what one batched pass needs, sized for the widest batch the
// group will ever run. A pass narrower than that shrinks nn.Batch.Size, which
// the kernels read as the column count while Stride keeps the layout the
// buffers were allocated with.
type BatchScratch struct {
	size       int
	wide, deep *nn.Batch
	qkv        [][]float32 // size x 3*DModel
	qs         [][]float32 // views into qkv: the query of each stream
	out        [][]float32 // size x DModel
	gate       [][]float32 // size x 2*DimFF
	ff         [][]float32 // size x DModel
	attn       [][]float32 // size x DModel
	logits     [][]float32 // size x TextCard
	xs         [][]float32 // size x DModel: the activation each stream carries
	// scores is one window of scores per stream and head, because the heads of
	// every stream are softmaxed at the same time on different cores.
	scores [][]float32
}

func NewBatchScratch(size int) *BatchScratch {
	s := &BatchScratch{
		size:   size,
		wide:   nn.NewBatch(DModel, size),
		deep:   nn.NewBatch(DimFF, size),
		qkv:    columnsOf(size, 3*DModel),
		qs:     make([][]float32, size),
		out:    columnsOf(size, DModel),
		gate:   columnsOf(size, 2*DimFF),
		ff:     columnsOf(size, DModel),
		attn:   columnsOf(size, DModel),
		logits: columnsOf(size, TextCard),
		xs:     columnsOf(size, DModel),
		scores: columnsOf(size*NumHeads, Context),
	}
	return s
}

func columnsOf(n, width int) [][]float32 {
	ys := make([][]float32, n)
	for i := range ys {
		ys[i] = make([]float32, width)
	}
	return ys
}

// productBatch is product for more than one activation: it quantizes only when
// the weights are quantized, for the reason product gives.
func productBatch(m nn.Matrix, b *nn.Batch, ys [][]float32) {
	if m.Quant != nn.BF16 {
		b.Quantize()
	}
	m.MatVecBatch(b, ys)
}

// StepBatch advances every stream of the batch one position through this block.
// xs, kvs and the scratch's columns are indexed alike, and every stream must be
// at a position of its own — they share the weights and nothing else.
func (l *Layer) StepBatch(xs [][]float32, kvs []*KV, s *BatchScratch) {
	n := len(xs)
	if n > s.size {
		panic(fmt.Sprintf("stt: a batch of %d in a scratch built for %d", n, s.size))
	}
	s.wide.Size, s.deep.Size = n, n

	for i := 0; i < n; i++ {
		copy(s.wide.F[i], xs[i])
		nn.RMSNormPlain(s.wide.F[i], l.Norm1, NormEps)
	}
	productBatch(l.InProj, s.wide, s.qkv[:n])

	for i := 0; i < n; i++ {
		qkv := s.qkv[i]
		q, k, v := qkv[:DModel], qkv[DModel:2*DModel], qkv[2*DModel:]
		for head := 0; head < NumHeads; head++ {
			nn.ApplyRoPE(q[head*HeadDim:(head+1)*HeadDim], kvs[i].Position, MaxPeriod)
			nn.ApplyRoPE(k[head*HeadDim:(head+1)*HeadDim], kvs[i].Position, MaxPeriod)
		}
		kvs[i].write(k, v)
		s.qs[i] = q
	}
	attendBatch(kvs, s.qs[:n], s.attn[:n], s)

	for i := 0; i < n; i++ {
		copy(s.wide.F[i], s.attn[i])
	}
	productBatch(l.OutProj, s.wide, s.out[:n])
	for i := 0; i < n; i++ {
		x, out := xs[i], s.out[i]
		for j := range x {
			x[j] += out[j]
		}
	}

	for i := 0; i < n; i++ {
		copy(s.wide.F[i], xs[i])
		nn.RMSNormPlain(s.wide.F[i], l.Norm2, NormEps)
	}
	productBatch(l.GateIn, s.wide, s.gate[:n])
	for i := 0; i < n; i++ {
		gate := s.gate[i]
		nn.SwiGLURange(gate[:DimFF], gate[DimFF:], 0, DimFF)
		copy(s.deep.F[i], gate[:DimFF])
	}
	productBatch(l.GateOut, s.deep, s.ff[:n])
	for i := 0; i < n; i++ {
		x, ff := xs[i], s.ff[i]
		for j := range x {
			x[j] += ff[j]
		}
		kvs[i].Position++
	}
}

// attendBatch is attend for several streams at once. The unit of work is one
// head of one stream: sixteen heads times the batch, spread over the pool in
// one section rather than one section per stream, so a batch of two does not
// pay two wake-ups a block.
func attendBatch(kvs []*KV, qs, outs [][]float32, s *BatchScratch) {
	n := len(kvs)
	scale := float32(1.0 / math.Sqrt(float64(HeadDim)))
	work := 0
	for i := 0; i < n; i++ {
		clear(outs[i])
		work += visible(kvs[i]) * NumHeads * HeadDim * 2
	}
	nn.InParallel(n*NumHeads, work, func(start, end int) {
		for u := start; u < end; u++ {
			i, h := u/NumHeads, u%NumHeads
			kv := kvs[i]
			first := kv.Position + 1 - visible(kv)
			buffer := s.scores[u][:visible(kv)]
			qh := qs[i][h*HeadDim : (h+1)*HeadDim]
			for p := first; p <= kv.Position; p++ {
				slot := p % Context
				kp := kv.K[(slot*NumHeads+h)*HeadDim : (slot*NumHeads+h+1)*HeadDim]
				buffer[p-first] = nn.DotF32(qh, kp) * scale
			}
			nn.SoftmaxInPlace(buffer)
			oh := outs[i][h*HeadDim : (h+1)*HeadDim]
			for p := first; p <= kv.Position; p++ {
				slot := p % Context
				vp := kv.V[(slot*NumHeads+h)*HeadDim : (slot*NumHeads+h+1)*HeadDim]
				nn.AxpyFull(oh, vp, buffer[p-first])
			}
		}
	})
}

// visible is how many positions of the past this cache still holds.
func visible(kv *KV) int {
	if kv.Position+1 > Context {
		return Context
	}
	return kv.Position + 1
}

// Group steps the streams opened through it together.
//
// A stream hands the group a frame and waits for it: the group gathers whatever
// other streams are ready, runs one batched pass for all of them, and releases
// them together. The wait is what buys the batch, and it is bounded — a frame
// is eighty milliseconds of audio and the pass costs a fraction of that, so a
// few milliseconds spent gathering is latency the budget already has.
type Group struct {
	m      *Model
	ctx    context.Context
	cancel context.CancelFunc
	size   int
	window time.Duration

	in      chan *frameRequest
	done    chan struct{}
	scratch *BatchScratch

	// live is how many streams are open. A stream that closes gives its slot
	// back, so a server whose client hangs up can take another: the batch is
	// never wider than this, which is what keeps it inside the scratch.
	mu   sync.Mutex
	live int

	// card is the model's, held here so a pass does not reach through the
	// model for it every block. It is nil when the processor carries the trunk.
	card *Card

	// err is the first fault a pass met, which every stream of the group then
	// reports. A card that stops answering stops all of them.
	errMu sync.Mutex
	err   error
}

// frameRequest is one stream's frame, and the channel that says it is finished.
type frameRequest struct {
	l     *Live
	chunk []float32
	ready chan struct{}
}

// GroupWindow is how long the group waits for a second stream before stepping
// what it has. Long enough that streams driven by real time find each other,
// short enough to disappear inside a frame.
const GroupWindow = 4 * time.Millisecond

// Group opens a batch of at most size concurrent transcriptions. Close it, or
// cancel the context, when they are all finished.
func (m *Model) Group(ctx context.Context, size int) *Group {
	if size < 1 {
		size = 1
	}
	inner, cancel := context.WithCancel(ctx)
	g := &Group{
		m:       m,
		ctx:     inner,
		cancel:  cancel,
		size:    size,
		window:  GroupWindow,
		card:    m.card,
		in:      make(chan *frameRequest, size),
		done:    make(chan struct{}),
		scratch: NewBatchScratch(size),
	}
	go g.serve()
	return g
}

// ErrGroupFull is what Stream answers when every slot of the group is taken. A
// caller that must not be refused can wait and ask again; the group does not
// queue, because a stream it accepted has to be stepped inside the frame budget
// of the streams already running.
var ErrGroupFull = errors.New("stt: every stream of the group is open")

// Stream opens one transcription inside the group. It refuses past the size the
// group was built for, because a batch wider than its scratch has nowhere to
// put its columns.
func (g *Group) Stream(ctx context.Context) (*Live, error) {
	g.mu.Lock()
	if g.live >= g.size {
		g.mu.Unlock()
		return nil, fmt.Errorf("%w: it carries %d", ErrGroupFull, g.size)
	}
	g.live++
	g.mu.Unlock()

	// The stream ends when its caller's context does, and also when the
	// group's does. It is the caller's context that matters here: a request
	// that goes away has to release its slot and stop being gathered, and
	// before this the streams of a group all carried the group's context and
	// so noticed nothing until the server itself stopped.
	inner, cancel := context.WithCancel(ctx)
	l := g.m.newLive(inner, g)
	l.cancel = cancel
	l.stopWatch = context.AfterFunc(g.ctx, cancel)
	// A caller that abandons a stream without closing it must not cost the
	// others the gathering window on every frame for ever.
	context.AfterFunc(inner, l.release)

	// Priming is outside the lock: it writes half a second of silence through
	// the group, and the pass that carries it asks how many streams are live.
	l.prime()
	return l, nil
}

// Close stops the group's pass. The streams it opened must be closed first.
func (g *Group) Close() error {
	g.cancel()
	<-g.done
	return nil
}

// fail records the fault a pass met and ends the streams that were in it.
func (g *Group) fail(err error, live []*Live) {
	g.errMu.Lock()
	if g.err == nil {
		g.err = err
	}
	g.errMu.Unlock()
	for _, l := range live {
		l.faulted(err)
	}
}

// Err is the first fault a pass of this group met, or nil. A transcription
// that ended early asks it for the reason.
func (g *Group) Err() error {
	g.errMu.Lock()
	defer g.errMu.Unlock()
	return g.err
}

// leave is a stream saying it will send no more frames, and giving its slot to
// whoever asks next.
func (g *Group) leave() {
	g.mu.Lock()
	g.live--
	g.mu.Unlock()
}

// Transcribe reads a whole sound file on one of the group's streams, ending it
// with the caller's context.
func (g *Group) Transcribe(ctx context.Context, file []byte) (string, error) {
	var whole strings.Builder
	if err := g.TranscribeStream(ctx, file, func(seg Segment) {
		whole.WriteString(seg.Text)
	}); err != nil {
		return "", err
	}
	return strings.TrimSpace(whole.String()), nil
}

// TranscribeStream is Model.TranscribeStream on one of the group's streams, so
// that several files being transcribed at once are stepped together. It refuses
// when the group is full rather than queueing: a caller that wants to wait can
// wait, and one that wants to answer "busy" has something to answer with.
func (g *Group) TranscribeStream(ctx context.Context, file []byte, each func(Segment)) error {
	live, err := g.Stream(ctx)
	if err != nil {
		return err
	}
	if err := runStream(live, file, each); err != nil {
		return err
	}
	// A stream that was evicted mid-pass ends quietly, so the group is asked
	// whether it ended for a reason.
	return g.Err()
}

// waiting is how many streams could still send a frame for this pass. The
// group does not wait on streams that have gone.
func (g *Group) waiting() int {
	g.mu.Lock()
	defer g.mu.Unlock()
	return g.live
}

// submit hands one frame to the group and waits for the pass that carries it.
func (g *Group) submit(l *Live, chunk []float32) {
	r := &frameRequest{l: l, chunk: chunk, ready: make(chan struct{})}
	select {
	case g.in <- r:
	case <-g.ctx.Done():
		return
	}
	select {
	case <-r.ready:
	case <-g.ctx.Done():
	}
}

// serve gathers frames and runs one pass for each gathering.
func (g *Group) serve() {
	defer close(g.done)
	for {
		var batch []*frameRequest
		select {
		case r := <-g.in:
			batch = append(batch, r)
		case <-g.ctx.Done():
			return
		}
		// Whatever is already queued is free to take.
		for len(batch) < g.size {
			select {
			case r := <-g.in:
				batch = append(batch, r)
				continue
			default:
			}
			break
		}
		// Then wait, but only while a stream that has not sent could still
		// send: a group carrying one stream must not pay the window a frame.
		if len(batch) < min(g.size, g.waiting()) {
			timer := time.NewTimer(g.window)
		gather:
			for len(batch) < min(g.size, g.waiting()) {
				select {
				case r := <-g.in:
					batch = append(batch, r)
				case <-timer.C:
					break gather
				case <-g.ctx.Done():
					timer.Stop()
					return
				}
			}
			timer.Stop()
		}
		g.step(batch)
		for _, r := range batch {
			close(r.ready)
		}
	}
}

// step runs one batched pass over the frames gathered.
//
// The codec is per stream and runs first, because it decides how many trunk
// steps each frame is worth: usually one, sometimes none, and the streams that
// disagree are carried in separate rounds rather than padded to the longest.
func (g *Group) step(batch []*frameRequest) {
	type pending struct {
		l     *Live
		codes [][]int
	}
	rounds := 0
	work := make([]pending, 0, len(batch))
	for _, r := range batch {
		codes := r.l.codesFor(r.chunk)
		work = append(work, pending{l: r.l, codes: codes})
		if len(codes) > rounds {
			rounds = len(codes)
		}
	}

	s := g.scratch
	xs := make([][]float32, 0, len(batch))
	kvs := make([]*KV, 0, len(batch))
	live := make([]*Live, 0, len(batch))
	for round := 0; round < rounds; round++ {
		// The embedding carries the token the previous position produced, so it
		// is built round by round and not once for the whole frame.
		xs, kvs, live = xs[:0], kvs[:0], live[:0]
		for _, p := range work {
			if round >= len(p.codes) {
				continue
			}
			x := s.xs[len(xs)]
			p.l.embed(p.codes[round], x)
			xs = append(xs, x)
			kvs = append(kvs, nil) // filled per layer below
			live = append(live, p.l)
		}
		for layer := range g.m.weights.Layers {
			for i, l := range live {
				kvs[i] = l.kv[layer]
			}
			if g.card != nil {
				if err := g.card.StepBatchOn(layer, g.m.weights.Layers[layer], xs, kvs, s); err != nil {
					// The card failing mid-frame is not something a transcript
					// can be salvaged from: half a block ran there and half
					// here. The streams of this pass are ended and the reason
					// is carried out through the group.
					g.fail(err, live)
					return
				}
				continue
			}
			g.m.weights.Layers[layer].StepBatch(xs, kvs, s)
		}
		n := len(xs)
		s.wide.Size = n
		for i := 0; i < n; i++ {
			nn.RMSNormPlain(xs[i], g.m.weights.OutNorm, NormEps)
			copy(s.wide.F[i], xs[i])
		}
		productBatch(g.m.weights.Head, s.wide, s.logits[:n])
		for i, l := range live {
			l.emit(s.logits[i])
		}
	}
}
