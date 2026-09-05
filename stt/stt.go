package stt

// Kyutai STT: Speech to text in pure Go.
//
// Sound in, words out — either as a whole file transcribed at once, or
// streaming from a live audio source.

import (
	"bytes"
	"context"
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"
	"sync"

	"github.com/ThiraSoft/golem/audio/decode"
	"github.com/ThiraSoft/golem/audio/resample"
	"github.com/ThiraSoft/golem/internal/kyutai/mimi"
	"github.com/ThiraSoft/golem/nn"
	"github.com/ThiraSoft/golem/tensors"
	"github.com/ThiraSoft/golem/token/sentencepiece"
)

// SampleRate is the audio rate expected by the Mimi codec.
const SampleRate = 24000

// AudioDelayFrames is the 0.5 s delay in 80 ms frames (6 frames).
const AudioDelayFrames = 6

type Options struct {
	Weights   string
	Mimi      string
	Tokenizer string

	// Quant is the format the trunk's sixteen blocks are converted to at load.
	// The zero value means nn.Q8_0, and the choice was measured rather than
	// assumed: on half a minute of read English, Q8_0 gives bfloat16's
	// transcript word for word at twice real time, where Q4_0 is half again as
	// fast and drops the "st" of a date — three per cent of the words, and on
	// exactly the kind of token a transcriber is for. nn.BF16 keeps the file's
	// own weights, which is what the tests that hold this against PyTorch read.
	Quant nn.Quant
}

// Locate fills the three paths from a directory holding the checkpoint as it
// ships. A field already set is left alone.
func Locate(dir string) (Options, error) {
	o := Options{}

	wPath := filepath.Join(dir, "model.safetensors")
	if _, err := os.Stat(wPath); err == nil {
		o.Weights = wPath
	} else {
		return o, fmt.Errorf("stt: %w", err)
	}

	mMatches, _ := filepath.Glob(filepath.Join(dir, "mimi*.safetensors"))
	if len(mMatches) > 0 {
		o.Mimi = mMatches[0]
	} else {
		return o, fmt.Errorf("stt: no mimi codec safetensors found in %s", dir)
	}

	tMatches, _ := filepath.Glob(filepath.Join(dir, "tokenizer*.model"))
	if len(tMatches) > 0 {
		o.Tokenizer = tMatches[0]
	} else {
		return o, fmt.Errorf("stt: no tokenizer model found in %s", dir)
	}
	return o, nil
}

// Quant names the format the trunk's blocks were converted to at load. It is
// worth a startup line: the same checkpoint runs at three times the speed in
// one format than in the other, and a server that quietly fell back would look
// like a slow machine.
func (m *Model) Quant() nn.Quant { return m.weights.Quant }

type Model struct {
	mimiModel *tensors.Model
	sttModel  *tensors.Model
	encoder   *mimi.STTEncoder
	quantizer *mimi.Quantizer
	weights   *Weights
	tokenizer *sentencepiece.Tokenizer
	// card holds the trunk's four products in device memory when UseVulkan
	// has put them there, and is nil when the processor carries them.
	card *Card
}

func Open(o Options) (*Model, error) {
	var err error
	m := &Model{}

	if m.mimiModel, err = tensors.Open(o.Mimi); err != nil {
		return nil, fmt.Errorf("stt: open mimi: %w", err)
	}
	if m.encoder, err = mimi.LoadSTTEncoder(m.mimiModel, mimi.STTConfig); err != nil {
		m.Close()
		return nil, fmt.Errorf("stt: load encoder: %w", err)
	}
	if m.quantizer, err = mimi.LoadQuantizer(m.mimiModel, mimi.STTConfig); err != nil {
		m.Close()
		return nil, fmt.Errorf("stt: load quantizer: %w", err)
	}

	if m.sttModel, err = tensors.Open(o.Weights); err != nil {
		m.Close()
		return nil, fmt.Errorf("stt: open weights: %w", err)
	}
	quant := o.Quant
	if quant == 0 {
		quant = nn.Q8_0
	}
	if m.weights, err = LoadWeights(m.sttModel, quant); err != nil {
		m.Close()
		return nil, fmt.Errorf("stt: load weights: %w", err)
	}

	if m.tokenizer, err = sentencepiece.Load(o.Tokenizer); err != nil {
		m.Close()
		return nil, fmt.Errorf("stt: load tokenizer: %w", err)
	}

	return m, nil
}

func (m *Model) Close() error {
	var firstErr error
	// The card first: its buffers are mapped from the weights the models below
	// own, and it holds a device that outlives neither.
	if m.card != nil {
		m.card.Close()
		m.card = nil
	}
	if m.mimiModel != nil {
		if err := m.mimiModel.Close(); err != nil && firstErr == nil {
			firstErr = err
		}
	}
	if m.sttModel != nil {
		if err := m.sttModel.Close(); err != nil && firstErr == nil {
			firstErr = err
		}
	}
	return firstErr
}

// Segment is one piece of text and the frame it was decided at; frames are
// 80 ms, so Frame/12.5 is its time in seconds.
type Segment struct {
	Text  string
	Frame int
}

// Live transcribes speech as it is fed.
type Live struct {
	m          *Model
	ctx        context.Context
	state      *mimi.STTState
	kv         []*KV
	prevToken  int
	frameCount int
	pending    []float32
	scratch    *Scratch
	group      *Group // nil when this stream steps alone
	// cancel ends this stream alone, and left gives its slot back exactly
	// once however it ends — closed by its caller, cancelled by the caller's
	// context, or evicted for not reading what it asked for.
	cancel    context.CancelFunc
	stopWatch func() bool
	left      sync.Once
	// batch is the one-column scratch a lone stream needs when the trunk is on
	// a card, built the first time a frame reaches it.
	batch  *BatchScratch
	errMu  sync.Mutex
	err    error
	textCh chan Segment
	closed bool
	mu     sync.Mutex
}

// Stream transcribes as it is fed. Write takes any number of samples at
// 24 kHz mono; Text yields segments as they are decided and closes when Close
// has been called and the tail has been flushed.
func (m *Model) Stream(ctx context.Context) *Live {
	l := m.newLive(ctx, nil)
	l.prime()
	return l
}

// newLive builds one stream. When g is not nil the stream hands its frames to
// that group instead of stepping them itself. The caller primes it, so that a
// group can finish wiring the stream's ending before half a second of silence
// goes through the shared pass.
func (m *Model) newLive(ctx context.Context, g *Group) *Live {
	return &Live{
		m:         m,
		ctx:       ctx,
		state:     m.encoder.NewState(),
		kv:        NewKV(),
		scratch:   NewScratch(),
		group:     g,
		prevToken: TextCard, // 8000: start of sequence
		textCh:    make(chan Segment, 64),
	}
}

// prime writes the half second of silence stt_config.audio_delay_seconds asks
// for before anything real is fed.
func (l *Live) prime() {
	silence := make([]float32, AudioDelayFrames*mimi.SamplesPerFrame)
	l.writeSamples(silence)
}

func (l *Live) Text() <-chan Segment {
	return l.textCh
}

func (l *Live) Write(samples []float32) {
	l.mu.Lock()
	defer l.mu.Unlock()
	if l.closed {
		return
	}
	l.writeSamples(samples)
}

func (l *Live) writeSamples(samples []float32) {
	l.pending = append(l.pending, samples...)
	for len(l.pending) >= mimi.SamplesPerFrame {
		chunk := l.pending[:mimi.SamplesPerFrame]
		l.pending = l.pending[mimi.SamplesPerFrame:]
		if l.group != nil {
			l.group.submit(l, chunk)
			continue
		}
		l.stepFrame(chunk)
	}
}

func (l *Live) stepFrame(chunk []float32) {
	x := make([]float32, DModel)
	logits := make([]float32, TextCard)
	// A stream that steps alone still reads the whole trunk for its one
	// column, so a card is worth having here too: it is a batch of one, which
	// is a width the kernels are built at.
	var xs [][]float32
	var kvs []*KV
	if l.m.card != nil {
		if l.batch == nil {
			l.batch = NewBatchScratch(1)
		}
		xs, kvs = [][]float32{x}, make([]*KV, 1)
	}
	for _, codes := range l.codesFor(chunk) {
		l.embed(codes, x)
		if l.m.card != nil {
			for layerIdx, layer := range l.m.weights.Layers {
				kvs[0] = l.kv[layerIdx]
				if err := l.m.card.StepBatchOn(layerIdx, layer, xs, kvs, l.batch); err != nil {
					l.faulted(err)
					return
				}
			}
		} else {
			for layerIdx, layer := range l.m.weights.Layers {
				layer.Step(x, l.kv[layerIdx], l.scratch)
			}
		}
		nn.RMSNormPlain(x, l.m.weights.OutNorm, NormEps)
		product(l.m.weights.Head, l.scratch.wide, x, logits)
		l.emit(logits)
	}
}

// faulted ends a lone stream that the card stopped answering for, and keeps the
// reason for whoever asks. Half a block ran on the card and half here, so there
// is no transcript to salvage.
func (l *Live) faulted(err error) {
	l.errMu.Lock()
	if l.err == nil {
		l.err = err
	}
	l.errMu.Unlock()
	if l.cancel != nil {
		l.cancel()
	}
}

// Err is the fault that ended this stream early, or nil.
func (l *Live) Err() error {
	l.errMu.Lock()
	defer l.errMu.Unlock()
	return l.err
}

// codesFor runs the codec over one frame of sound and returns the codes of
// every latent it produced — usually one, and none while the encoder is still
// filling its first window.
//
// It is separate from embed because it depends on nothing the trunk decides,
// which is what lets a group run the codec of several streams before stepping
// any of them. embed cannot be hoisted the same way: the activation carries the
// token the previous position produced.
func (l *Live) codesFor(chunk []float32) [][]int {
	latents, n := l.m.encoder.Push(chunk, l.state)
	if n == 0 {
		return nil
	}
	latent := make([]float32, mimi.STTConfig.LatentDim)
	out := make([][]int, n)
	for f := 0; f < n; f++ {
		for c := 0; c < mimi.STTConfig.LatentDim; c++ {
			latent[c] = latents[c*n+f]
		}
		codes := make([]int, l.m.quantizer.Codebooks)
		l.m.quantizer.Encode(latent, codes)
		out[f] = codes
	}
	return out
}

// embed builds the activation of one position: the thirty-two audio codebooks
// summed, plus the embedding of the token the previous position decided.
func (l *Live) embed(codes []int, x []float32) {
	clear(x)
	for q := 0; q < Codebooks; q++ {
		c := codes[q]
		base := c * DModel * 2
		for j := 0; j < DModel; j++ {
			bits := uint32(binary.LittleEndian.Uint16(l.m.weights.Audio[q][base+j*2:])) << 16
			x[j] += math.Float32frombits(bits)
		}
	}
	textBase := l.prevToken * DModel * 2
	for j := 0; j < DModel; j++ {
		bits := uint32(binary.LittleEndian.Uint16(l.m.weights.Text[textBase+j*2:])) << 16
		x[j] += math.Float32frombits(bits)
	}
}

// send hands one segment to whoever is reading, and decides what to do when
// nobody is.
//
// A stream that steps alone may wait: the only clock it holds up is its own.
// A stream in a group may not — the pass that produced this segment is
// carrying every other stream of the group, and blocking it on one reader
// stops all of them. The buffer is sixty-four segments, which at twelve and a
// half frames a second is about five seconds of speech; a reader that far
// behind is gone rather than slow, so the stream is ended instead of waited
// for, and its caller sees a transcript that stops.
func (l *Live) send(seg Segment) {
	if l.group == nil {
		select {
		case l.textCh <- seg:
		case <-l.ctx.Done():
		}
		return
	}
	select {
	case l.textCh <- seg:
	case <-l.ctx.Done():
	default:
		l.evict()
	}
}

// evict ends this stream because nothing is reading it. The group stops
// gathering it at once; the caller still has to close it.
func (l *Live) evict() {
	if l.cancel != nil {
		l.cancel()
	}
	l.release()
}

// release gives the group's slot back, once, however the stream ended.
func (l *Live) release() {
	if l.group == nil {
		return
	}
	l.left.Do(func() {
		if l.stopWatch != nil {
			l.stopWatch()
		}
		l.group.leave()
	})
}

// emit takes the argmax of one position, remembers it for the next embedding,
// and sends whatever text it names.
func (l *Live) emit(logits []float32) {
	bestTok, bestVal := 0, logits[0]
	for tok := 1; tok < TextCard; tok++ {
		if logits[tok] > bestVal {
			bestVal = logits[tok]
			bestTok = tok
		}
	}
	l.prevToken = bestTok
	if bestTok > TextPadID {
		if text := l.m.decodePiece(bestTok); text != "" {
			l.send(Segment{Text: text, Frame: l.frameCount - AudioDelayFrames})
		}
	}
	l.frameCount++
}

func (l *Live) Close() error {
	l.mu.Lock()
	defer l.mu.Unlock()
	if l.closed {
		return nil
	}
	l.closed = true

	// Pad remainder of pending to whole frame
	if len(l.pending) > 0 {
		pad := make([]float32, mimi.SamplesPerFrame-len(l.pending))
		l.writeSamples(pad)
	}

	// Flush tail with 6 frames of silence
	tail := make([]float32, AudioDelayFrames*mimi.SamplesPerFrame)
	l.writeSamples(tail)

	// The group must stop waiting for a stream that will send nothing more, or
	// the streams still open would spend the gathering window on it every frame.
	l.release()
	if l.cancel != nil {
		l.cancel()
	}
	close(l.textCh)
	return nil
}

func (m *Model) decodePiece(id int) string {
	if id < 0 || id >= m.tokenizer.Size() {
		return ""
	}
	piece := m.tokenizer.Piece(id)
	if len(piece) == 6 && strings.HasPrefix(piece, "<0x") && piece[5] == ">"[0] {
		if v, ok := byteValue(piece); ok {
			return string([]byte{v})
		}
	}
	return strings.ReplaceAll(piece, sentencepiece.Space, " ")
}

func byteValue(s string) (byte, bool) {
	var v byte
	for _, c := range []byte(s[3:5]) {
		switch {
		case c >= "0"[0] && c <= "9"[0]:
			v = v<<4 | (c - "0"[0])
		case c >= "A"[0] && c <= "F"[0]:
			v = v<<4 | (c - "A"[0] + 10)
		case c >= "a"[0] && c <= "f"[0]:
			v = v<<4 | (c - "a"[0] + 10)
		default:
			return 0, false
		}
	}
	return v, true
}

// TranscribeStream decodes a sound file — WAV, MP3 or FLAC, any rate, any channel
// count — and calls back with each segment as it is decided.
func (m *Model) TranscribeStream(ctx context.Context, file []byte, each func(Segment)) error {
	live := m.Stream(ctx)
	if err := runStream(live, file, each); err != nil {
		return err
	}
	return live.Err()
}

// runStream feeds one decoded file to one stream and drains what it says. It is
// shared by the lone path and the group's, which differ only in where the
// stream came from. The context is not taken here on purpose: it is the stream
// that has to carry it — Model.Stream and Group.Stream each build the stream
// around the caller's context — and a second copy of it in this function would
// be one that stops nothing.
func runStream(live *Live, file []byte, each func(Segment)) error {
	samples, rate, channels, err := decode.Decode(bytes.NewReader(file))
	if err != nil {
		live.Close()
		return err
	}
	resampled := resample.To(resample.Mono(samples, channels), rate, SampleRate)

	done := make(chan struct{})
	go func() {
		defer close(done)
		for seg := range live.Text() {
			each(seg)
		}
	}()
	live.Write(resampled)
	if err := live.Close(); err != nil {
		return err
	}
	<-done
	return nil
}

// Transcribe decodes a sound file and returns what is said in it.
func (m *Model) Transcribe(file []byte) (string, error) {
	var b strings.Builder
	err := m.TranscribeStream(context.Background(), file, func(seg Segment) {
		b.WriteString(seg.Text)
	})
	if err != nil {
		return "", err
	}
	return strings.TrimSpace(b.String()), nil
}
