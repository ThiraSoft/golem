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

type Model struct {
	mimiModel *tensors.Model
	sttModel  *tensors.Model
	encoder   *mimi.STTEncoder
	quantizer *mimi.Quantizer
	weights   *Weights
	tokenizer *sentencepiece.Tokenizer
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
	if m.weights, err = LoadWeights(m.sttModel); err != nil {
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
	textCh     chan Segment
	closed     bool
	mu         sync.Mutex
}

// Stream transcribes as it is fed. Write takes any number of samples at
// 24 kHz mono; Text yields segments as they are decided and closes when Close
// has been called and the tail has been flushed.
func (m *Model) Stream(ctx context.Context) *Live {
	l := &Live{
		m:         m,
		ctx:       ctx,
		state:     m.encoder.NewState(),
		kv:        NewKV(),
		prevToken: TextCard, // 8000: start of sequence
		textCh:    make(chan Segment, 64),
	}
	// Write 0.5 s (6 frames) of silence as required by stt_config.audio_delay_seconds.
	silence := make([]float32, AudioDelayFrames*mimi.SamplesPerFrame)
	l.writeSamples(silence)
	return l
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
		l.stepFrame(chunk)
	}
}

func (l *Live) stepFrame(chunk []float32) {
	latents, n := l.m.encoder.Push(chunk, l.state)
	if n == 0 {
		return
	}
	latent := make([]float32, mimi.STTConfig.LatentDim)
	codes := make([]int, l.m.quantizer.Codebooks)
	logits := make([]float32, TextCard)

	for f := 0; f < n; f++ {
		for c := 0; c < mimi.STTConfig.LatentDim; c++ {
			latent[c] = latents[c*n+f]
		}
		l.m.quantizer.Encode(latent, codes)

		x := make([]float32, DModel)
		// Sum audio embeddings
		for q := 0; q < Codebooks; q++ {
			c := codes[q]
			base := c * DModel * 2
			for j := 0; j < DModel; j++ {
				bits := uint32(binary.LittleEndian.Uint16(l.m.weights.Audio[q][base+j*2:])) << 16
				x[j] += math.Float32frombits(bits)
			}
		}

		// Add previous text token embedding
		textBase := l.prevToken * DModel * 2
		for j := 0; j < DModel; j++ {
			bits := uint32(binary.LittleEndian.Uint16(l.m.weights.Text[textBase+j*2:])) << 16
			x[j] += math.Float32frombits(bits)
		}

		// Trunk transformer layers
		for layerIdx, layer := range l.m.weights.Layers {
			layer.Step(x, l.kv[layerIdx])
		}

		// Out norm and head
		nn.RMSNormPlain(x, l.m.weights.OutNorm, 1e-5)
		l.m.weights.Head.Apply(x, logits)

		// Argmax over TextCard
		bestTok, bestVal := 0, logits[0]
		for tok := 1; tok < TextCard; tok++ {
			if logits[tok] > bestVal {
				bestVal = logits[tok]
				bestTok = tok
			}
		}

		l.prevToken = bestTok
		if bestTok > TextPadID {
			text := l.m.decodePiece(bestTok)
			if text != "" {
				frameIdx := l.frameCount - AudioDelayFrames
				select {
				case l.textCh <- Segment{Text: text, Frame: frameIdx}:
				case <-l.ctx.Done():
					return
				}
			}
		}
		l.frameCount++
	}
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
	samples, rate, channels, err := decode.Decode(bytes.NewReader(file))
	if err != nil {
		return err
	}
	mono := resample.Mono(samples, channels)
	resampled := resample.To(mono, rate, SampleRate)

	live := m.Stream(ctx)
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
