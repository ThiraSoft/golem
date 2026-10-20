package yue2

// A song from a request: yue2.cpp's pipeline_generate (src/pipeline.h) for
// one song and one variation. The score stage writes the ABC unless the
// request brings one or turns it off, the semantic stage writes the codes
// unless the request replays them, and then the flow matching half renders
// the codes chunk by chunk over the AR half's cache and the decoder turns the
// latents into audio.
//
// yue2.cpp loads one half at a time and lets it go after its stage. Here
// the three stay on the card for the pipeline's life, so a song costs no
// load and the AR's cache is copied into the NAR's on the card.

import (
	"fmt"
	"os"
	"path/filepath"
	"time"

	"github.com/ThiraSoft/golem/vk"
)

// The decoder's tiling, yue2.cpp's vae_core and vae_halo defaults.
const (
	pipelineVAECore = 512
	pipelineVAEHalo = 16
)

// FrameRate is how many semantic codes, and latent frames, a second holds.
const FrameRate = 25

// Options say where the weights are and how much context the pipeline holds.
type Options struct {
	// Dir holds YuE2-3B-<Model>.gguf and YuE2-Vae-F32.gguf; empty is
	// GOLEM_YUE2_DIR or its default.
	Dir string
	// Model is Q8_0 (the default) or BF16; the AR half on the card reads
	// Q8_0 only for now.
	Model string
	// Context is the cache in positions, yue2.cpp's --max-seq; 0 is the
	// model's whole context. It decides how the acoustic stage is chunked.
	Context int
}

// Pipeline holds the three stages on one device.
type Pipeline struct {
	dev      *vk.Device
	backbone *Backbone
	tok      *Tokenizer
	ar       *AR
	nar      *NAR
	vae      *VAE

	// context is the cache the chunks are cut for. The tests lower it
	// below what was allocated to reach two chunks on a short replay.
	context int
}

// Song is one rendered song.
type Song struct {
	// Score is the ABC the codes follow, empty when cot is off.
	Score string
	// Codes are the semantic stream's codec values, [0, CodecSize).
	Codes []int32
	// Latents are [Frames][64], Audio planar stereo [2][Samples] at 48 kHz.
	Latents         []float32
	Audio           []float32
	Frames, Samples int
	// Truncated says a stage hit its budget before its end token.
	Truncated bool
	// LMSeed and Seed are the seeds the song consumed, -1 resolved.
	LMSeed, Seed int64
	// Times are the stages' wall clocks: score, semantic, prefill (the
	// chunks' AR forwards), nar, vae and total.
	Times map[string]time.Duration
}

// Open puts the AR half, the NAR half and the decoder on the first Vulkan
// device, and keeps them there until Close.
func Open(o Options) (*Pipeline, error) {
	if o.Dir == "" {
		o.Dir = weightsDir()
	}
	if o.Model == "" {
		o.Model = "Q8_0"
	}
	d, err := vk.Open()
	if err != nil {
		return nil, err
	}
	p, err := openOn(d, o)
	if err != nil {
		d.Close()
		return nil, err
	}
	// The pipeline owns the device it opened, and closes it with the rest.
	p.dev = d
	return p, nil
}

func openOn(d *vk.Device, o Options) (*Pipeline, error) {
	if !d.Coopmat() {
		return nil, fmt.Errorf("yue2: the flow matching half and the decoder want cooperative matrices")
	}
	b, err := OpenBackbone(filepath.Join(o.Dir, "YuE2-3B-"+o.Model+".gguf"))
	if err != nil {
		return nil, err
	}
	// dev stays nil here: openOn's caller owns d, and a failure must not
	// close it.
	p := &Pipeline{backbone: b}
	fail := func(err error) (*Pipeline, error) {
		p.Close()
		return nil, err
	}
	cfg := b.Cfg
	p.context = o.Context
	if p.context <= 0 {
		p.context = cfg.Context
	}
	if p.context > cfg.Context {
		return fail(fmt.Errorf("yue2: a context of %d past the model's %d", p.context, cfg.Context))
	}
	if p.tok, err = NewTokenizer(b.file); err != nil {
		return fail(err)
	}
	// Two slots: the unconditional stream decodes beside the conditional one
	// when the guidance is not one.
	if p.ar, err = NewAR(d, b, p.context, 2); err != nil {
		return fail(err)
	}
	// A chunk of f frames behind a prefix of n holds n + f + 1 keys and its
	// block f + 2 more, and the chunk rule keeps n + 2f + 3 within the
	// context: the NAR's cache is the context, its arena the longest chunk.
	maxFrames := max((p.context-3)/2, 1)
	if p.nar, err = newNAR(d, b, maxFrames, p.context, p.context); err != nil {
		return fail(err)
	}
	w, err := OpenVAE(filepath.Join(o.Dir, "YuE2-Vae-F32.gguf"))
	if err != nil {
		return fail(err)
	}
	if p.vae, err = NewVAE(d, w, pipelineVAECore+2*pipelineVAEHalo); err != nil {
		return fail(err)
	}
	return p, nil
}

// Close gives the card back.
func (p *Pipeline) Close() {
	if p.vae != nil {
		p.vae.Close()
		p.vae = nil
	}
	if p.nar != nil {
		p.nar.Close()
		p.nar = nil
	}
	if p.ar != nil {
		p.ar.Close()
		p.ar = nil
	}
	if p.backbone != nil {
		p.backbone.Close()
		p.backbone = nil
	}
	if p.dev != nil {
		p.dev.Close()
		p.dev = nil
	}
}

// Context is the cache the chunks are cut for.
func (p *Pipeline) Context() int { return p.context }

// Tokenizer is the backbone's BPE, the one the score is written in.
func (p *Pipeline) Tokenizer() *Tokenizer { return p.tok }

// plan is what a request asks of the stages: the cot it renders with (a
// supplied stream with no score renders without one, whatever the mode: a
// score planned now is not the one the codes follow), whether a score goes
// in the prompt and whether the model writes it, and whether the codes are
// replayed.
type plan struct {
	cot                       Cot
	hasScore, planned, replay bool
}

func planOf(r Request) (plan, error) {
	cot, err := ParseCot(r.Cot)
	if err != nil {
		return plan{}, fmt.Errorf("yue2: cot must be full, melody or off: %w", err)
	}
	pl := plan{cot: cot, replay: r.SemanticTokens != ""}
	if pl.replay && r.ABC == "" {
		pl.cot = CotOff
	}
	pl.hasScore = pl.cot != CotOff
	pl.planned = pl.hasScore && r.ABC == ""
	return pl, nil
}

// supported refuses what yue2.cpp's pipeline does and this one does not yet.
func supported(r Request) error {
	switch {
	case r.Steps < 1:
		return fmt.Errorf("yue2: %d flow matching steps", r.Steps)
	case r.LMBatchSize != 1 || r.SynthBatchSize != 1:
		return fmt.Errorf("yue2: batches of %d songs and %d variations; one of each for now", r.LMBatchSize, r.SynthBatchSize)
	case len(r.Adapters) > 0:
		return fmt.Errorf("yue2: adapters are not supported yet")
	case r.SourceTokens != "":
		return fmt.Errorf("yue2: continuations are not supported yet")
	}
	if err := r.ABCSampling.Valid(); err != nil {
		return fmt.Errorf("yue2: abc sampling: %w", err)
	}
	if err := r.SemanticSampling.Valid(); err != nil {
		return fmt.Errorf("yue2: semantic sampling: %w", err)
	}
	return nil
}

// Generate renders r. Its negative seeds are drawn here and come back in the
// song. cancel, when not nil, is asked between tokens, flow steps, decoder
// tiles and stages; it ends the song with an error wrapping ErrCancelled.
func (p *Pipeline) Generate(r Request, cancel func() bool) (*Song, error) {
	return p.generate(r, false, cancel)
}

// GenerateForced is Generate with the request's score and codes taken as
// forced tokens through the token-by-token decode rather than replayed by a
// prefill: every step computes its distribution and then takes the given
// token. It costs what drawing them would, which is what it is for.
func (p *Pipeline) GenerateForced(r Request, cancel func() bool) (*Song, error) {
	return p.generate(r, true, cancel)
}

func (p *Pipeline) generate(r Request, forced bool, cancel func() bool) (*Song, error) {
	start := time.Now()
	if err := supported(r); err != nil {
		return nil, err
	}
	pl, err := planOf(r)
	if err != nil {
		return nil, err
	}
	stopped := func(stage string) error {
		if cancel != nil && cancel() {
			return fmt.Errorf("%w: before %s", ErrCancelled, stage)
		}
		return nil
	}
	r = r.Resolve()
	song := &Song{LMSeed: r.LMSeed, Seed: r.Seed, Times: map[string]time.Duration{}}

	var codes []int32
	if pl.replay || forced {
		values, err := ParseCodes(r.SemanticTokens)
		if err != nil {
			return nil, err
		}
		for _, v := range values {
			codes = append(codes, v+CodecOffset)
		}
	}
	// In forced mode the stages decode token by token what a replay would
	// prefill: a request's score is planned, its codes generated.
	if forced {
		pl.planned = pl.hasScore
		pl.replay = false
	}

	// The score: the request's, the model's or none.
	var abc []int32
	switch {
	case pl.planned:
		if err := stopped("the score"); err != nil {
			return nil, err
		}
		t := time.Now()
		open := BuildPrompt(p.tok, pl.cot, r.Style, r.Lyrics, nil, false)
		s := r.ABCSampling
		var force []int32
		if forced {
			force = p.tok.Encode(r.ABC)
			s.MaxTokens = len(force) + 1
			s.MinTokens = min(s.MinTokens, len(force))
		}
		g, err := Generate(p.ar, open, nil, 1, s, r.LMSeed, PhaseABC, force, cancel)
		if err != nil {
			return nil, err
		}
		abc = g.Tokens
		song.Score = p.tok.Decode(abc)
		song.Truncated = g.Truncated
		song.Times["score"] = time.Since(t)
	case pl.hasScore:
		abc = p.tok.Encode(r.ABC)
		song.Score = r.ABC
	}

	prefix := BuildPrompt(p.tok, pl.cot, r.Style, r.Lyrics, abc, pl.hasScore)
	guidance := r.CfgScale
	if guidance < 0 {
		guidance = DefaultGuidance(pl.cot)
	}
	var negative []int32
	if guidance != 1 {
		negative = BuildNegative(p.tok, pl.cot, abc, pl.hasScore)
	}

	// The codes: the request's, or the model's.
	if !pl.replay {
		if err := stopped("the codes"); err != nil {
			return nil, err
		}
		t := time.Now()
		s := r.SemanticSampling
		var force []int32
		if forced {
			force = codes
			s.MaxTokens = len(force) + 1
			s.MinTokens = min(s.MinTokens, len(force))
		} else if budget := int(r.Duration * FrameRate); budget > 0 && budget < s.MaxTokens {
			// The requested length caps the stage's budget, never raises it.
			s.MaxTokens = budget
			s.MinTokens = min(s.MinTokens, budget)
		}
		g, err := Generate(p.ar, prefix, negative, guidance, s, r.LMSeed, PhaseSemantic, force, cancel)
		if err != nil {
			return nil, err
		}
		codes = g.Tokens
		song.Truncated = song.Truncated || g.Truncated
		song.Times["semantic"] = time.Since(t)
	}
	frames := len(codes)
	if frames < 1 {
		return nil, fmt.Errorf("yue2: an empty semantic stream")
	}
	song.Frames = frames
	song.Codes = make([]int32, frames)
	for i, c := range codes {
		song.Codes[i] = c - CodecOffset
	}

	// The acoustic chunks: the context holds the prefix, the chunk's codes
	// and its latent block, once as tokens and once as frames.
	chunk := (p.context - len(prefix) - 3) / 2
	if chunk < 1 {
		return nil, fmt.Errorf("yue2: a prefix of %d leaves no acoustic context in %d", len(prefix), p.context)
	}
	lat := p.backbone.Cfg.LatentDim
	// The noise is drawn once for the whole song, each chunk taking its view.
	noise := make([]float32, frames*lat)
	TorchRandn(uint64(r.Seed), noise)
	song.Latents = make([]float32, frames*lat)

	// A generated song of several chunks has its first one sealed while the
	// AR's cache still holds the whole stream: the end token takes the row
	// that closes the chunk.
	if !pl.replay && frames > chunk {
		t := time.Now()
		p.ar.Trim(0, len(prefix)+chunk)
		if _, err := p.ar.Prefill(0, []int32{MusicEnd}, PhaseSemantic); err != nil {
			return nil, err
		}
		song.Times["prefill"] += time.Since(t)
	}
	for at := 0; at < frames; at += chunk {
		if err := stopped(fmt.Sprintf("chunk %d", at/chunk+1)); err != nil {
			return nil, err
		}
		n := min(chunk, frames-at)
		arLen := len(prefix) + n + 1
		// The chunk's sequence is the prefix, its codes and the end token.
		// Its head already sits in the cache: the prefix from the second
		// chunk on, the sealed sequence of a generated song's first; only the
		// tail is forwarded, its logits going nowhere.
		kept := arLen
		switch {
		case at > 0:
			kept = len(prefix)
		case pl.replay:
			kept = 0
		}
		if kept < arLen {
			t := time.Now()
			seq := append(append(append([]int32(nil), prefix...), codes[at:at+n]...), MusicEnd)
			p.ar.Trim(0, kept)
			if _, err := p.ar.Prefill(0, seq[kept:], PhaseSemantic); err != nil {
				return nil, err
			}
			song.Times["prefill"] += time.Since(t)
		}
		t := time.Now()
		if err := p.nar.SetPrefixFromStack(p.ar, 0, arLen); err != nil {
			return nil, err
		}
		x, err := p.nar.Solve(noise[at*lat:(at+n)*lat], n, r.Steps, cancel)
		if err != nil {
			return nil, err
		}
		copy(song.Latents[at*lat:], x)
		song.Times["nar"] += time.Since(t)
	}

	t := time.Now()
	audio, err := p.vae.decodeTiled(song.Latents, frames, pipelineVAECore, pipelineVAEHalo, cancel)
	if err != nil {
		return nil, err
	}
	song.Audio = audio
	song.Samples = len(audio) / 2
	song.Times["vae"] = time.Since(t)
	song.Times["total"] = time.Since(start)
	return song, nil
}

// Replay is request_replay for the song: r with the song's score (r's own
// when the song has none), its codes and the seeds it consumed, so that it
// renders the song again without the autoregression.
func (s *Song) Replay(r Request) Request {
	if s.Score != "" {
		r.ABC = s.Score
	}
	r.SemanticTokens = FormatCodes(s.Codes)
	r.LMSeed, r.Seed = s.LMSeed, s.Seed
	r.LMBatchSize, r.SynthBatchSize = 1, 1
	r.SourceTokens, r.SourceStart, r.SourceEnd = "", 0, -1
	return r
}

// ReadRequest parses the request file at path.
func ReadRequest(path string) (Request, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return Request{}, err
	}
	return ParseRequest(b)
}
