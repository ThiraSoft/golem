package yue2

import (
	"bytes"
	"errors"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"sync"
	"testing"
)

// The recording's acoustic seed (bench/guerreros-replay.json's seed).
const pipelineSeed int64 = 9207606299741175932

// pipelineContext is the cache the shared pipeline holds: the short replay's
// 3107 keys fit in one chunk, as in yue2.cpp's whole context.
const pipelineContext = 4096

// chunksContext is yue2-synth's --max-seq of pipeline/chunks: the short
// replay's prefix of 3006 leaves chunks of (3150 - 3006 - 3) / 2 = 70 frames,
// so its 100 frames take two.
const chunksContext = 3150

var pipelineShared struct {
	once sync.Once
	skip string
	err  error
	p    *Pipeline
}

// sharedPipeline is one pipeline for the tests that render, on the device
// the AR and NAR tests share.
func sharedPipeline(t *testing.T) *Pipeline {
	t.Helper()
	d := testDevice(t)
	s := &pipelineShared
	s.once.Do(func() {
		if _, err := os.Stat(filepath.Join(weightsDir(), "YuE2-3B-Q8_0.gguf")); err != nil {
			s.skip = "no backbone weights: see ref/yue2/README.md"
			return
		}
		if !d.Coopmat() {
			s.skip = "the pipeline wants cooperative matrices"
			return
		}
		// The device is the tests', which outlives the pipeline: openOn
		// leaves it out of the pipeline's Close.
		s.p, s.err = openOn(d, Options{Dir: weightsDir(), Model: "Q8_0", Context: pipelineContext})
	})
	if s.skip != "" {
		t.Skip(s.skip)
	}
	if s.err != nil {
		t.Fatal(s.err)
	}
	return s.p
}

func shortRequest(t *testing.T) Request {
	t.Helper()
	r, err := ReadRequest(filepath.Join("..", "ref", "yue2", "requests", "pipeline-short.json"))
	if err != nil {
		t.Fatal(err)
	}
	return r
}

// checkSNR holds got to bar decibels from want.
func checkSNR(t *testing.T, name string, got, want []float32, bar float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: %d values, want %d", name, len(got), len(want))
	}
	snr := snrDB(got, want)
	t.Logf("%s: SNR %.2f dB, rms error %.4f%%", name, snr, 100*rmsError(got, want))
	if !(snr >= bar) {
		t.Errorf("%s: SNR %.2f dB below %.2f dB", name, snr, bar)
	}
}

// TestPipelineShortReplay renders ref/yue2/requests/pipeline-short.json, the
// first 100 codes of the Guerreros replay with its score, against yue-synth's
// render of the same request: the latents (which are nar/latent byte for
// byte) and the decoder's float audio, before normalization.
//
// The audio's 30 dB bar holds against pipeline/short_exact, yue-synth's
// render with the integer dot products and the matrix cores off, whose
// products sum fp16 operands in float32 as golem's do. Against the default
// render the bar is yue2.cpp's own: at its defaults ggml-vulkan rounds every
// activation entering a Q8_0 product to Q8_1, and that alone puts its two
// renders 28.5 dB apart on this audio (ref/yue2/README.md). golem has to be
// at least as close to the default render as yue2.cpp's exact one is.
func TestPipelineShortReplay(t *testing.T) {
	fix := loadFixtures(t)
	want := fix.read(t, "pipeline/short_exact/audio")
	def := fix.read(t, "pipeline/short/audio")
	wantLatent := fix.read(t, "nar/latent")
	p := sharedPipeline(t)
	r := shortRequest(t)
	song, err := p.Generate(r, nil)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("times %v", song.Times)
	if song.Frames != 100 || song.Samples != 1920*100-64 || song.Truncated {
		t.Errorf("%d frames, %d samples, truncated %v; want 100, %d, false", song.Frames, song.Samples, song.Truncated, 1920*100-64)
	}
	if song.LMSeed != r.LMSeed || song.Seed != pipelineSeed || song.Score != r.ABC {
		t.Errorf("seeds %d %d and a score of %d bytes; want the request's", song.LMSeed, song.Seed, len(song.Score))
	}
	checkSNR(t, "latents", song.Latents, wantLatent, 30)
	checkSNR(t, "latents, exact render", song.Latents, fix.read(t, "pipeline/short_exact/latent"), 30)
	checkSNR(t, "audio, exact render", song.Audio, want, 30)
	checkSNR(t, "audio, default render", song.Audio, def, snrDB(want, def))

	stop := 0
	_, err = p.Generate(r, func() bool { stop++; return stop > 1 })
	if !errors.Is(err, ErrCancelled) {
		t.Errorf("a cancelled render returned %v, want ErrCancelled", err)
	}
}

// TestPipelineChunks renders the same request with the context of
// pipeline/chunks, yue-synth's render at --max-seq 3150: two chunks of 70 and
// 30 frames, the second re-prefilling its codes behind the prefix.
func TestPipelineChunks(t *testing.T) {
	fix := loadFixtures(t)
	want := fix.read(t, "pipeline/chunks/audio")
	wantLatent := fix.read(t, "pipeline/chunks/latent")
	p := sharedPipeline(t)
	p.context = chunksContext
	defer func() { p.context = pipelineContext }()
	song, err := p.Generate(shortRequest(t), nil)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("times %v", song.Times)
	checkSNR(t, "latents", song.Latents, wantLatent, 30)
	checkSNR(t, "audio", song.Audio, want, 30)
	// The chunk boundary is where the two renders differ: 70 frames in.
	single := fix.read(t, "pipeline/short/latent")
	t.Logf("chunked against single render: first chunk %.2f dB, second %.2f dB",
		snrDB(wantLatent[:70*64], single[:70*64]), snrDB(wantLatent[70*64:], single[70*64:]))
}

// TestPipelineSeal is the first chunk of a generated song: the codes are
// decoded token by token (GenerateForced), so the cache already holds the
// whole stream when the acoustic stage starts, and the first chunk is sealed
// by trimming it and forwarding the end token rather than prefilled again.
// Without a score the request renders with cot off: a prompt of 563 and
// guidance in a second slot. The context is cut for chunks of 70, as in
// TestPipelineChunks, and the render is held to the replay of the same
// request, whose first chunk is prefilled whole: the same keys by another
// road, so only the order of the sums may differ.
func TestPipelineSeal(t *testing.T) {
	p := sharedPipeline(t)
	r := shortRequest(t)
	r.ABC = ""
	prefix := BuildPrompt(p.tok, CotOff, r.Style, r.Lyrics, nil, false)
	p.context = len(prefix) + 3 + 2*70
	defer func() { p.context = pipelineContext }()
	replay, err := p.Generate(r, nil)
	if err != nil {
		t.Fatal(err)
	}
	sealed, err := p.GenerateForced(r, nil)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("prompt %d, context %d; replay %v, forced %v", len(prefix), p.context, replay.Times, sealed.Times)
	if sealed.Score != "" || sealed.Truncated || !reflect.DeepEqual(sealed.Codes, replay.Codes) {
		t.Errorf("score %q, truncated %v, codes equal %v", sealed.Score, sealed.Truncated, reflect.DeepEqual(sealed.Codes, replay.Codes))
	}
	checkSNR(t, "latents, first chunk", sealed.Latents[:70*64], replay.Latents[:70*64], 60)
	checkSNR(t, "latents, second chunk", sealed.Latents[70*64:], replay.Latents[70*64:], 60)
	checkSNR(t, "audio", sealed.Audio, replay.Audio, 60)
}

// TestRequestReplayShapes reads the recorded replay and checks what each
// shape of request asks of the stages.
func TestRequestReplayShapes(t *testing.T) {
	raw, err := os.ReadFile(filepath.Join("..", "testdata", "yue2", "bench", "guerreros-replay.json"))
	if err != nil {
		t.Skipf("no recorded replay (%v): see ref/yue2/README.md", err)
	}
	r, err := ParseRequest(raw)
	if err != nil {
		t.Fatal(err)
	}
	// The seeds are past 2^53: a float64 would round them.
	if r.LMSeed != arSeed || r.Seed != pipelineSeed {
		t.Errorf("seeds %d and %d, want %d and %d", r.LMSeed, r.Seed, arSeed, pipelineSeed)
	}
	codes, err := ParseCodes(r.SemanticTokens)
	if err != nil {
		t.Fatal(err)
	}
	if len(codes) != 4870 || r.OutputFormat != FormatWAV16 || r.Steps != 32 || r.Duration != 360 {
		t.Errorf("%d codes, format %s, %d steps, %g s", len(codes), r.OutputFormat, r.Steps, r.Duration)
	}
	if FormatCodes(codes) != r.SemanticTokens {
		t.Error("the codes do not write back as they were read")
	}

	for _, c := range []struct {
		name               string
		cot, abc, codes    string
		want               Cot
		score, planned, re bool
	}{
		{"cot absent, a score and codes", "", "X:1", "1,2", CotFull, true, false, true},
		{"melody, a score and codes", "melody", "X:1", "1,2", CotMelody, true, false, true},
		{"full, codes without a score", "full", "", "1,2", CotOff, false, false, true},
		{"melody, codes without a score", "melody", "", "1,2", CotOff, false, false, true},
		{"full, nothing given", "", "", "", CotFull, true, true, false},
		{"full, a score", "full", "X:1", "", CotFull, true, false, false},
		{"off, a score and codes", "off", "X:1", "1,2", CotOff, false, false, true},
	} {
		q := DefaultRequest()
		if c.cot != "" {
			q.Cot = c.cot
		}
		q.ABC, q.SemanticTokens = c.abc, c.codes
		pl, err := planOf(q)
		if err != nil {
			t.Fatal(err)
		}
		if pl.cot != c.want || pl.hasScore != c.score || pl.planned != c.planned || pl.replay != c.re {
			t.Errorf("%s: %+v, want cot %d, score %v, planned %v, replay %v", c.name, pl, c.want, c.score, c.planned, c.re)
		}
	}
	if _, err := planOf(Request{Cot: "some"}); err == nil {
		t.Error("an unknown cot was taken")
	}

	// A replay of a song: its score, its codes and the seeds it consumed.
	song := &Song{Codes: codes[:3], LMSeed: 7, Seed: 8}
	q := DefaultRequest()
	q.Style, q.ABC, q.LMBatchSize, q.SourceTokens = "s", "X:2", 4, "9"
	rep := song.Replay(q)
	if rep.ABC != "X:2" || rep.SemanticTokens != FormatCodes(codes[:3]) || rep.LMSeed != 7 || rep.Seed != 8 ||
		rep.LMBatchSize != 1 || rep.SourceTokens != "" || rep.Style != "s" {
		t.Errorf("replay %+v", rep)
	}
	song.Score = "X:3"
	if song.Replay(q).ABC != "X:3" {
		t.Error("the replay does not carry the song's score")
	}

	// The JSON writes the fields away from their defaults and reads back.
	if got := string(DefaultRequest().JSON()); got != "{}\n" {
		t.Errorf("the default request writes %q", got)
	}
	r.ABCSampling.TopK = 12
	r.CfgScale = 1.5
	r.Adapters = []Adapter{{Name: "a", Scale: 0.5}}
	back, err := ParseRequest(r.JSON())
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(back, r) {
		t.Errorf("the request does not read back as written:\n%s", r.JSON())
	}

	// A member of the wrong type keeps its default, as yyjson's getters.
	odd, err := ParseRequest([]byte(`{"seed": 1.5, "steps": "8", "lm_seed": 12, "duration": 30, "cot": null, "output_format": null}`))
	if err != nil {
		t.Fatal(err)
	}
	if odd.Seed != -1 || odd.Steps != 32 || odd.LMSeed != 12 || odd.Duration != 30 || odd.Cot != "full" || odd.OutputFormat != FormatMP3 {
		t.Errorf("seed %d, steps %d, lm_seed %d, duration %g, cot %q, format %q", odd.Seed, odd.Steps, odd.LMSeed, odd.Duration, odd.Cot, odd.OutputFormat)
	}
	res := odd.Resolve()
	if res.LMSeed != 12 || res.Seed < 0 {
		t.Errorf("resolved seeds %d and %d", res.LMSeed, res.Seed)
	}
	if _, err := ParseCodes("1,32768"); err == nil {
		t.Error("a code past the codec was taken")
	}
}

// TestNormalize holds Normalize and the WAV writers to yue2.cpp's
// audio_normalize and audio_encode_wav on the short replay's audio
// (dump_audio's normalize/), bit for bit.
func TestNormalize(t *testing.T) {
	fix := loadFixtures(t)
	audio := fix.read(t, "pipeline/short/audio")
	samples := fix.shape(t, "pipeline/short/audio")[1]
	for _, clip := range []struct {
		name string
		clip int
	}{{"normalize/out_0", 0}, {"normalize/out_10", 10}, {"normalize/out_999", 999}} {
		want := fix.read(t, clip.name)
		got := append([]float32(nil), audio...)
		Normalize(got, clip.clip)
		diff := 0
		for i := range want {
			if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
				diff++
			}
		}
		if diff > 0 {
			t.Errorf("peak_clip %d: %d of %d samples differ", clip.clip, diff, len(want))
		}
	}
	bytesOf := func(name string) []byte {
		fix.shape(t, name)
		b, err := os.ReadFile(filepath.Join(fix.dir, name))
		if err != nil {
			t.Fatal(err)
		}
		return b
	}
	song := &Song{Audio: audio, Samples: samples}
	for _, f := range []struct{ format, fixture string }{
		{FormatWAV16, "normalize/wav16"}, {FormatWAV24, "normalize/wav24"}, {FormatWAV32, "normalize/wav32"},
	} {
		got, err := song.Encode(f.format, 10)
		if err != nil {
			t.Fatal(err)
		}
		if !bytes.Equal(got, bytesOf(f.fixture)) {
			t.Errorf("%s: %d bytes differ from yue2.cpp's %d", f.format, len(got), len(bytesOf(f.fixture)))
		}
	}
	if _, err := song.Encode(FormatMP3, 10); err == nil {
		t.Error("an MP3 was asked for and something came back")
	}
	if !reflect.DeepEqual(audio, fix.read(t, "pipeline/short/audio")) {
		t.Error("Encode changed the song's audio")
	}
}
