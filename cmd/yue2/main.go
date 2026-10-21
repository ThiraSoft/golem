// Command yue2 renders a song from a yue2.cpp request with YuE2: the score,
// the semantic codes, the flow matching and the decoder, all on the card.
//
//	yue2 -request song.json -out song.wav
//	yue2 -request replay.json -out song.wav -forced
//
// A request with semantic_tokens is a replay: the codes are prefilled rather
// than drawn. -forced decodes them token by token instead, every step
// computing its distribution and then taking the given token, which costs
// what drawing them would: the benchmark's mode. Beside the audio goes
// <out>.replay.json, the request that renders the same song again.
//
// -tokens and -score write the codes and the score as yue-synth's --tokens
// and --score do. -compare takes the prefix of a yue-synth run (the one
// ref/yue2/bench_yue2cpp.sh writes) and says where this song first leaves
// its <prefix>.abc and <prefix>.tokens.csv: the benchmark's same-seeds line.
// When the two songs have the same length it also gives the SNR of this
// audio against <prefix>.wav.
package main

import (
	"bytes"
	"flag"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"time"

	"github.com/ThiraSoft/golem/audio/wav"
	"github.com/ThiraSoft/golem/yue2"
)

func main() {
	dir := flag.String("dir", "", "the directory of YuE2-3B-<model>.gguf and YuE2-Vae-F32.gguf (default GOLEM_YUE2_DIR, or "+
		"/mnt/data/LLMs_models/Serveurperso/YuE2-GGUF)")
	model := flag.String("model", "Q8_0", "the backbone's format")
	request := flag.String("request", "", "the request JSON, yue2.cpp's")
	out := flag.String("out", "song.wav", "where to write the audio")
	forced := flag.Bool("forced", false, "decode the request's score and codes token by token rather than replay them")
	context := flag.Int("context", 0, "the cache in positions, yue2.cpp's --max-seq; 0 is the model's whole context")
	tokens := flag.String("tokens", "", "also write the semantic codes as CSV, yue-synth's --tokens")
	score := flag.String("score", "", "also write the score, yue-synth's --score")
	compare := flag.String("compare", "", "the prefix of a yue-synth run: report where the score and codes first differ from <prefix>.abc and <prefix>.tokens.csv")
	flag.Parse()

	if *request == "" {
		fmt.Fprintln(os.Stderr, "yue2: -request is required")
		os.Exit(2)
	}
	r, err := yue2.ReadRequest(*request)
	if err != nil {
		fail(err)
	}
	format := r.OutputFormat
	if format == yue2.FormatMP3 {
		fmt.Fprintln(os.Stderr, "yue2: golem writes no MP3, writing wav16")
		format = yue2.FormatWAV16
	}

	vram := vramUsed()
	start := time.Now()
	p, err := yue2.Open(yue2.Options{Dir: *dir, Model: *model, Context: *context})
	if err != nil {
		fail(err)
	}
	defer p.Close()
	load := time.Since(start)
	if after := vramUsed(); vram >= 0 && after >= 0 {
		fmt.Fprintf(os.Stderr, "[Load] %.1f s, %.2f GB on the card, context %d\n", load.Seconds(), float64(after-vram)/1e9, p.Context())
	} else {
		fmt.Fprintf(os.Stderr, "[Load] %.1f s, context %d\n", load.Seconds(), p.Context())
	}

	generate := p.Generate
	if *forced {
		generate = p.GenerateForced
	}
	song, err := generate(r, nil)
	if err != nil {
		fail(err)
	}
	// What the song left on the card: the open's figure, plus the second
	// cache slot when the codes were drawn with guidance.
	if after := vramUsed(); vram >= 0 && after >= 0 {
		fmt.Fprintf(os.Stderr, "[Card] %.2f GB after the song\n", float64(after-vram)/1e9)
	}
	t := time.Now()
	audio, err := song.Encode(format, r.PeakClip)
	if err != nil {
		fail(err)
	}
	encode := time.Since(t)
	if err := os.WriteFile(*out, audio, 0o644); err != nil {
		fail(err)
	}
	replay := strings.TrimSuffix(*out, filepath.Ext(*out)) + ".replay.json"
	if err := os.WriteFile(replay, song.Replay(r).JSON(), 0o644); err != nil {
		fail(err)
	}

	if *tokens != "" {
		if err := os.WriteFile(*tokens, []byte(yue2.FormatCodes(song.Codes)+"\n"), 0o644); err != nil {
			fail(err)
		}
	}
	if *score != "" {
		if err := os.WriteFile(*score, []byte(song.Score), 0o644); err != nil {
			fail(err)
		}
	}

	seconds := float64(song.Samples) / yue2.SampleRate
	for _, stage := range []string{"score", "semantic", "prefill", "nar", "vae"} {
		if d, ok := song.Times[stage]; ok {
			fmt.Fprintf(os.Stderr, "[Time] %-8s %8.2f s\n", stage, d.Seconds())
		}
	}
	fmt.Fprintf(os.Stderr, "[Time] %-8s %8.2f s\n", "encode", encode.Seconds())
	fmt.Fprintf(os.Stderr, "[Time] %-8s %8.2f s (%.1f s of audio, %.2fx real time; load %.1f s apart)\n",
		"total", song.Times["total"].Seconds()+encode.Seconds(), seconds,
		seconds/(song.Times["total"].Seconds()+encode.Seconds()), load.Seconds())
	trunc := ""
	if song.Truncated {
		trunc = ", truncated"
	}
	fmt.Fprintf(os.Stderr, "[Done] %d frames%s, seeds %d and %d -> %s, %s\n", song.Frames, trunc, song.LMSeed, song.Seed, *out, replay)

	if *compare != "" {
		if err := compareRun(p.Tokenizer(), song, audio, *compare); err != nil {
			fail(err)
		}
	}
}

// compareRun reports where song first leaves a yue-synth run's score and
// codes. The score is compared in tokens, both texts through the backbone's
// tokenizer, because a token is what the sampler draws.
func compareRun(tok *yue2.Tokenizer, song *yue2.Song, audio []byte, prefix string) error {
	abc, err := os.ReadFile(prefix + ".abc")
	if err != nil {
		return err
	}
	csv, err := os.ReadFile(prefix + ".tokens.csv")
	if err != nil {
		return err
	}
	codes, err := yue2.ParseCodes(string(csv))
	if err != nil {
		return err
	}
	got, want := tok.Encode(song.Score), tok.Encode(string(abc))
	fmt.Fprintf(os.Stderr, "[Compare] score: %d tokens against %d, text %s, %s\n",
		len(got), len(want), equalWord(song.Score == string(abc)), firstDifference(got, want))
	fmt.Fprintf(os.Stderr, "[Compare] codes: %d against %d, %s\n", len(song.Codes), len(codes), firstDifference(song.Codes, codes))

	f, err := os.Open(prefix + ".wav")
	if err != nil {
		return err
	}
	ref, _, _, err := wav.ReadChannels(f)
	f.Close()
	if err != nil {
		return err
	}
	ours, _, _, err := wav.ReadChannels(bytes.NewReader(audio))
	if err != nil {
		return err
	}
	if len(ours) != len(ref) {
		fmt.Fprintf(os.Stderr, "[Compare] audio: %d samples against %d, no SNR\n", len(ours), len(ref))
		return nil
	}
	fmt.Fprintf(os.Stderr, "[Compare] audio: SNR %.2f dB against %s.wav\n", snr(ours, ref), prefix)
	return nil
}

// snr is the signal to noise ratio of got against ref in decibels, as
// TestGuerrerosReplay measures it.
func snr(got, ref []float32) float64 {
	var num, den float64
	for i := range ref {
		d := float64(got[i] - ref[i])
		num += d * d
		den += float64(ref[i]) * float64(ref[i])
	}
	return 10 * math.Log10(den/num)
}

func equalWord(same bool) string {
	if same {
		return "equal"
	}
	return "different"
}

// firstDifference names the first index where a and b disagree, counting
// the end of the shorter one as a disagreement.
func firstDifference(a, b []int32) string {
	if slices.Equal(a, b) {
		return "identical"
	}
	n := min(len(a), len(b))
	for i := range n {
		if a[i] != b[i] {
			return fmt.Sprintf("first difference at index %d", i)
		}
	}
	return fmt.Sprintf("identical over the first %d, then one ends", n)
}

// vramUsed is what the amdgpu driver says the card holds, or -1.
func vramUsed() int64 {
	paths, _ := filepath.Glob("/sys/class/drm/card*/device/mem_info_vram_used")
	for _, p := range paths {
		b, err := os.ReadFile(p)
		if err != nil {
			continue
		}
		if v, err := strconv.ParseInt(strings.TrimSpace(string(b)), 10, 64); err == nil {
			return v
		}
	}
	return -1
}

func fail(err error) {
	fmt.Fprintln(os.Stderr, "yue2:", err)
	os.Exit(1)
}
