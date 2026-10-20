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
package main

import (
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

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
