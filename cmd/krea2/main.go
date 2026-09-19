// Command krea2 draws a picture from a sentence with Krea 2, the way the
// ComfyUI mobile front draws it: the same three files, the same sampler, and
// for the same seed the same picture.
//
//	krea2 -prompt "a cat on a wooden table" -seed 42 -out cat.png
//
// It needs a Vulkan device with cooperative matrices and about thirteen
// gigabytes on it.
package main

import (
	"flag"
	"fmt"
	"image/png"
	"math/rand"
	"os"
	"time"

	"github.com/ThiraSoft/golem/krea2"
)

func main() {
	prompt := flag.String("prompt", "", "what to draw")
	negative := flag.String("negative", "", "what not to draw; read only when -cfg is not 1")
	width := flag.Int("width", 768, "width, a multiple of 16")
	height := flag.Int("height", 1024, "height, a multiple of 16")
	steps := flag.Int("steps", 8, "sampling steps")
	cfg := flag.Float64("cfg", 1, "classifier-free guidance; 1 skips the negative prompt")
	seed := flag.Int64("seed", -1, "the seed; -1 draws one")
	out := flag.String("out", "krea2.png", "where to write the picture")
	n := flag.Int("n", 1, "draw this many pictures, the seed going up by one each time, the DiT kept between them")
	dit := flag.String("dit", krea2.DiTPath(), "the DiT, fp8")
	encoder := flag.String("encoder", krea2.EncoderPath(), "the text encoder, Qwen3-VL-4B fp8-scaled")
	vae := flag.String("vae", krea2.VAEPath(), "the VAE")
	tokenizer := flag.String("tokenizer", krea2.TokenizerDir(), "a directory with vocab.json, merges.txt and tokenizer_config.json")
	flag.Parse()
	if *prompt == "" {
		fmt.Fprintln(os.Stderr, "krea2: -prompt is required")
		os.Exit(2)
	}
	if *seed < 0 {
		*seed = rand.Int63()
	}

	start := time.Now()
	p, err := krea2.Open(krea2.Options{Encoder: *encoder, DiT: *dit, VAE: *vae, Tokenizer: *tokenizer,
		Keep: *n > 1, MaxPixels: max(*width**height, 1024*1024)})
	if err != nil {
		fail(err)
	}
	defer p.Close()
	fmt.Fprintf(os.Stderr, "open %v\n", time.Since(start).Round(time.Millisecond))

	for i := 0; i < *n; i++ {
		r := krea2.Request{Prompt: *prompt, Negative: *negative, Width: *width, Height: *height, Steps: *steps,
			CFG: float32(*cfg), Seed: uint64(*seed + int64(i))}
		img, tm, err := p.Generate(r, func(step, steps int) { fmt.Fprintf(os.Stderr, "\rstep %d/%d", step, steps) })
		if err != nil {
			fail(err)
		}
		path := *out
		if *n > 1 {
			path = fmt.Sprintf("%s.%d.png", *out, r.Seed)
		}
		f, err := os.Create(path)
		if err != nil {
			fail(err)
		}
		if err := png.Encode(f, img); err != nil {
			fail(err)
		}
		f.Close()
		fmt.Fprintf(os.Stderr, "\r%s: seed %d, encode %v, load %v, sample %v, decode %v, total %v\n", path, r.Seed,
			tm.Encode.Round(time.Millisecond), tm.Load.Round(time.Millisecond), tm.Sample.Round(time.Millisecond),
			tm.Decode.Round(time.Millisecond), tm.Total.Round(time.Millisecond))
	}
}

func fail(err error) {
	fmt.Fprintln(os.Stderr, "krea2:", err)
	os.Exit(1)
}
