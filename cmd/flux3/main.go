// Command flux3 draws a picture from a sentence with FLUX 3 Image Klein, the way
// the ComfyUI mobile front draws it: the same three files, the same sampler, and
// for the same seed the same picture.
//
//	flux3 -prompt "a cat on a wooden table" -seed 42 -out cat.png
//	flux3 -prompt "..." -image photo.jpg -match-image -out out.png
//
// It draws at 2048 x 2048 by default, the workflows' 4 MP; -match-image
// sizes the picture after the first -image instead. It needs a Vulkan
// device with cooperative matrices and about sixteen gigabytes on it.
package main

import (
	"flag"
	"fmt"
	"image"
	_ "image/jpeg"
	_ "image/png"
	"math/rand"
	"os"
	"strings"
	"time"

	"github.com/ThiraSoft/golem/flux3"
	_ "golang.org/x/image/webp"
)

type stringList []string

func (s *stringList) String() string {
	return strings.Join(*s, ",")
}

func (s *stringList) Set(val string) error {
	*s = append(*s, val)
	return nil
}

func main() {
	prompt := flag.String("prompt", "", "what to draw")
	negative := flag.String("negative", "", "what not to draw; read only when -cfg is not 1")
	width := flag.Int("width", 2048, "width, a multiple of 16")
	height := flag.Int("height", 2048, "height, a multiple of 16")
	matchImage := flag.Bool("match-image", false, "size the picture after the first -image scaled to 4 MP, as the workflows' match_input_image")
	steps := flag.Int("steps", 16, "sampling steps")
	cfg := flag.Float64("cfg", 1, "classifier-free guidance; 1 skips the negative prompt")
	seed := flag.Int64("seed", -1, "the seed; -1 draws one")
	out := flag.String("out", "flux3.png", "where to write the picture")
	n := flag.Int("n", 1, "draw this many pictures, the seed going up by one each time, the DiT kept between them")
	dit := flag.String("dit", flux3.DiTPath(), "the DiT safetensors")
	encoder := flag.String("encoder", flux3.EncoderPath(), "the text encoder, Qwen3-VL-4B fp8-scaled")
	vae := flag.String("vae", flux3.VAEPath(), "the VAE")
	tokenizer := flag.String("tokenizer", flux3.TokenizerDir(), "a directory with vocab.json, merges.txt and tokenizer_config.json")
	fp8 := flag.Bool("fp8", false, "store DiT weights in FP8 instead of FP16 (default is FP16)")
	mp := flag.Float64("megapixels", 1.0, "megapixels for reference image scaling")
	hidePrompt := flag.Bool("hide-prompt", false, "leave the prompt and the negative prompt out of the PNG's metadata")

	var imagePaths stringList
	flag.Var(&imagePaths, "image", "reference image path (can be repeated)")
	flag.Parse()

	if *prompt == "" {
		fmt.Fprintln(os.Stderr, "flux3: -prompt is required")
		os.Exit(2)
	}
	if *seed < 0 {
		*seed = rand.Int63()
	}
	if *matchImage && len(imagePaths) == 0 {
		fmt.Fprintln(os.Stderr, "flux3: -match-image needs an -image")
		os.Exit(2)
	}

	var refImages []image.Image
	for _, p := range imagePaths {
		f, err := os.Open(p)
		if err != nil {
			fail(err)
		}
		img, err := flux3.DecodeImage(f)
		f.Close()
		if err != nil {
			fail(fmt.Errorf("decode reference image %s: %w", p, err))
		}
		refImages = append(refImages, img)
	}

	prec := flux3.PrecisionFP16
	if *fp8 {
		prec = flux3.PrecisionFP8
	}

	start := time.Now()
	p, err := flux3.Open(flux3.Options{
		Encoder:   *encoder,
		DiT:       *dit,
		VAE:       *vae,
		Tokenizer: *tokenizer,
		Keep:      *n > 1,
		Precision: prec,
	})
	if err != nil {
		fail(err)
	}
	defer p.Close()
	fmt.Fprintf(os.Stderr, "open %v\n", time.Since(start).Round(time.Millisecond))

	for i := 0; i < *n; i++ {
		r := flux3.Request{
			Prompt:     *prompt,
			Negative:   *negative,
			Width:      *width,
			Height:     *height,
			Steps:      *steps,
			CFG:        float32(*cfg),
			Seed:       uint64(*seed + int64(i)),
			Images:     refImages,
			MegaPixels: *mp,
			HidePrompt: *hidePrompt,
			MatchImage: *matchImage,
		}
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
		if err := p.WritePNG(f, img, r); err != nil {
			fail(err)
		}
		f.Close()
		fmt.Fprintf(os.Stderr, "\r%s: seed %d, encode %v, load %v, sample %v, decode %v, total %v\n", path, r.Seed,
			tm.Encode.Round(time.Millisecond), tm.Load.Round(time.Millisecond), tm.Sample.Round(time.Millisecond),
			tm.Decode.Round(time.Millisecond), tm.Total.Round(time.Millisecond))
	}
}

func fail(err error) {
	fmt.Fprintln(os.Stderr, "flux3:", err)
	os.Exit(1)
}
