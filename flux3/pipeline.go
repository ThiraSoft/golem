package flux3

import (
	"fmt"
	"image"
	"image/color"
	"io"
	"math"
	"path/filepath"
	"strings"
	"time"

	"github.com/ThiraSoft/golem/internal/version"
	"github.com/ThiraSoft/golem/krea2"
	"github.com/ThiraSoft/golem/token/bytebpe"
	"github.com/ThiraSoft/golem/vk"
	"golang.org/x/image/draw"
)

// Request is one picture, with the fields the mobile front sends.
type Request struct {
	Prompt     string        `json:"prompt"`
	Negative   string        `json:"negative"`
	Width      int           `json:"width"`
	Height     int           `json:"height"`
	Steps      int           `json:"steps"`
	CFG        float32       `json:"cfg"`
	Seed       uint64        `json:"seed"`
	Images     []image.Image `json:"images,omitempty"`
	HidePrompt bool          `json:"hide_prompt,omitempty"`
	MegaPixels float64       `json:"megapixels,omitempty"`
}

// Parameters is the request as a picture's "parameters" text.
func (r Request) Parameters(model string) string {
	var b strings.Builder
	b.WriteString(r.Prompt)
	if r.Negative != "" {
		fmt.Fprintf(&b, "\nNegative prompt: %s", r.Negative)
	}
	fmt.Fprintf(&b, "\nSteps: %d, Sampler: euler, Schedule type: simple, CFG scale: %g, Seed: %d, Size: %dx%d, Model: %s, Version: golem %s",
		r.Steps, r.CFG, r.Seed, r.Width, r.Height, model, version.String())
	return b.String()
}

// Pipeline holds the Vulkan device, tokenizer, text encoder in system memory,
// the VAE, and the DiT.
type Pipeline struct {
	o   Options
	d   *vk.Device
	tok *krea2.Tokenizer
	enc *krea2.Encoder
	dit *DiT
	vae *VAE

	onStep     func(step int, x []float32)
	lastLatent []float32
	lastTokens [2]int
	saveLatent bool
}

// Open readies a pipeline. The DiT is loaded on the first picture if not kept.
func Open(o Options) (*Pipeline, error) {
	if o.Encoder == "" {
		o.Encoder = EncoderPath()
	}
	if o.DiT == "" {
		o.DiT = DiTPath()
	}
	if o.VAE == "" {
		o.VAE = VAEPath()
	}
	if o.Tokenizer == "" {
		o.Tokenizer = TokenizerDir()
	}
	if o.MaxPixels == 0 {
		o.MaxPixels = 1024 * 1024
	}
	if o.MaxPixels > MaxImageTokens*256 {
		return nil, fmt.Errorf("flux3: MaxPixels %d exceeds limit of %d (1 MP DiT limit)", o.MaxPixels, MaxImageTokens*256)
	}
	p := &Pipeline{o: o}
	fail := func(err error) (*Pipeline, error) {
		p.Close()
		return nil, err
	}
	var err error
	if p.tok, err = bytebpe.LoadFiles(o.Tokenizer); err != nil {
		return fail(fmt.Errorf("flux3: tokenizer: %w", err))
	}
	if p.d, err = vk.Open(); err != nil {
		return fail(err)
	}
	// Keep text encoder in host memory (host=true) like krea2
	if p.enc, err = OpenEncoder(p.d, o.Encoder, true); err != nil {
		return fail(err)
	}
	p.enc.Trim()
	// Open VAE with maxLatentPix = o.MaxPixels / 256 (downscale 16x -> 16*16 = 256)
	if p.vae, err = OpenVAE(p.d, o.VAE, o.MaxPixels/256); err != nil {
		return fail(err)
	}
	p.vae.Trim()
	return p, nil
}

func (p *Pipeline) check(r Request) error {
	switch {
	case r.Width < 64 || r.Height < 64 || r.Width%16 != 0 || r.Height%16 != 0:
		return fmt.Errorf("flux3: %d × %d: width and height are multiples of 16, from 64", r.Width, r.Height)
	case r.Width*r.Height > p.o.MaxPixels:
		return fmt.Errorf("flux3: %d × %d is past the %d pixels this pipeline was opened for (1 MP DiT limit)", r.Width, r.Height, p.o.MaxPixels)
	case r.Steps < 1 || r.Steps > 100:
		return fmt.Errorf("flux3: %d steps", r.Steps)
	case r.CFG <= 0:
		return fmt.Errorf("flux3: a CFG of %g", r.CFG)
	}
	return nil
}

var lanczosKernel = &draw.Kernel{
	Support: 3,
	At: func(t float64) float64 {
		if t == 0 {
			return 1
		}
		pit := math.Pi * t
		return (math.Sin(pit) / pit) * (math.Sin(pit/3) / (pit / 3))
	},
}

// ResizeReference reproduces ReferenceMultiLatent's resize:
// scale = sqrt(mp * 2^20 / (w * h)), then lanczos scaling to (rw, rh).
// The target dimensions are capped so that the resulting latent patches
// do not exceed maxTokens.
func ResizeReference(img image.Image, mp float64, maxTokens int) ([]float32, int, int) {
	bounds := img.Bounds()
	w, h := bounds.Dx(), bounds.Dy()
	if w <= 0 || h <= 0 {
		return nil, 0, 0
	}
	if mp <= 0 {
		mp = 1.0
	}
	// Cap mp so tokens fit in maxTokens (each token is 16x16 = 256 pixels)
	if maxTokens > 0 {
		maxPix := float64(maxTokens * 256)
		if mp*1024*1024 > maxPix {
			mp = maxPix / (1024 * 1024)
		}
	}

	scale := math.Sqrt(mp * 1024 * 1024 / float64(w*h))
	rw := max(1, int(math.RoundToEven(float64(w)*scale)))
	rh := max(1, int(math.RoundToEven(float64(h)*scale)))

	if maxTokens > 0 {
		for (rw/16)*(rh/16) > maxTokens && (rw > 16 || rh > 16) {
			if rw > rh && rw > 16 {
				rw -= 16
			} else if rh > 16 {
				rh -= 16
			} else {
				break
			}
		}
	}

	img = ensureOpaque(img)
	bounds = img.Bounds()
	dst := image.NewRGBA(image.Rect(0, 0, rw, rh))
	lanczosKernel.Scale(dst, dst.Bounds(), img, bounds, draw.Src, nil)

	rgb := make([]float32, rw*rh*3)
	for y := 0; y < rh; y++ {
		for x := 0; x < rw; x++ {
			c := dst.RGBAAt(x, y)
			idx := (y*rw + x) * 3
			rgb[idx+0] = float32(c.R) / 255.0
			rgb[idx+1] = float32(c.G) / 255.0
			rgb[idx+2] = float32(c.B) / 255.0
		}
	}
	return rgb, rw, rh
}

// ensureOpaque ensures reference images with alpha are composited onto opaque
// (straight RGB like PIL convert("RGB"), via NRGBA).
func ensureOpaque(img image.Image) image.Image {
	if op, ok := img.(interface{ Opaque() bool }); ok && op.Opaque() {
		return img
	}
	bounds := img.Bounds()
	w, h := bounds.Dx(), bounds.Dy()
	opaque := image.NewRGBA(image.Rect(0, 0, w, h))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			c := color.NRGBAModel.Convert(img.At(bounds.Min.X+x, bounds.Min.Y+y)).(color.NRGBA)
			opaque.SetRGBA(x, y, color.RGBA{R: c.R, G: c.G, B: c.B, A: 255})
		}
	}
	return opaque
}

// Generate draws one picture. progress, if set, hears about each step.
func (p *Pipeline) Generate(r Request, progress krea2.Progress) (*image.NRGBA, krea2.Timings, error) {
	var tm krea2.Timings
	start := time.Now()
	if r.Steps == 0 {
		r.Steps = Defaults.Steps
	}
	if r.CFG == 0 {
		r.CFG = Defaults.CFG
	}
	if err := p.check(r); err != nil {
		return nil, tm, err
	}
	cfg := r.CFG != 1.0

	// 1. Text encoding
	posIDs, posReal := Tokenize(p.tok, r.Prompt)
	if len(posIDs) > MaxTextTokens {
		return nil, tm, fmt.Errorf("flux3: prompt token count %d exceeds limit of %d", len(posIDs), MaxTextTokens)
	}

	var negIDs []int32
	var negReal int
	if cfg {
		negIDs, negReal = Tokenize(p.tok, r.Negative)
		if len(negIDs) > MaxTextTokens {
			return nil, tm, fmt.Errorf("flux3: negative prompt token count %d exceeds limit of %d", len(negIDs), MaxTextTokens)
		}
	}

	pos, err := p.enc.EncodeMasked(posIDs, posReal)
	if err != nil {
		return nil, tm, err
	}
	posTokens := len(posIDs)

	var neg []float32
	var negTokens int
	if cfg {
		neg, err = p.enc.EncodeMasked(negIDs, negReal)
		if err != nil {
			return nil, tm, err
		}
		negTokens = len(negIDs)
	}
	p.enc.Trim()
	tm.Encode = time.Since(start)

	// 2. Reference images encoding (with VAE)
	var refLatents [][]float32
	var refShapes [][2]int
	if len(r.Images) > 0 {
		budget := MaxRefTokens / len(r.Images)
		mp := r.MegaPixels
		if mp <= 0 {
			mp = 1.0
		}
		for _, img := range r.Images {
			rgb, rw, rh := ResizeReference(img, mp, budget)
			lat, err := p.vae.Encode(rgb, rh, rw)
			if err != nil {
				return nil, tm, err
			}
			refLatents = append(refLatents, lat)
			refShapes = append(refShapes, [2]int{rh / 16, rw / 16})
		}
		p.vae.Trim()
	}

	// 3. Load / ready DiT
	tLoad := time.Now()
	if p.dit == nil {
		p.dit, err = OpenDiT(p.d, p.o.DiT, p.o.Precision)
		if err != nil {
			return nil, tm, err
		}
	}
	cleanupDiT := func() {
		if !p.o.Keep && p.dit != nil {
			p.dit.Close()
			p.dit = nil
		}
	}
	if err := p.dit.SetText(0, pos, posTokens); err != nil {
		cleanupDiT()
		return nil, tm, err
	}
	if err := p.dit.SetRefs(0, refLatents, refShapes); err != nil {
		cleanupDiT()
		return nil, tm, err
	}
	if cfg {
		if err := p.dit.SetText(1, neg, negTokens); err != nil {
			cleanupDiT()
			return nil, tm, err
		}
		if err := p.dit.SetRefs(1, refLatents, refShapes); err != nil {
			cleanupDiT()
			return nil, tm, err
		}
	}
	p.lastTokens = [2]int{p.dit.ctxTokens[0], p.dit.ctxTokens[1]}
	tm.Load = time.Since(tLoad)

	// 4. Sampling
	tSample := time.Now()
	h, w := r.Height/16, r.Width/16
	latent, err := p.sample(posTokens, negTokens, cfg, r, h, w, progress)
	if p.o.Keep {
		p.dit.Trim()
	} else {
		p.dit.Close()
		p.dit = nil
	}
	if err != nil {
		return nil, tm, err
	}
	if p.saveLatent {
		p.lastLatent = append([]float32(nil), latent...)
	}
	tm.Sample = time.Since(tSample)

	// 5. VAE Decode
	tDecode := time.Now()
	rgb, err := p.vae.Decode(latent, h, w)
	p.vae.Trim()
	if err != nil {
		return nil, tm, err
	}
	tm.Decode = time.Since(tDecode)
	tm.Total = time.Since(start)

	return krea2.Image(rgb, r.Width, r.Height), tm, nil
}

func (p *Pipeline) sample(posTokens, negTokens int, cfg bool, r Request, h, w int, progress krea2.Progress) ([]float32, error) {
	sigmas := Sigmas(r.Steps)
	denoise := func(slot int, x []float32, sigma, next float32) ([]float32, error) {
		v, err := p.dit.Step(slot, x, h, w, sigma, next)
		if err != nil {
			return nil, err
		}
		denoised := make([]float32, len(v))
		for i := range v {
			denoised[i] = x[i] - sigma*v[i]
		}
		return denoised, nil
	}
	d := func(step int, x []float32, sigma float32) ([]float32, error) {
		if p.onStep != nil {
			p.onStep(step, x)
		}
		nextIdx := step + 1
		if nextIdx >= len(sigmas) {
			nextIdx = len(sigmas) - 1
		}
		next := sigmas[nextIdx]

		c, err := denoise(0, x, sigma, next)
		if err != nil || !cfg {
			return c, err
		}
		u, err := denoise(1, x, sigma, next)
		if err != nil {
			return nil, err
		}
		return ApplyCFG(c, u, r.CFG)
	}
	x := StartingNoise(r.Seed, h, w)
	return Sample(d, x, sigmas, r.Seed, progress)
}

// WritePNG writes the picture as a PNG carrying the request that drew it.
func (p *Pipeline) WritePNG(w io.Writer, img image.Image, r Request) error {
	return WritePNG(w, img, r, strings.TrimSuffix(filepath.Base(p.o.DiT), ".safetensors"))
}

// WritePNG writes img as a PNG with three text chunks after its header:
// "parameters", "Software" and "golem", the request as JSON.
func WritePNG(w io.Writer, img image.Image, r Request, model string) error {
	reqCopy := r
	if reqCopy.HidePrompt {
		reqCopy.Prompt, reqCopy.Negative, reqCopy.HidePrompt = "", "", false
	}
	reqCopy.Images = nil
	return krea2.WritePNGWithMetadata(w, img, reqCopy.Parameters(model), reqCopy)
}

// Close frees all GPU and host resources.
func (p *Pipeline) Close() {
	if p.dit != nil {
		p.dit.Close()
		p.dit = nil
	}
	if p.vae != nil {
		p.vae.Close()
		p.vae = nil
	}
	if p.enc != nil {
		p.enc.Close()
		p.enc = nil
	}
	if p.d != nil {
		p.d.Close()
		p.d = nil
	}
}
