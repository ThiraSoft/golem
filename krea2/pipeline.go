package krea2

// A picture from a sentence: the text encoder, the sampler over the DiT, the
// VAE, in the order ComfyUI's graph runs them, phased so that the card holds
// one network's working memory at a time.

import (
	"fmt"
	"image"
	"path/filepath"
	"strings"
	"time"

	"github.com/ThiraSoft/golem/token/bytebpe"
	"github.com/ThiraSoft/golem/vk"
)

// Options says where the files are and what may stay on the card.
type Options struct {
	// The three checkpoints and the tokenizer; empty means ComfyUI's.
	Encoder, DiT, VAE, Tokenizer string
	// LoRAs is the directory a request's LoRA is named in; empty means
	// ComfyUI's.
	LoRAs string
	// Keep leaves the DiT's twelve gigabytes on the card between pictures,
	// which is most of the time a picture takes when it is not kept.
	Keep bool
	// MaxPixels bounds width × height and sizes the VAE; 1024 × 1024 when 0.
	MaxPixels int
}

// Request is one picture, with the fields the mobile front sends.
type Request struct {
	Prompt   string  `json:"prompt"`
	Negative string  `json:"negative"`
	Width    int     `json:"width"`
	Height   int     `json:"height"`
	Steps    int     `json:"steps"`
	CFG      float32 `json:"cfg"`
	Seed     uint64  `json:"seed"`
	// Lora is a file in Options.LoRAs, as the front's lora_name, and
	// LoraStrength how much of it, from 0 to 2; none when either is zero.
	Lora         string  `json:"lora,omitempty"`
	LoraStrength float32 `json:"lora_strength,omitempty"`
	// HidePrompt leaves the prompt and the negative prompt out of the
	// picture's metadata, and is left out itself; the settings, the seed
	// and the LoRA stay.
	HidePrompt bool `json:"hide_prompt,omitempty"`
}

// MaxLoRAStrength is the front's bound on a LoRA's strength.
const MaxLoRAStrength = 2

// Timings is where a picture's time went.
type Timings struct {
	Encode, Load, Sample, Decode, Total time.Duration
}

// Pipeline holds the device, the tokenizer, the text encoder in system
// memory, the VAE, and the DiT while it is kept.
type Pipeline struct {
	o     Options
	d     *vk.Device
	vocab *bytebpe.Vocab
	enc   *Encoder
	dit   *DiT
	vae   *VAE
	lora  *LoRA // the last one asked for, kept in memory
}

// Open readies a pipeline. The DiT is read at the first picture.
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
	if o.LoRAs == "" {
		o.LoRAs = LoRADir()
	}
	if o.MaxPixels == 0 {
		o.MaxPixels = 1024 * 1024
	}
	p := &Pipeline{o: o}
	fail := func(err error) (*Pipeline, error) {
		p.Close()
		return nil, err
	}
	var err error
	if p.vocab, err = bytebpe.LoadFiles(o.Tokenizer); err != nil {
		return fail(fmt.Errorf("krea2: tokenizer: %w", err))
	}
	if p.d, err = vk.Open(); err != nil {
		return fail(err)
	}
	if p.enc, err = OpenEncoder(p.d, o.Encoder, true); err != nil {
		return fail(err)
	}
	p.enc.Trim()
	if p.vae, err = OpenVAE(p.d, o.VAE, o.MaxPixels/64); err != nil {
		return fail(err)
	}
	p.vae.Trim()
	return p, nil
}

func (p *Pipeline) check(r Request) error {
	switch {
	case r.Width < 64 || r.Height < 64 || r.Width%16 != 0 || r.Height%16 != 0:
		return fmt.Errorf("krea2: %d × %d: width and height are multiples of 16, from 64", r.Width, r.Height)
	case r.Width*r.Height > p.o.MaxPixels:
		return fmt.Errorf("krea2: %d × %d is past the %d pixels this pipeline was opened for", r.Width, r.Height, p.o.MaxPixels)
	case r.Steps < 1 || r.Steps > 100:
		return fmt.Errorf("krea2: %d steps", r.Steps)
	case r.CFG <= 0:
		return fmt.Errorf("krea2: a CFG of %g", r.CFG)
	case r.LoraStrength < 0 || r.LoraStrength > MaxLoRAStrength:
		return fmt.Errorf("krea2: a LoRA strength of %g, from 0 to %d", r.LoraStrength, MaxLoRAStrength)
	case r.Lora != "" && (r.Lora != filepath.Base(r.Lora) || strings.HasPrefix(r.Lora, ".")):
		return fmt.Errorf("krea2: LoRA %q is not a file name", r.Lora)
	}
	return nil
}

// loraFor reads the request's LoRA, or keeps the one read last when it is
// the same; nil for none.
func (p *Pipeline) loraFor(r Request) (*LoRA, error) {
	if r.Lora == "" || r.LoraStrength == 0 {
		return nil, nil
	}
	path := filepath.Join(p.o.LoRAs, r.Lora)
	if p.lora != nil && p.lora.path == path {
		return p.lora, nil
	}
	if p.dit != nil {
		// The one on the card goes before this one is read: it is in memory
		// only for the card's sake.
		if err := p.dit.SetLoRA(nil, 0); err != nil {
			return nil, err
		}
	}
	p.lora = nil
	l, err := OpenLoRA(path)
	if err != nil {
		return nil, err
	}
	p.lora = l
	return l, nil
}

// Generate draws one picture. progress, if set, hears about each step.
func (p *Pipeline) Generate(r Request, progress Progress) (*image.NRGBA, Timings, error) {
	var tm Timings
	start := time.Now()
	if err := p.check(r); err != nil {
		return nil, tm, err
	}
	cfg := r.CFG != 1

	// The text, then the encoder's working memory back.
	encode := func(text string) ([]float32, int, error) {
		return p.enc.Encode(Tokenize(p.vocab, text))
	}
	pos, posLen, err := encode(r.Prompt)
	var neg []float32
	var negLen int
	if err == nil && cfg {
		neg, negLen, err = encode(r.Negative)
	}
	p.enc.Trim()
	if err != nil {
		return nil, tm, err
	}
	tm.Encode = time.Since(start)

	t := time.Now()
	lora, err := p.loraFor(r)
	if err != nil {
		return nil, tm, err
	}
	if p.dit == nil {
		if p.dit, err = OpenDiT(p.d, p.o.DiT); err != nil {
			return nil, tm, err
		}
	}
	if err := p.dit.SetLoRA(lora, r.LoraStrength); err != nil {
		return nil, tm, err
	}
	tm.Load = time.Since(t)
	t = time.Now()
	h, w := r.Height/8, r.Width/8
	latent, err := p.sample(pos, posLen, neg, negLen, cfg, r, h, w, progress)
	if p.o.Keep {
		p.dit.Trim()
	} else {
		p.dit.Close()
		p.dit = nil
	}
	if err != nil {
		return nil, tm, err
	}
	tm.Sample = time.Since(t)

	t = time.Now()
	LatentOut(latent)
	rgb, err := p.vae.Decode(latent, h, w)
	p.vae.Trim()
	if err != nil {
		return nil, tm, err
	}
	tm.Decode = time.Since(t)
	tm.Total = time.Since(start)
	return Image(rgb, r.Width, r.Height), tm, nil
}

// sample is KSampler's er_sde over the simple schedule, with ComfyUI's CFG
// when it is not 1: the negative's denoised plus CFG times the difference.
func (p *Pipeline) sample(pos []float32, posLen int, neg []float32, negLen int, cfg bool, r Request, h, w int, progress Progress) ([]float32, error) {
	if err := p.dit.SetText(0, pos, posLen); err != nil {
		return nil, err
	}
	if cfg {
		if err := p.dit.SetText(1, neg, negLen); err != nil {
			return nil, err
		}
	}
	denoise := func(slot int, x []float32, sigma float32) ([]float32, error) {
		v, err := p.dit.Step(slot, x, h, w, sigma)
		if err != nil {
			return nil, err
		}
		for i := range v {
			v[i] = x[i] - sigma*v[i]
		}
		return v, nil
	}
	d := func(step int, x []float32, sigma float32) ([]float32, error) {
		c, err := denoise(0, x, sigma)
		if err != nil || !cfg {
			return c, err
		}
		u, err := denoise(1, x, sigma)
		if err != nil {
			return nil, err
		}
		for i := range c {
			c[i] = u[i] + (c[i]-u[i])*r.CFG
		}
		return c, nil
	}
	// The starting latent is torch's CPU noise at the first sigma, which is
	// 1: the empty latent it is mixed with is weighed by nothing.
	x := TorchCPURandn(r.Seed, 16*h*w)
	return SampleERSDE(d, x, Sigmas(r.Steps), NewCUDARandn(r.Seed), progress)
}

// Close frees everything.
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
