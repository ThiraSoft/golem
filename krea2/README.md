# krea2

Krea 2 from a sentence to a picture, the way the ComfyUI mobile front draws
it: the same three files, the same sampler, and for the same prompt and seed
the same picture.

| file, under ComfyUI's `models/` | what it is |
|---|---|
| `unet/krea2_turbo_fp8.safetensors` | the 12B single-stream DiT, fp8 e4m3 |
| `text_encoders/qwen3vl_4b_fp8_scaled.safetensors` | Qwen3-VL-4B, fp8 with a scale a tensor, read at twelve of its layers |
| `vae/qwen_image_vae.safetensors` | the Wan 2.1 VAE, of which the decoder |

They are not in this repository. `ref/krea2/README.md` says how the fixtures
the tests compare against are recorded.

```go
p, err := krea2.Open(krea2.ComfyUI("/mnt/data/dev/ComfyUI")) // or krea2.Options{...}
img, times, err := p.Generate(krea2.Request{
	Prompt: "a red fox in the snow, photo",
	Width: 768, Height: 1024, Steps: 8, CFG: 1, Seed: 5,
}, nil)
```

`cmd/krea2` draws one from the command line, and `golem-server -krea2 DIR`
serves `/v1/images/generations`.

It wants a Vulkan device with cooperative matrices and about fourteen
gigabytes on it: the DiT is twelve, the rest is working memory, taken and
given back one network at a time. The text encoder stays in system memory and
the card reads it across the bus, half a second a prompt, so that
`Options.Keep` can leave the DiT on the card between pictures.

## The same seed

ComfyUI draws two kinds of noise and both are drawn here bit for bit: the
starting latent is torch's CPU `randn` (mt19937 and its sixteen-wide
Box-Muller), and the noise ER-SDE adds at each step is torch's `randn` on a
ROCm card, which is rocrand's Philox4x32-10 spread over threads the way torch
launches its kernel on the RX 9070 XT (`rng.go`). Fed the model answers
ComfyUI recorded, the sampler goes through every latent ComfyUI went through,
to 1.5e-6.

The networks are float arithmetic done in another order, and there the two
part a little. ComfyUI computes the DiT in bf16: on one call it is 1.8% (rms)
from the same modules run in float32 on the CPU, and 6% by the middle of the
schedule. golem keeps float32 and feeds the matrix cores fp16, and is 0.7%
and 1.5% from float32. Rounding like bf16 was tried and brought the picture
further from ComfyUI's, not nearer: ComfyUI's rounding is ROCm's products',
which torch in bf16 on the CPU does not reproduce either.

So the pictures are the same to the eye and not to the bit. At the front's
defaults the recorded portrait is 30.7 dB from ComfyUI's; at 256 × 256, a size
the model was not made for, it is the same cat with other whiskers, 22 dB.

## Measured

RX 9070 XT, RADV, 768 × 1024, 8 steps, cfg 1, against ComfyUI on the same card
(ROCm 7.2, torch 2.11, its `start.sh` flags):

| | golem | ComfyUI |
|---|---:|---:|
| a DiT step | 1.38 s | 1.89 s |
| the eight steps | 11.4 s | 15 s |
| text encoder | 0.33 s | |
| VAE decode | 0.26 s | |
| a picture, DiT kept | 12.1 s | |
| reading the DiT, cold / cached | 14 s / 2.3 s | |

A step at that size is recorded as seven submissions of four blocks: the
driver allows a submission two seconds before it declares the card hung.

The DiT's products run at 52 to 58 TFLOPS in fp16, its attention at about 27.
