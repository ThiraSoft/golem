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

A LoRA of the DiT is named as the front names it, a file in ComfyUI's
`models/loras` and a strength from 0 to 2: `Request.Lora` and
`Request.LoraStrength`, `-lora` and `-lora-strength`, `lora` and
`lora_strength` in the server's request. The picture's metadata carries it,
`<lora:name:strength>` after the prompt. `HidePrompt` (`-hide-prompt`,
`hide_prompt`) leaves the prompt and the negative prompt out of it.

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
defaults the recorded portrait is 30.7 dB from ComfyUI's, and 24 dB at seeds
43 and 44 (`seeds/`); at 256 × 256, a size
the model was not made for, it is the same cat with other whiskers, 22 dB.

## LoRA

ComfyUI folds a LoRA into the weights two ways according to where it put the
module: one it holds on the card is patched once and rounded back to fp8
stochastically, one it left in system memory is patched in bf16 at every
call. The second is the arithmetic; the first adds a noise, weight by weight,
larger than the change. Which modules are which follows the card's free
memory: on the recording, 210 of the 256 changed weights were of the second
kind.

golem leaves the fp8 weights alone and adds s·B·(A·x) inside the same
product, which is what both stand for and what ComfyUI's bypass loader does
(`lora.go`, `shaders/krea2_mm.comp`). A LoRA is then a few hundred megabytes
beside the DiT, and with the DiT kept another strength costs nothing and
another LoRA the time to upload it, not the time to read twelve gigabytes.

With the front's LoRA a DiT call is 0.73% from the same call in float32 and
2.4% from ComfyUI's, what the call without one is. The portrait at seeds 43
and 44 is 23 to 25 dB from ComfyUI's with the LoRA and 24 without, the same
picture to the eye; ComfyUI's two loaders part by 26 and 31 dB there. At
seed 42 it is 21 dB with the LoRA, the same woman and dress with other lace,
and ComfyUI's two loaders 23.

A step at 768 × 1024:

| | a DiT step | putting it on |
|---|---:|---:|
| no LoRA | 1.02 s | |
| a LoRA of rank 32 (the front's) | 1.03 s | 0.3 to 1.2 s |
| a LoRA of rank 256 | 1.49 s, before the fp16 operands below | 2.5 to 5 s |
| another strength | same | 60 µs |

## Measured

RX 9070 XT, RADV, 768 × 1024, 8 steps, cfg 1, against ComfyUI on the same card
(ROCm 7.2, torch 2.11, its `start.sh` flags):

| | golem | ComfyUI |
|---|---:|---:|
| a DiT step | 1.02 s | 1.89 s |
| the eight steps | 8.2 s | 15 s |
| text encoder | 0.36 s | |
| VAE decode | 0.28 s | |
| a picture, DiT kept | 8.9 s | |
| reading the DiT, cold / cached | 14 s / 2.3 s | |

A step at that size is recorded as seven submissions of four blocks: the
driver allows a submission two seconds before it declares the card hung.

The products are what a step is made of, and what held them back was
reading their operand, not the matrix cores (183 TFLOPS on this card,
measured on registers alone). So the norms and the elementwise steps before
them write it in fp16, rounded and clamped as the product would round it,
and the product reads it as it is: half the bytes, and the same picture to
the bit. The large ones take tiles of 256 × 256, which read half the bytes
per product again; k and v, too small to fill the card with those, keep
tiles of 128. Back to back, as a step runs them, they run at 90 to 100 TFLOPS
(k and v at 60), and the attention at about 39.

The step's host work is out of the way too: the rope table, which only
depends on the sizes, is worked out once a picture, and the sampler's noise
is drawn while the card runs the step.
