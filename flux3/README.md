# flux3

FLUX 3 Image Klein from a sentence to a picture, the way the ComfyUI mobile
front draws it: the same three files, the same sampler, and for the same prompt
and seed the same picture.

| file, under ComfyUI's `models/` | what it is |
|---|---|
| `diffusion_models/flux3-image-klein-bf16.safetensors` | the single-stream DiT (5 early blocks, 28 single blocks) |
| `text_encoders/qwen3vl_4b_fp8_scaled.safetensors` | Qwen3-VL-4B, fp8 with a scale a tensor, read at 12 layers |
| `vae/flux2-vae.safetensors` | the 128-channel 16x VAE (encoder and decoder) |

They are not in this repository. `ref/flux3/dump.py` records the fixtures
the tests compare against.

```go
p, err := flux3.Open(flux3.ComfyUI("/mnt/data/dev/ComfyUI")) // or flux3.Options{...}
img, times, err := p.Generate(flux3.Request{
	Prompt: "a red fox in the snow, photo",
	Width: 1024, Height: 1024, Steps: 16, CFG: 1, Seed: 42,
}, nil)
```

`cmd/flux3` draws one from the command line, and `golem-server -flux3 DIR`
serves `/v1/images/generations`.

`Options.Precision` selects weight storage precision in the DiT:
`PrecisionFP8` (default) packs weights into 8-bit floats for lower VRAM, while
`PrecisionFP16` keeps weights in half precision.

`Options.Keep` leaves the DiT on the card between pictures. The text encoder
stays in system memory and the card reads it across the bus, under 300 ms a prompt.

## Reference Images

`Request.Images` (`-image`, `images` in the server request) passes reference
images for image-to-image or multi-reference conditioning.

Each reference image is resized using a Lanczos3 filter to match the requested
`MegaPixels` (4.0 by default, matching node 145 of the ComfyUI workflow), capped
so the token budget does not exceed `MaxRefTokens` (1024 tokens total).

The resized images are encoded by the VAE into 128-channel latents. Reference RoPE
position IDs use `l = -1, -2, ...` across consecutive reference inputs, matching
ComfyUI's `ref_ids`.

## Measured

RX 9070 XT, RADV, 16 steps, Euler / simple schedule, warm DiT:

| Resolution | Precision | Sample time | Total time | ms / step |
|---|---|---:|---:|---:|
| 1024 x 1024 | FP8 | 9.30 s | 10.18 s | 581.0 ms |
| 1024 x 1024 | FP16 | 12.00 s | 12.89 s | 750.2 ms |
| 768 x 1024 | FP8 | 6.66 s | 7.39 s | 416.1 ms |
| 768 x 1024 | FP16 | 8.78 s | 9.52 s | 548.8 ms |

At 768 x 1024, FLUX 3 Image Klein in FP8 draws a 16-step picture in 7.39 s,
faster than Krea 2's 8-step generation (8.9 s) on the same card.
