# flux3

FLUX 3 Image Klein from a sentence to a picture, the way the ComfyUI mobile
front draws it: the same three files, the same sampler, and for the same prompt
and seed the same picture.

| file, under ComfyUI's `models/` | what it is |
|---|---|
| `diffusion_models/flux3-image-klein-bf16.safetensors` | the single-stream DiT (5 early blocks, 28 single blocks) |
| `text_encoders/qwen3vl_4b_fp8_scaled.safetensors` | Qwen3-VL-4B, fp8 with a scale a tensor, read at 8 layers |
| `vae/flux2-vae.safetensors` | the 128-channel 16x VAE (encoder and decoder) |

They are not in this repository. `ref/flux3/dump.py` records the fixtures
the tests compare against.

```go
p, err := flux3.Open(flux3.ComfyUI("/mnt/data/dev/ComfyUI")) // or flux3.Options{...}
img, times, err := p.Generate(flux3.Request{
	Prompt: "a red fox in the snow, photo",
	Width: 2048, Height: 2048, Steps: 16, CFG: 1, Seed: 42,
}, nil)
```

`cmd/flux3` draws one from the command line, and `golem-server -flux3 DIR`
serves `/v1/images/generations`.

`Options.Precision` selects weight storage precision in the DiT:
`PrecisionFP16` (default) keeps weights in half precision, achieving 34.12 dB
PSNR against ComfyUI on the 1024x1024 acceptance portrait. `PrecisionFP8` packs
weights into 8-bit floats with a per-tensor scale (5.17 GB weights instead of
10.34 GB), but costs 10.43 dB in fidelity (23.69 dB on the same portrait).
Because the difference exceeds 3 dB, FP16 is the default.

A pipeline draws up to `DefaultMaxPixels` (4456448, the workflows' 4 MP
sizes with room for 2:3, 1680 x 2512), each side at most 8192, with up to
four references (`Options.MaxPixels`, `Options.MaxRefs`). The DiT's arena is
sized from those limits and runs each block's MLP 4096 tokens at a time, so
2048 x 2048 with four references takes 3.66 GB of arena, under the 4 GiB one
storage buffer holds. Past 1 MP the VAE decodes in bands: the whole latent up
to level 2 (its attention sees the whole picture), then full-width bands of
the rest, run twice so every GroupNorm uses the whole picture's statistics,
merged from the bands' first run. That decode is 53.6 dB from a whole one
(ComfyUI's own on this card is its tiled fallback, 31.1 dB from it).

`Options.Keep` leaves the DiT on the card between pictures. The text encoder
stays in system memory and the card reads it across the bus, 325 ms a prompt.

## Performance

Measured on one RX 9070 XT (16 GB VRAM), every figure from a run in this
repository, compared to ComfyUI (worktree at origin/master with
`--use-split-cross-attention`) on the same card for the same picture:

| component | ComfyUI | golem FP16 (default) | golem FP8 |
|---|---:|---:|---:|
| text encoder (warm, positive prompt) | 410 ms | 325 ms | 325 ms |
| text encoder (cold, includes CLIP upload to VRAM) | 2504 ms | 335 ms | 335 ms |
| DiT load (host read + upload; warm page cache) | 15.31 s (host read 13.03 s + upload; 9.54 s seen on earlier run) | 3.37 s | 6.07 s |
| VAE decode 1024 x 1024 (warm min) | 429 ms | 581 ms | 581 ms |
| VAE decode 768 x 1024 (warm min) | 268 ms | 435 ms | 435 ms |
| DiT step 1024 x 1024 | 0.902 s | 0.756 s | 0.584 s |
| DiT step 768 x 1024 | 0.644 s | 0.561 s | 0.419 s |

Total picture generation time (prompt to decoded image, DiT kept):

| size | steps | ComfyUI (prompt changed) | ComfyUI (prompt cached, encode skipped) | golem FP16 (default) | golem FP8 |
|---|---|---:|---:|---:|---:|
| 1024 x 1024 | 16 | 19.42 s | 16.90 s | 13.25 s | 10.41 s |
| 1024 x 1024 | 8 | 11.18 s | 9.18 s | 7.10 s | 5.68 s |
| 768 x 1024 | 16 | 15.44 s | 11.96 s | 9.86 s | 7.57 s |
| 768 x 1024 | 8 | 8.94 s | 6.39 s | 5.33 s | 4.18 s |

ComfyUI's own log on this 16 GB card shows the FLUX 3 DiT loaded partially (3869 MB resident, 6714 MB offloaded) and part of it evicted when the VAE loads, so its step times include that streaming; golem holds the whole DiT and the arena on the card (12.17 GB). golem keeps its encoder in host memory and never swaps.

Without Keep (loading the DiT on every picture):

| size | steps | golem FP16 (default) | golem FP8 |
|---|---|---:|---:|
| 1024 x 1024 | 16 | 16.71 s | 16.62 s |
| 768 x 1024 | 16 | 13.40 s | 13.79 s |

### At 4 MP

The workflows' settings: 16 steps, references at 1 MP each, the size matched
to the first image for i2i and mi2i. ComfyUI recorded with
`--use-pytorch-cross-attention` (its split attention asks for 19 GiB at 4 MP
with a reference) and decoding tiled, its out-of-memory fallback; golem FP16,
DiT kept:

| picture | ComfyUI step (median) | golem step | ComfyUI decode | golem decode | ComfyUI total | golem total |
|---|---:|---:|---:|---:|---:|---:|
| t2i 2048 x 2048 | 3.69 s | 4.63 s | 10.97 s | 4.42 s | 74.5 s | 82.2 s |
| i2i 1776 x 2368, 1 ref | 5.15 s | 6.26 s | 9.61 s | 4.35 s | 98.5 s | 109.5 s |
| mi2i 1776 x 2368, 4 refs | 10.84 s | 13.07 s | 9.74 s | 4.35 s | 190.7 s | 219.5 s |

At 1 MP golem's step is the faster one; at 4 MP ComfyUI's is, by about a
fifth: attention grows with the square of the tokens (16.5k to 32.7k here)
and PyTorch's attention kernel on this card outruns golem's. The decode is
2.5 times ComfyUI's tiled one. Against ComfyUI's latents decoded whole, the
pictures are 24.96 dB (t2i, the same picture with details apart), 35.29 dB
(i2i) and 37.52 dB (mi2i).

VRAM footprint:
- FP16: weights 10.34 GB; arena 1.35 GB for a pipeline opened for 1 MP and
  one reference, 3.58 GB for the default 4.25 MP and four; 14.3 GB used on
  the card during a 4 MP step with four references
- FP8: weights 5.17 GB, the same arenas
- VAE: arena 3.08 GB at 4.25 MP, freed while the DiT samples

## Reference Images

`Request.Images` (`-image`, `images` in the server request) passes reference
images for image-to-image or multi-reference conditioning.

The output image dimensions follow the request's `Width` and `Height`, or,
with `MatchImage` (`-match-image`, `match_image` in the server request), the
first image scaled to 4 MP with sides a multiple of 16, ComfyUI's
`ImageScaleToTotalPixels` (lanczos, 4.0, 16) under the workflows'
`match_input_image`; that scaled image is then the first reference. The size
comes from the image as its EXIF orientation turns it.

Up to four references, each resized with a Lanczos3 filter to
`MegaPixels` (1.0 MP by default, the `ReferenceMultiLatent` widget in the
workflows), each at most 4096 tokens of its own, as ComfyUI does.

The resized images are encoded by the VAE into 128-channel latents. Reference RoPE
position IDs use `l = -1, -2, ...` across consecutive reference inputs, matching
ComfyUI's `ref_ids`.
