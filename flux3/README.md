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
| DiT load (host read + upload; warm page cache) | 15.31 s (host read 13.03 s + upload; 9.54 s seen on earlier run) | 3.72 s | 6.25 s |
| VAE decode 1024 x 1024 (warm min) | 429 ms | 581 ms | 581 ms |
| VAE decode 768 x 1024 (warm min) | 268 ms | 435 ms | 435 ms |
| DiT step 1024 x 1024 | 0.902 s | 0.431 s | 0.436 s |
| DiT step 768 x 1024 | 0.644 s | 0.324 s | 0.339 s |

Total picture generation time (prompt to decoded image, DiT kept):

| size | steps | ComfyUI (prompt changed) | ComfyUI (prompt cached, encode skipped) | golem FP16 (default) | golem FP8 |
|---|---|---:|---:|---:|---:|
| 1024 x 1024 | 16 | 19.42 s | 16.90 s | 7.96 s | 8.07 s |
| 1024 x 1024 | 8 | 11.18 s | 9.18 s | 4.41 s | 4.49 s |
| 768 x 1024 | 16 | 15.44 s | 11.96 s | 6.04 s | 6.27 s |
| 768 x 1024 | 8 | 8.94 s | 6.39 s | 3.37 s | 3.50 s |

ComfyUI's own log on this 16 GB card shows the FLUX 3 DiT loaded partially (3869 MB resident, 6714 MB offloaded) and part of it evicted when the VAE loads, so its step times include that streaming; golem holds the whole DiT and the arena on the card (12.17 GB). golem keeps its encoder in host memory and never swaps.

Without Keep (loading the DiT on every picture):

| size | steps | golem FP16 (default) | golem FP8 |
|---|---|---:|---:|
| 1024 x 1024 | 16 | 11.89 s | 14.54 s |
| 768 x 1024 | 16 | 10.01 s | 12.74 s |

### At 4 MP

The workflows' settings: 16 steps, references at 1 MP each, the size matched
to the first image for i2i and mi2i. ComfyUI recorded with
`--use-pytorch-cross-attention` (its split attention asks for 19 GiB at 4 MP
with a reference) and decoding tiled, its out-of-memory fallback; golem FP16,
DiT kept:

| picture | ComfyUI step (median) | golem step | ComfyUI decode | golem decode | ComfyUI total | golem total |
|---|---:|---:|---:|---:|---:|---:|
| t2i 2048 x 2048 | 3.69 s | 2.46 s | 10.97 s | 4.42 s | 74.5 s | 47.9 s |
| i2i 1776 x 2368, 1 ref | 5.15 s | 3.32 s | 9.61 s | 4.34 s | 98.5 s | 62.8 s |
| mi2i 1776 x 2368, 4 refs | 10.84 s | 6.81 s | 9.74 s | 4.34 s | 190.7 s | 120.1 s |

golem's step is a third under ComfyUI's at every size. It was a fifth over
(4.63, 6.26 and 13.07 s, totals 82.2, 109.5 and 219.5 s) until:

- the attention (vk/shaders/krea2_flash.comp) works on the transposes, so
  that the softmax happens in the registers the matrix cores leave the
  scores in, 32 queries a wave, from an fp16 copy of the keys and values
  (the values transposed) made beside the arena: 68.8 to 31.7 ms a block
  at 16.5k tokens, 281 to 137 ms at 32.7k;
- the products go over every column a band of eight row tiles at a time,
  so that the MLP input's weight, 113 MB in fp16 and past the card's 64 MB
  of cache, is read from memory once rather than once a column tile (48 to
  112 TFLOPS), and the large fp16 ones take tiles of 256;
- SwiGLU happens in the MLP input product's epilogue, its weight's rows
  stored in runs of 32 of the gate's and the up's; the attention and MLP
  outputs add into the stream through the products' gate; the queries' and
  keys' norm and rotary position are one pass.

A 2048 x 2048 block is now 72 ms: attention 32, the MLP's two products 25,
q, k, v and the attention's output 13, the rest 2. The 1 MP step went from
0.756 to 0.431 s with it. Loading the DiT costs 0.17 s more (3.74 s, warm),
to store the MLP input in runs. Against ComfyUI's latents decoded whole, the
pictures are 24.97 dB (t2i, the same picture with details apart), 35.27 dB
(i2i) and 37.54 dB (mi2i), where they were 24.96, 35.29 and 37.52 dB; the
1 MP acceptance portrait is 34.13 dB.

VRAM footprint:
- FP16: weights 10.34 GB; arena 1.04 GB for a pipeline opened for 1 MP and
  one reference, 3.11 GB for the default 4.25 MP and four, plus the
  attention's fp16 keys and values beside it (0.11 and 0.43 GB); 14.35 GB
  used on the card during a 4 MP step with four references
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
