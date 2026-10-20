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
	Width: 1024, Height: 1024, Steps: 16, CFG: 1, Seed: 42,
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

VRAM footprint:
- FP16: weights 10.34 GB, arena 1.84 GB, total 12.17 GB
- FP8: weights 5.17 GB, arena 1.84 GB, total 7.00 GB

## Reference Images

`Request.Images` (`-image`, `images` in the server request) passes reference
images for image-to-image or multi-reference conditioning.

The output image dimensions follow the request's `Width` and `Height`.

Each reference image is resized using a Lanczos3 filter to match the requested
`MegaPixels` (1.0 MP by default, matching the `ReferenceMultiLatent` widget in
ComfyUI's workflow), capped so the token budget does not exceed `MaxRefTokens`
(4096 tokens total).

The resized images are encoded by the VAE into 128-channel latents. Reference RoPE
position IDs use `l = -1, -2, ...` across consecutive reference inputs, matching
ComfyUI's `ref_ids`.
