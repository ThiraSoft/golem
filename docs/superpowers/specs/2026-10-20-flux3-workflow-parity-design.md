# FLUX 3 Klein workflow parity: design

Turns `docs/superpowers/plans/2026-10-20-flux3-workflow-parity.md` (the rough
plan) into decisions. FLUX 3 only: nothing under `krea2/` and no Krea 2 path
in golem-server or krea2-mobile changes behaviour.

## Goal

golem and krea2-mobile draw FLUX 3 Klein pictures with the settings of the
BFL workflows `ref/flux3/workflows/flux3_klein_bf16_{t2i,i2i,mi2i}.json`:

| Setting | Workflow (target) |
|---|---|
| t2i size | `ResolutionSelector` 4.0 MP, multiple 16 (1:1 = 2048×2048) |
| i2i/mi2i size | `match_input_image`: image 1 through `ImageScaleToTotalPixels` (lanczos, 4.0 MP, steps 16); output = that size |
| references | `ReferenceMultiLatent` resize, 1.0 MP each, no split, up to 4 |
| reference 1 under match | the 4 MP rescaled image, then resized to 1 MP like any reference |

Shift ln 16, CFG 1, euler/simple, 16 steps already match.

## Numbers that bind

- 4 MP sizes (ComfyUI `round(r*sqrt(4 MiB/(a*b))/16)*16`): 1:1 2048×2048,
  2:3 1680×2512, 3:4 1776×2368, 9:16 1536×2736, 3:2 2512×1680, 4:3 2368×1776,
  16:9 2736×1536, 21:9 3136×1344.
- `DefaultMaxPixels = 4456448` (4.25 MiB pixels; the 2:3 size is 16485 tokens).
- Each side at most `MaxSide = 8192` pixels.
- `RefTokens = 4096` (one 1 MP reference), `MaxRefs = 4`.
- Text stays `MaxTextTokens = 960`.
- One Vulkan storage buffer addresses less than 4 GiB: every arena is
  checked against `1<<32` bytes with an error naming the limit.
- Card: RX 9070 XT, 17095983104 bytes of VRAM (`mem_info_vram_total`),
  nothing else resident when golem runs.

## 1. Fixtures at 4 MP (ComfyUI is the reference)

`ref/flux3/dump.py` gains three stages, recorded with ComfyUI's own nodes:

- `t2i4`: `PROMPT_ACCEPT`, 2048×2048, seed 42, 16 steps.
- `i2i4`: the phone photo `ComfyUI/input/IMG_20191030_202255.jpg`
  (4160×3120, EXIF 1) re-saved with EXIF Orientation 6 as `i2i4/ref0.jpg`, so
  its oriented form is a 3120×4160 portrait. Loaded with
  `ImageOps.exif_transpose`, `ImageScaleToTotalPixels(lanczos, 4.0, 16)` gives
  the output size, the scaled image is the only reference
  (`ReferenceMultiLatent` resize 1.0). Seed 7.
- `mi2i4`: the same `ref0.jpg` first (match_input_image), then
  `klein_ref_toast.jpg`, `ComfyUI/input/eyes_renarde.png`,
  `ComfyUI/input/Capture d'écran_20260803_084025.png`, copied as
  `ref1.jpg`, `ref2.png`, `ref3.png`. Four references at 1.0 MP. Seed 7.

Each stage writes: `request.json` (prompt, seed, steps, width, height, ref
shapes), `noise0`, `sigmas`, the first DiT call (`in.x`, `in.t`, `in.ctx`,
`in.next`, `out`, no block waypoints), for i2i4/mi2i4 `ref_rgb0` (reference
1 as encoded) and `ref_latents` and `ref_ids`, the final `latent`, the
decoded `rgb`, the PNG, and `timings.json`: per-step seconds, decode
seconds, total, and `decode_mode` (`"full"` or `"tiled"`, from wrapping
`VAE.decode_tiled_`). `bench_comfy.py` gets a 2048×2048 row.

The dump runs alone, by the controller, in
`systemd-run --user --scope -q -p MemoryMax=24G -p MemorySwapMax=16G`.
Implementers never load a model in Python.

## 2. DiT sized from options, MLP in chunks

`OpenDiTSized(d, path, prec, maxImageTokens, maxRefTokens)` sizes the arena
for `N = maxImageTokens + maxRefTokens + MaxTextTokens`; `OpenDiT(d, path,
prec...)` stays as the 1 MP call (4096, 4096) so every existing test runs
unchanged.

Arena per token after the change: `x, q, k, v, att, out` fp32 (6×3072), `nh`
as halves (1536 floats), rope table 128: 20096 floats. The MLP runs over the
tokens in chunks of `mlpChunk = 4096`: `mlpIn` into a `mlpChunk×2×9216` fp32
buffer, SwiGLU into a `mlpChunk×9216` halves buffer, `mlpOut` added into
`out`. The order inside a block becomes norm → q,k,v → qk-norm → rope →
attention → attnOut (store) → MLP chunks (add) → gate. `embImgBuf` goes:
`emb_in.image` writes straight into `x`. Halves regions (`nh`, `half`) are
placed first in the arena so no halves index passes 2^31.

At 17408 + 16384 + 960 tokens the arena is about 3.96 GB (under 4 GiB);
`OpenDiTSized` refuses a size past `1<<32` bytes naming both token limits.

Precision rule (the user's): measure peak VRAM of a 2048×2048 step with 4
refs, fp16, DiT kept (sysfs `mem_info_vram_used`, after everything else
golem holds at that point is resident). If it does not fit in 16 GB with
256 MiB to spare, the default precision of a pipeline opened past 1 MP is
`PrecisionFP8`; otherwise fp16 stays. The result and the numbers go in the
ledger and flux3/README.md.

## 3. VAE decode in bands

Full decode needs 655 KB a latent pixel (10.7 GB at 2048²) and a 256-channel
full-resolution tensor is 4 GB alone, so 4 MP decodes in bands:

1. **Trunk, exact, whole latent**: unpatch, post_quant_conv, conv_in, mid
   (with its global attention at /8), up level 3, upsample, up level 2's
   three resnets at /4. Its output (512 ch at 4h×4w) is kept in a `trunk`
   region.
2. **Bands**: full-width horizontal bands of `core` latent rows plus
   `vaeBandOverlap = 2` latent rows each side (clipped at the image edges, so
   edge bands see the same zero padding as a full decode). Band height
   `floor(vaeBandLatentPix / w)` with `vaeBandLatentPix = 4096`, core = that
   minus 4. Each band copies its trunk rows (512 planes, one strided copy),
   runs level-2 upsample, level 1, level 1 upsample, level 0, norm_out,
   conv_out, and the core rows of its RGB go to the host image.
3. **GroupNorm statistics shared across bands, two passes** (13 norms in the
   band part): pass A runs every band with its own statistics and copies
   each norm's slice statistics (n, mean, m2) out; the host merges them per
   norm and group (Chan's parallel formula in float64) and writes one slice
   per group; pass B runs every band with `GNorm(Slices=1)` on the merged
   statistics, no `GStats`, and its output is kept.

Rules: decode of `h*w <= vaeBandLatentPix` latent pixels stays the current
full path (1 MP output unchanged). The VAE arena is laid out at open from
`maxLatentPix` (whole latent) and `vaeBandLatentPix`; buffers are the max of
what the trunk and a band need; total under 4 GiB or a clear error. Encode
is unchanged and only ever sees references at 1 MP. A `K2.CopyStrided`
records `count` copies of `n` floats with source and destination strides in
one `vkCmdCopyBuffer`, then a barrier.

Targets: banded vs full decode of the same 1024² latent (band size forced
small) ≥ 40 dB; banded decode of `t2i4/latent` vs ComfyUI's `t2i4/rgb` ≥
35 dB; decode time measured against ComfyUI's.

## 4. Pipeline

- `Options.MaxPixels` default `DefaultMaxPixels`; `Options.MaxRefs` default
  `MaxRefs`. DiT opened with `OpenDiTSized(MaxPixels/256, MaxRefs*RefTokens)`,
  VAE with `MaxPixels/256`.
- References: each `ResizeReference(img, mp, RefTokens)` with `mp` =
  `Request.MegaPixels` or 1.0; no budget split. More than `MaxRefs` images is
  an error naming the limit.
- `Request.MatchImage bool` (`json:"match_image,omitempty"`): with images,
  `ScaleToTotalPixels(images[0], 4.0, 16)` (ComfyUI's formula, Python
  `round` = `math.RoundToEven`, lanczos through `lanczosKernel` into an RGBA
  image like PIL's uint8 result) gives Width and Height, replacing the
  request's; the scaled image replaces `images[0]` before the 1 MP resize.
  `DecodeImage` already applies EXIF orientation, so size comes from the
  oriented image. Without images, MatchImage is an error.
- `check`: multiples of 16 from 64, each side ≤ `MaxSide`, `w*h ≤
  MaxPixels`; errors name the limit. A tiny reference blown up to 4 MP is
  allowed.
- Targets: end to end t2i4/i2i4/mi2i4 vs fixtures ≥ 30 dB; timings beside
  ComfyUI's from `timings.json`.

## 5. cmd/flux3 and golem-server

- cmd/flux3: `-match-image` flag, default `-width 2048 -height 2048`,
  `MaxPixels` from the default.
- golem-server: `match_image` in the generation JSON, carried to
  `flux3.Request.MatchImage`; FLUX 3 default size `2048x2048`; a size the
  pipeline would refuse (past MaxPixels or MaxSide) answers 400 with the
  pipeline's message before the job runs (`Pipeline.Check(Request) error`,
  exported). Krea 2 paths unchanged.
- flux3/README.md: 4 MP rows in the performance and VRAM tables, the
  reference section rewritten (1 MP each, match_image).

## 6. krea2-mobile (FLUX 3 only)

- With FLUX 3 picked, the preset row shows the eight 4 MP sizes above plus a
  fast `1024²`; the default is 2048×2048. Krea 2 gets its current six
  presets and 768×1024 back on switching.
- Compose and Edit get a switch « taille de l'image 1 », on by default
  (saved in localStorage), which hides width, height and presets and sends
  `match_image: true`. Off: the typed size is sent as today.
- server.go: `match_image` read from the page and passed in `picture`;
  FLUX 3 default size 2048×2048, clamp per side 8192; Krea 2 keeps 1024.
  Go tests cover both.
- Commit on master, restart `krea2-mobile.service` at the end (it rebuilds
  golem-server from `~/dev/golem`, so after the merge).

## Out of scope

Prompt upsampler, bounding boxes, int8 checkpoints, a faster attention
kernel, anything Krea 2.
