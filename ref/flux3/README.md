# ref/flux3: FLUX 3 Image Klein as ComfyUI runs it

The reference is ComfyUI itself (worktree at `/mnt/data/dev/ComfyUI-flux3`,
origin/master commit `8f04fd6e`, sharing the main `.venv`), driven through its
own nodes in-process on the three files the t2i/i2i workflows load:

| file | under `models/` |
|---|---|
| `flux3-image-klein-bf16.safetensors` | `diffusion_models/` |
| `qwen3vl_4b_fp8_scaled.safetensors` | `text_encoders/` |
| `flux2-vae.safetensors` | `vae/` |

The text encoder file named by the upstream workflows is `qwen3vl_4b_bf16.safetensors`.
ComfyUI's `CLIPLoader` with `clip_type="flux3"` accepts the fp8-scaled checkpoint
`qwen3vl_4b_fp8_scaled.safetensors` without modification, tapping the 8 layers
[4, 8, 12, 16, 20, 24, 28, 32] for conditioning (conditioning width (8 taps x 2560)).

The reference workflows and sample input images are copied under `ref/flux3/workflows/`:
- `flux3_klein_bf16_t2i.json`
- `flux3_klein_bf16_i2i.json`
- `flux3_klein_bf16_mi2i.json`

## Workflow defaults

Read from `ref/flux3/workflows/flux3_klein_bf16_t2i.json`:
- Sampler: `euler`
- Scheduler: `simple`
- Steps: 16
- CFG: 1.0
- Size: 2048 x 2048 (the workflow default; node 128 width and height are wired from node 19 `ResolutionSelector` set to `["1:1 (Square)", 4.0, 16]`, computing 4.0 MP -> 2048 x 2048 via `comfy_extras/nodes_resolution.py`; `flux3_klein_bf16_i2i.json` and `_mi2i.json` node 145 also select 4.0 MP). golem defaults to 1024 x 1024 for the 16 GB card budget.
- Alpha (timestep shift): 16.0 (`shift = ln(16.0) = 2.77259`, patched via `ModelSamplingFlux`, versus base model shift `ln(6.93) = 1.93586`)
- Prompt upsampler: node 127 runs an LLM upsampler (`qwen3.5_9b_flux3_prompt_upsampler_bf16.safetensors`), which is bypassed in `smoke.py` and is a non-goal for golem.

## Headless execution

To run the reference smoke generation on the RX 9070 XT:

```sh
ulimit -c 0; export HSA_ENABLE_COREDUMP=0
/mnt/data/dev/ComfyUI/.venv/bin/python ref/flux3/smoke.py
```

The script runs ComfyUI with:
- `--use-split-cross-attention`
- `--fp16-text-enc`
- `--fp16-vae`
- `--disable-mmap`
- `--reserve-vram 1.5`
- `--deterministic`

Card exclusivity: nothing else may hold the GPU while the script runs. Check
`rocm-smi --showmemuse` and `pgrep -af 'main.py|golem-server|krea2'` first.
Outputs are saved to `testdata/flux3/smoke.png`.

## Fixture recording (`dump.py` and `dump_f32.py`)

To record reference activations and test fixtures:

```sh
ulimit -c 0; export HSA_ENABLE_COREDUMP=0
/mnt/data/dev/ComfyUI/.venv/bin/python ref/flux3/dump.py \
    /mnt/data/dev/ComfyUI-flux3 testdata/flux3 [stage ...]
```

The 4 MP stages are recorded apart, alone on the card and in a memory scope
(ComfyUI's i2i alone takes about 21 GB of host memory), from the main
ComfyUI directory whose `input/` holds the photos they read:

```sh
systemd-run --user --scope -q -p MemoryMax=24G -p MemorySwapMax=16G /mnt/data/dev/ComfyUI/.venv/bin/python ref/flux3/dump.py /mnt/data/dev/ComfyUI testdata/flux3 t2i4 i2i4 mi2i4
```

- `t2i4/`: the accept prompt at 2048 x 2048, seed 42.
- `i2i4/`: `input/IMG_20191030_202255.jpg` (4160 x 3120) saved again as
  `i2i4/ref0.jpg` with EXIF orientation 6, so it reads as a 3120 x 4160
  portrait; `ImageScaleToTotalPixels` (lanczos, 4.0 MP, steps 16) on the
  oriented photo gives the size, and the scaled photo is the one reference
  (`ReferenceMultiLatent` resize, 1.0 MP). Seed 7.
- `mi2i4/`: the same `ref0.jpg` matched, then `ref1.jpg` (toast), `ref2.png`
  and `ref3.png` (copied from `input/`), four references at 1.0 MP. Seed 7.

Each records `request.json` (with `ref_shapes`, the references' latent
`[h, w]`), `noise0`, `sigmas`, the first DiT call (`in.x`, `in.t`, `in.ctx`,
`in.next`, `out`; no block waypoints), `latent`, `rgb`, the PNG, and
`timings.json`: seconds per step (the first includes loading the DiT),
sampling, decode, total, and `decode_mode`, `full` or `tiled` (whether
ComfyUI fell back to `decode_tiled_`). With references: `ref_rgb0` (the
first reference as encoded), `ref_latents`, `ref_ids`.

Stages:
- `tokens/`: three prompts: `p0` (short prompt, 10 unpadded tokens, pads to 80),
  `p1` (exactly 80 unpadded tokens, 0 pads), `p2` (130 unpadded tokens, pads to 160).
  Records `tokens/p{0,1,2}.ids` (int32), `encoder/p{i}.hidden_l4` (the state entering
  layer 4 of Qwen3-VL, shape T x 2560), and `encoder/p{i}.cond` (shape T x 20480).
- `sample/`: 256 x 256 t2i, seed 7, steps 16, euler/simple, alpha 16.0: records
  `sigmas`, `noise0`, `x_{step}`, `denoised_{step}`, `latent`, and `rgb`.
- `sample_cfg/`: 256 x 256 run with CFG 2.0 and empty negative: records conditionings
  `cond` (length 160) and `uncond` (length 80), showing that uncond retains its own length.
- `dit/`: first DiT step of `sample/`: records `in.x`, `in.t`, `in.ctx`, `in.next`
  (flow map target sigma), `pe` (4-axis RoPE tensor), `wp.emb_img`, `wp.txt_early`,
  `wp.img_early`, `wp.single_{0,13,27}`, and `out`.
- `vae/`: decode of `sample/latent` with waypoints per up block (`up_0`..`up_3`);
  encode of `ref/flux3/workflows/klein_ref_toast.jpg` via `ReferenceMultiLatent` (`enc.in`, `enc.latent`).
- `i2i/`: i2i workflow at 256 x 256, seed 7, with `klein_ref_toast.jpg`: records
  `ref_rgb` (resized reference), `ref_latent`, `ref_states` (early blocks output), `dit.out0`, and `rgb`.
- `mi2i/`: multi-reference with `klein_ref_toast.jpg` and `klein_ref_great_wave.jpg`, prompt
  "a breakfast table with toast in front of the great wave": records `ref_rgb0`, `ref_rgb1`,
  `ref_latents` (shape `[1, 494, 128]`), `ref_ids` (ids at l = -1, -2, shape `[494, 4]`), and `rgb`.
- `accept/`: t2i at 1024 x 1024, seed 42, 16 steps: `latent`, `rgb`, and PNG.

### Reference image preprocessing

Following the upstream i2i and mi2i workflows, reference images are preprocessed through
ComfyUI's own `ReferenceMultiLatent` node (`comfy_extras/nodes_edit_model.py:65-74`), not a PIL resize.
The node is called with:
- `resolution`: `{"resolution": "resize", "megapixels": 0.0625}` (1/16 MP)
- Aspect ratio is preserved: `scale = math.sqrt(megapixels * 1024 * 1024 / (w * h))`
- Scaled width `round(w * scale)`, height `round(h * scale)`
- Resized via `comfy.utils.common_upscale(..., "lanczos", "disabled")` before VAE encoding

For the test fixtures:
- `toast` (1776 x 1184): scale = 0.1765, resized to 314 x 209 (preserving 3:2 aspect ratio),
  encoded latent shape `[1, 128, 13, 19]`. Saved as `vae/enc.in` and `i2i/ref_rgb` (`mi2i/ref_rgb0`).
- `great_wave` (1760 x 1184): scale = 0.1773, resized to 312 x 210,
  encoded latent shape `[1, 128, 13, 19]`. Saved as `mi2i/ref_rgb1`.

### Float32 reference DiT (`dump_f32.py`)

To record the same DiT call in float32 arithmetic on CPU and measure the gap to bf16:

```sh
/mnt/data/dev/ComfyUI/.venv/bin/python ref/flux3/dump_f32.py \
    /mnt/data/dev/ComfyUI testdata/flux3 [dit]
```

Writes waypoints under `dit32/` and logs relative RMS difference against `dit/`.
