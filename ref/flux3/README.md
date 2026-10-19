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
  encode of `ref/flux3/workflows/klein_ref_toast.jpg` at 256 x 256 (`enc.in`, `enc.latent`).
- `i2i/`: i2i workflow at 256 x 256, seed 7, with `klein_ref_toast.jpg`: records
  `ref_latent`, `ref_states` (early blocks output), `dit.out0`, and `rgb`.
- `mi2i/`: multi-reference with `klein_ref_toast.jpg` (256 x 256) and `klein_ref_great_wave.jpg`
  (384 x 256): records `ref_latents`, `ref_ids` (ids at l = -1, -2), and `rgb`.
- `accept/`: t2i at 1024 x 1024, seed 42, 16 steps: `latent`, `rgb`, and PNG.

### Reference image preprocessing

Following the BFL i2i and mi2i workflows (`ReferenceMultiLatent` / `ImageScale` nodes),
reference images are resized with `lanczos` interpolation (`disabled` crop) before VAE encoding:
- `toast`: resized to 256 x 256 (latent shape `[1, 128, 16, 16]`)
- `great_wave`: resized to 384 x 256 (latent shape `[1, 128, 16, 24]`)

### Float32 reference DiT (`dump_f32.py`)

To record the same DiT call in float32 arithmetic on CPU and measure the gap to bf16:

```sh
/mnt/data/dev/ComfyUI/.venv/bin/python ref/flux3/dump_f32.py \
    /mnt/data/dev/ComfyUI testdata/flux3 [dit]
```

Writes waypoints under `dit32/` and logs relative RMS difference against `dit/`.
