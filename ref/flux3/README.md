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
[4, 8, 12, 16, 20, 24, 28, 32] for conditioning (hidden size 20480).

The reference workflows and sample input images are copied under `ref/flux3/workflows/`:
- `flux3_klein_bf16_t2i.json`
- `flux3_klein_bf16_i2i.json`
- `flux3_klein_bf16_mi2i.json`

## Workflow defaults

Read from `ref/flux3/workflows/flux3_klein_bf16_t2i.json` (node 128 / subgraph):
- Sampler: `euler`
- Scheduler: `simple`
- Steps: 16
- CFG: 1.0
- Size: 1024 x 1024
- Alpha (timestep shift): 16.0 (`shift = ln(16.0) = 2.77259`, patched via `ModelSamplingFlux`, versus base model shift `ln(6.93) = 1.93586`)

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
