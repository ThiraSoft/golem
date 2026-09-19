# ref/krea2: Krea 2 as ComfyUI runs it

The reference is ComfyUI itself, driven by `dump.py` through its own nodes, on
the three files the mobile front loads (`custom_nodes/comfyui-mobile/workflow.py`,
`build_prompt`, LoRA off):

| file | under `models/` |
|---|---|
| `krea2_turbo_fp8.safetensors` | `unet/` |
| `qwen3vl_4b_fp8_scaled.safetensors` | `text_encoders/` |
| `qwen_image_vae.safetensors` | `vae/` |

They are not in this repository. `GOLEM_KREA2_COMFY` points the Go side at
another ComfyUI than `/mnt/data/dev/ComfyUI`.

## Fixtures

ComfyUI checkout `2a7ba9e7`, torch 2.11 on ROCm 7.2, an RX 9070 XT, and the
flags its `start.sh` runs with (fp16 text encoder, fp16 VAE, split attention):

```sh
ulimit -c 0; export HSA_ENABLE_COREDUMP=0
/mnt/data/dev/ComfyUI/.venv/bin/python ref/krea2/dump.py \
    /mnt/data/dev/ComfyUI testdata/krea2 [tokens] [sample] [accept]
```

Nothing else may hold the card while it runs: ComfyUI loads the DiT partly
and a second process on the card is how it ends in a memory access fault. The
fault writes a GPU core dump the size of the card, which is what the first
line turns off.

- `tokens/`, `encoder/`: three prompts, their tokens and weights, the text
  model's hidden states after a few layers and the conditioning;
- `sample/`: 256 × 256, 8 steps, seed 7, the sigmas, the starting noise, what
  the sampler went through at every step, the noise ER-SDE added, the latent
  and the picture;
- `dit/`: the first DiT call of that run with its waypoints;
- `vae/`: the decode of that run with a waypoint per block;
- `accept/`: the front's defaults, 768 × 1024, seed 42, and the PNG.

The noise is torch's on this machine: the CPU generator for the start, the
ROCm one for each step. `krea2/rng.go` says why that is enough to know.
