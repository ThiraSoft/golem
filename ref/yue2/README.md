# ref/yue2: Reference Harness and Fixtures from yue2.cpp

The reference implementation is yue2.cpp at `/mnt/data/dev/musicbench/yue2`
(commit `8af0174`), built with Vulkan support in `/mnt/data/dev/musicbench/yue2/build/`.
The reference device is `Vulkan0` on the AMD Radeon RX 9070 XT.

## Build

The reference dumpers build against yue2.cpp headers and pre-built ggml shared
libraries:

```sh
cmake -B /mnt/data/golem-testdata/yue2/build-ref -S ref/yue2 -DYUE2_ROOT=/mnt/data/dev/musicbench/yue2
cmake --build /mnt/data/golem-testdata/yue2/build-ref
```

## Fixture Recording (`dump_vae`)

The VAE dumper runs under a systemd memory scope on an idle GPU:

```sh
systemd-run --user --scope -q -p MemoryMax=14G -p MemorySwapMax=2G \
    /mnt/data/golem-testdata/yue2/build-ref/dump_vae \
    /mnt/data/LLMs_models/Serveurperso/YuE2-GGUF/YuE2-Vae-F32.gguf \
    /mnt/data/golem-testdata/yue2/bench/q8-replay.vae \
    /mnt/data/golem-testdata/yue2
```

Outputs are written to `testdata/yue2/` with shapes indexed in `testdata/yue2/meta.json`.

### Recorded Stages

1. Untiled VAE decode on a short window (frames 1000..1063 of `bench/q8-replay.vae`):
   - `vae/latent`: `[64, 64]` time-major float32 input
   - `vae/conv1`: `[2048, 64]` channel-major float32 output of initial convolution
   - `vae/block_0`: `[1024, 384]` channel-major float32 output of decoder block 0
   - `vae/block_1`: `[512, 1919]` channel-major float32 output of decoder block 1
   - `vae/block_2`: `[256, 7676]` channel-major float32 output of decoder block 2
   - `vae/block_3`: `[128, 30704]` channel-major float32 output of decoder block 3
   - `vae/block_4`: `[64, 61408]` channel-major float32 output of decoder block 4
   - `vae/block_5`: `[64, 122816]` channel-major float32 output of decoder block 5
   - `vae/audio`: `[2, 122816]` planar stereo float32 waveform (1920 * 64 - 64 samples per channel)

2. Tiled VAE decode on a 600-frame window (frames 2000..2599 of `bench/q8-replay.vae`):
   - `vae_tiled/latent`: `[600, 64]` time-major float32 input
   - `vae_tiled/audio`: `[2, 1151936]` planar stereo float32 waveform decoded by `vae_ggml_decode_tiled` with core 512, halo 16 (1920 * 600 - 64 samples per channel)

## Reference measured 2026-10-20 (yue-synth, Vulkan0, DPM auto, card idle)

`ref/yue2/bench_yue2cpp.sh`, Guerreros request (style, lyrics, lm_seed,
seed, cot full), outputs under `/mnt/data/golem-testdata/yue2/bench/`:

| run | wall | AR score | AR codes | NAR 32 steps | VAE |
|---|---:|---:|---:|---:|---:|
| Q8_0 full | 108.5 s | 2433 tok, 13.4 s | 4870 tok, 25.9 s | 65.9 s (2.06 s/step) | 1.48 s |
| Q8_0 replay | 70.1 s | | | 65.9 s | 1.47 s |
| BF16 full | 155.6 s | 2883 tok, 23.2 s | 5415 tok, 41.6 s | 86.3 s (5415 frames) | 1.64 s |
| BF16 replay | 75.9 s | | | 70.5 s | 1.51 s |

Facts these runs establish:

- Q8_0 full regenerates the replay's 4870 codes exactly: yue2.cpp is
  deterministic against itself on one file.
- **BF16 and Q8_0 with the same seeds diverge at the first semantic token**
  (and write different scores). Same seeds give the same song only for the
  same arithmetic; golem's own logits will put it in the same position.
- **SNR floor of the reference against itself**: the Q8_0 full run and the
  Q8_0 replay of the same codes differ (prefill versus decode-written cache)
  by **34.8 dB** on the wav16; Q8_0 replay against BF16 replay is
  **25.8 dB**. These bracket what "the same audio" can mean between two
  engines; golem's replay SNR is reported against them.
