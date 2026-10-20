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
   - `vae/b0_ct`: `[1024, 384]` block 0's transposed convolution, before its residual units
   - `vae/b0_ru0`, `vae/b0_ru1`, `vae/b0_ru2`: `[1024, 384]` block 0 after each residual unit (`b0_ru2` is `block_0`)
   - `vae_f32acc/...`: the same waypoints (`conv1`, `b0_ct`, `b0_ru0..2`, `block_0..5`, `audio`)
     from the same graph with every `MUL_MAT` set to `GGML_PREC_F32`. At its default
     precision ggml-vulkan accumulates fp16 products in fp16 on this card (the device
     has fp16 accumulators on its matrix cores), which leaves yue2.cpp's own `conv1`
     0.15% away from the exact sum of its fp16 operands; this recording is the one an
     engine summing in float32 is held to. The operands stay fp16 (`im2col` to F16).

2. Tiled VAE decode on a 600-frame window (frames 2000..2599 of `bench/q8-replay.vae`):
   - `vae_tiled/latent`: `[600, 64]` time-major float32 input
   - `vae_tiled/audio`: `[2, 1151936]` planar stereo float32 waveform decoded by `vae_ggml_decode_tiled` with core 512, halo 16 (1920 * 600 - 64 samples per channel)

## Fixture Recording (`dump_nar`)

The NAR flow matching dumper runs on a short replay (first 100 semantic tokens of Guerreros):

```sh
systemd-run --user --scope -q -p MemoryMax=14G -p MemorySwapMax=2G \
    /mnt/data/golem-testdata/yue2/build-ref/dump_nar \
    /mnt/data/LLMs_models/Serveurperso/YuE2-GGUF/YuE2-3B-Q8_0.gguf \
    ref/yue2/requests/nar-short.json \
    /mnt/data/golem-testdata/yue2
```

Outputs are written under `testdata/yue2/nar/` and `testdata/yue2/rng/` with shapes in `testdata/yue2/meta.json`.

### Recorded Stages

1. AR sequence prefill into KV cache (3107 tokens: prefix 3006, 100 codes + offset 151853, music end 151852):
   - `nar/ar_ids`: `[3107]` int32 token sequence.
   - `nar/kv_k_l{0,13,27}`, `nar/kv_v_l{0,13,27}`: float32 `[3107, 8, 128]` position-major key/value cache rows (position, KV head, head dimension).
   - `nar/kv_all_k.f16`, `nar/kv_all_v.f16`: raw little-endian IEEE 754 binary16 (2 bytes per element), shape `[28, 3107, 8, 128]` (layer, position, KV head, head dimension).
2. Flow matching ODE midpoint solver (32 steps on 100 frames):
   - `nar/noise`: `[100, 64]` initial standard normal noise (seed 9207606299741175932).
   - `nar/temb_t`: `[2048]` timestep embedding of the first velocity call.
   - `nar/hidden_after_input`: `[102, 2048]` row-major hidden state after input projection, timestep embedding, and position embedding.
   - `nar/layer0_sa_output`: `[102, 2048]` self-attention output of layer 0.
   - `nar/hidden_after_layer{0,7,14,21,27}`: `[102, 2048]` hidden states after the corresponding layers.
   - `nar/v_first_0`, `nar/v_second_0`: `[100, 64]` first and second (midpoint) velocity evaluations of step 0.
   - `nar/xt_{0,1,7,31}`: `[100, 64]` ODE state after step k.
   - `nar/latent`: `[100, 64]` final latents after 32 midpoint steps.
3. The first velocity evaluation again, with yue2.cpp's products in float32
   (`dump_nar ... exact`, a second run after the default one, which it reads):
   - `nar_exact/temb_t`, `nar_exact/hidden_after_input`, `nar_exact/layer0_sa_output`,
     `nar_exact/hidden_after_layer{0,7,14,21,27}`, `nar_exact/v_first_0`: the probes
     above, same shapes, same noise, same prefix (`kv_all_{k,v}.f16` are loaded into
     the cache rather than prefilled again).
   - At its default settings ggml-vulkan quantizes the activations of every product
     against Q8_0 weights to Q8_1 (an fp16 scale and an int8 a value, blocks of 32),
     because this card has integer dot products. That alone puts the default
     `nar/hidden_after_input` 0.34% away from the exact sum of its noise and its
     Q8_0 weights. The exact run starts with `GGML_VK_DISABLE_INTEGER_DOT_PRODUCT`
     and `GGML_VK_DISABLE_COOPMAT` set (the int8 matrix cores are the other route to
     Q8_1) and every `MUL_MAT` at `GGML_PREC_F32`. Its products then take fp16
     operands (the Q8_0 weights dequantized to fp16, the activations rounded to
     fp16) and sum in float32: a float64 sum over those operands lands 0.0000% from
     `nar_exact/hidden_after_input`, and 0.0133% with the noise left in float32.
     This is the recording an engine with fp16 operands and float32 sums is held to.

```sh
systemd-run --user --scope -q -p MemoryMax=14G -p MemorySwapMax=2G \
    /mnt/data/golem-testdata/yue2/build-ref/dump_nar \
    /mnt/data/LLMs_models/Serveurperso/YuE2-GGUF/YuE2-3B-Q8_0.gguf \
    ref/yue2/requests/nar-short.json \
    /mnt/data/golem-testdata/yue2 exact
```

4. Standalone CPU RNG validation:
   - `rng/normal_37`: `[37]` values from torch_cpu_randn(seed, 37).
   - `rng/normal_6400`: `[6400]` values from torch_cpu_randn(seed, 6400), matching flattened `nar/noise`.

## Fixture Recording (`dump_tokens`)

The token, prompt, Philox, and sampler dumper runs against the Q8_0 backbone and records
tokenizer token IDs, assembled prompt sequences, Philox test cases, and sampled token draws:

```sh
systemd-run --user --scope -q -p MemoryMax=14G -p MemorySwapMax=2G \
    /mnt/data/golem-testdata/yue2/build-ref/dump_tokens \
    /mnt/data/LLMs_models/Serveurperso/YuE2-GGUF/YuE2-3B-Q8_0.gguf \
    /mnt/data/golem-testdata/yue2/bench/guerreros-replay.json \
    /mnt/data/dev/musicbench/yue2/tests/bpe-text.txt \
    /mnt/data/golem-testdata/yue2
```

Outputs are written under `testdata/yue2/{tokens,prompt,philox,sample}/` with shapes in `testdata/yue2/meta.json`.

### Recorded Stages

1. BPE Tokenizer:
   - `tokens/guerreros_request`: `[569]` int32 token IDs of the full Guerreros request conditioning text.
   - `tokens/bpe_text`: `[254]` int32 token IDs of `tests/bpe-text.txt` covering contractions, decomposed/precomposed accents, and ABC notation.
   - `tokens/guerreros_lyrics_nfd`: `[490]` int32 token IDs of the Guerreros lyrics decomposed to NFD.
   - `tokens/guerreros_abc`: `[2433]` int32 token IDs of the replay ABC score.
2. Prompt Sequences:
   - `prompt/guerreros_full_with_abc`: `[3006]` int32 token sequence for full chain-of-thought with ABC.
   - `prompt/guerreros_melody_no_abc`: `[572]` int32 token sequence for melody mode stopping at `ABC_START`.
   - `prompt/guerreros_off`: `[563]` int32 token sequence for chain-of-thought off mode.
   - `prompt/negative_off`: `[12]` int32 token sequence for unconditional stream with cot off.
   - `prompt/negative_full_with_abc`: `[2457]` int32 token sequence for unconditional stream with full ABC score.
3. Philox PRNG:
   - `philox/counters`: `[8, 4]` int32 counters.
   - `philox/seeds`: `[8, 2]` int32 key seeds.
   - `philox/words`: `[8, 4]` int32 output words from `philox4x32_10`.
4. Sampler:
   - `sample/score_{0,1,2}_*`: score phase (3 steps) logits, history, candidates (IDs and scores), step index, and drawn token.
   - `sample/semantic_{0,1,2}_*`: semantic phase (3 steps) logits, history, candidates, step index, and drawn token.
   - `sample/penalty_window_*`: repetition penalty with synthetic history longer than the window.
   - `sample/with_end_token_*`: score sampling at step >= min_tokens where `ABC_END` is eligible.

## Fixture Recording (`dump_ar`)

The AR half on the Guerreros request (`bench/guerreros-replay.json`), Vulkan0,
a cache of 4096 positions and two sets:

```sh
systemd-run --user --scope -q -p MemoryMax=14G -p MemorySwapMax=2G \
    /mnt/data/golem-testdata/yue2/build-ref/dump_ar \
    /mnt/data/LLMs_models/Serveurperso/YuE2-GGUF/YuE2-3B-Q8_0.gguf \
    /mnt/data/golem-testdata/yue2/bench/guerreros-replay.json \
    /mnt/data/golem-testdata/yue2
```

### Recorded Stages

1. Prefill of the full prompt with the replay's score (`qw3lm_forward`'s graph
   copied into the dumper, outputs set on the probes):
   - `ar/prefix`: `[3006]` int32, the prompt (equal to `prompt/guerreros_full_with_abc`).
   - `ar/prefill_hidden_l{0,13,27}`: `[4, 2048]` the hidden state after those layers, last four positions.
   - `ar/prefill_normed`: `[4, 2048]` after the final norm, last four positions.
   - `ar/prefill_logits_abc`: `[151849]` the last position's logits on the score rows [0, 151849).
   - `ar/prefill_logits_semantic`: `[32769]` on the code rows [151852, 184621).
2. Eight semantic decode steps (`qw3lm_forward_batch`), drawn as `yue2_generate`
   draws them (lm_seed 1497252302528095116, draw index = step):
   - `ar/decode_tokens`: `[8]` int32, the drawn tokens (the replay's first eight codes).
   - `ar/decode_logits`: `[8, 32769]` the logits after feeding each of them.
3. Cot off, guidance 1.01, four steps, the two branches in sets 0 and 1:
   - `ar/guided_prefix`: `[563]`, `ar/guided_negative`: `[12]` int32.
   - `ar/guided_cond_logits`, `ar/guided_uncond_logits`: `[4, 32769]` each step's logits before mixing.
   - `ar/guided_tokens`: `[4]` int32.
4. The prefill probes again with no product quantizing its activation
   (`dump_ar ... exact`: integer dot products and cooperative matrices off,
   every `MUL_MAT` of the probe graph at `GGML_PREC_F32`, as `dump_nar ... exact`):
   - `ar_exact/prefill_hidden_l{0,13,27}`, `ar_exact/prefill_normed`,
     `ar_exact/prefill_logits_{abc,semantic}`: same shapes.
   - Default against exact: 0.14% after layer 0, 0.38% after 13, 0.80% after 27,
     0.95% after the final norm, 1.45% on the code logits, 0.04% on the score
     logits. This is how far yue2.cpp moves under a change of arithmetic alone.
   - The head is a mat-vec on the float32 hidden state: a float64 sum of the
     Q8_0 rows over `ar/prefill_normed` lands 0.0000% from `ar/prefill_logits_*`,
     and 0.46% on the codes once the hidden state is quantized to Q8_0.

```sh
systemd-run --user --scope -q -p MemoryMax=14G -p MemorySwapMax=2G \
    /mnt/data/golem-testdata/yue2/build-ref/dump_ar \
    /mnt/data/LLMs_models/Serveurperso/YuE2-GGUF/YuE2-3B-Q8_0.gguf \
    /mnt/data/golem-testdata/yue2/bench/guerreros-replay.json \
    /mnt/data/golem-testdata/yue2 exact
```

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

## The acceptance bench (2026-10-20)

Both engines on the Guerreros request in one sitting, each run twice, the
second kept (page cache warm for both). Before it: unload the musicbench
worker (`curl -X POST localhost:8005/api/unload`), check that nothing else
holds the card (`rocm-smi --showmemuse`,
`pgrep -af "yue-server|golem|krea2|ComfyUI|python"`), and record the DPM
level (`/sys/class/drm/card1/device/power_dpm_force_performance_level`,
`auto`) and `powerprofilesctl get`. Each script prints the wall clock of
the whole process, models loaded included, and its stage lines.

```sh
Y=/mnt/data/dev/musicbench/yue2 B=/mnt/data/golem-testdata/yue2/bench O=$B/t10
scope() { systemd-run --user --scope -q -p MemoryMax=14G -p MemorySwapMax=2G "$@"; }
go build -o /tmp/yue2 ./cmd/yue2
# yue2.cpp: the free run (its tokens must equal bench/q8-full.tokens.csv) and the replay
scope ref/yue2/bench_yue2cpp.sh $Y Q8_0 $B/guerreros-full.json $O/cpp-full-2
scope ref/yue2/bench_yue2cpp.sh $Y Q8_0 $B/guerreros-replay.json $O/cpp-replay-2
# golem: teacher-forced (the acceptance run), replay, free run from the seeds
scope ref/yue2/bench_golem.sh /tmp/yue2 $B/guerreros-replay.json $O/golem-forced-2 -forced -compare $B/q8-full
scope ref/yue2/bench_golem.sh /tmp/yue2 $B/guerreros-replay.json $O/golem-replay-2 -compare $B/q8-replay
scope ref/yue2/bench_golem.sh /tmp/yue2 $B/guerreros-full.json $O/golem-full-2 -compare $B/q8-full
```

`-compare` prints where golem's score and codes first leave yue2.cpp's run,
and the wav16's SNR against its wav when the lengths agree. The results
are in `yue2/README.md` ("Speed", "Same seeds"). On 2026-10-20 yue2.cpp
reproduced `bench/q8-full.tokens.csv` and `bench/q8-full.abc` byte for byte
on both of its free runs, and its stage times were within 0.4 s of the
table above.

## Fixture Recording (`yue-synth` and `dump_audio`)

The whole pipeline is recorded with yue2.cpp's own CLI, no dumper of its own:
`ref/yue2/requests/pipeline-short.json` is `nar-short.json` (the first 100
codes of the Guerreros replay, its score, its seeds) with `output_format`
`wav32`, the decoder's float audio written as it is, without normalization.

```sh
Y=/mnt/data/dev/musicbench/yue2 M=/mnt/data/LLMs_models/Serveurperso/YuE2-GGUF O=/mnt/data/golem-testdata/yue2/pipeline
synth() { systemd-run --user --scope -q -p MemoryMax=14G -p MemorySwapMax=2G \
    $Y/build/yue-synth --device Vulkan0 --model $M/YuE2-3B-Q8_0.gguf --vae $M/YuE2-Vae-F32.gguf \
    --request ref/yue2/requests/pipeline-short.json "$@"; }
synth --out $O/short.wav --latent $O/short.vae --tokens $O/short.tokens.csv
synth --max-seq 3150 --out $O/chunks.wav --latent $O/chunks.vae
GGML_VK_DISABLE_INTEGER_DOT_PRODUCT=1 GGML_VK_DISABLE_COOPMAT=1 \
    synth --out $O/short_exact.wav --latent $O/short_exact.vae
```

`dump_audio` (no ggml: `audio-io.h` is header-only) turns each render into
fixtures, and with `normalize` records yue2.cpp's output stage on the short
render's audio:

```sh
D=/mnt/data/golem-testdata/yue2/build-ref/dump_audio F=/mnt/data/golem-testdata/yue2
$D $F short $O/short.wav $O/short.vae normalize
$D $F chunks $O/chunks.wav $O/chunks.vae
$D $F short_exact $O/short_exact.wav $O/short_exact.vae
```

### Recorded Stages

1. `pipeline/{short,chunks,short_exact}/audio`: `[2, 191936]` planar stereo float32,
   the decoder's output (1920 * 100 - 64 samples a channel), and
   `pipeline/{short,chunks,short_exact}/latent`: `[100, 64]` time-major latents.
   - `short` is yue-synth at its whole context: one chunk of 100 frames. Its
     latents are `nar/latent` byte for byte.
   - `chunks` is `--max-seq 3150`: the prefix of 3006 leaves chunks of
     (3150 - 3006 - 3) / 2 = 70 frames, so the song takes two, 70 and 30, the
     second prefilling its 30 codes and the end token behind the prefix.
   - `short_exact` is `short` with the integer dot products and the matrix cores
     off. At its defaults ggml-vulkan rounds every activation entering a Q8_0
     product to Q8_1 (see `nar_exact/` above); with both off, the products take
     fp16 operands and sum in float32, as golem's do. The two yue2.cpp renders
     are 39.33 dB apart on the latents and 28.53 dB on the audio: that is
     yue2.cpp's own arithmetic noise on this replay, and the reason the
     pipeline's 30 dB audio bar is held against `short_exact`.
2. `normalize/out_{0,10,999}`: `[2, 191936]`, `pipeline/short/audio` through
   `audio_normalize` at those `peak_clip` values.
3. `normalize/wav16`, `normalize/wav24`: the raw bytes of `audio_encode_wav` over
   `normalize/out_10` (the shape is the byte count); `normalize/wav32`: over the
   unnormalized audio, which is `pipeline/short.wav` byte for byte.
