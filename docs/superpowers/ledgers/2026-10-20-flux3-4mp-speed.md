# FLUX 3 Klein at 4 MP: catching ComfyUI

RX 9070 XT, FP16 DiT kept on the card, 16 steps. Every figure measured in
this worktree; ComfyUI's from testdata/flux3/{t2i4,i2i4,mi2i4}/timings.json
(`--use-pytorch-cross-attention`, median step).

## Baseline (main at f7ea5ee)

| picture | step | total | PSNR vs ComfyUI's latent |
|---|---:|---:|---:|
| t2i4 2048x2048 | 4.629 s | 82.23 s | 24.96 dB |
| i2i4 1776x2368, 1 ref | 6.265 s | 109.48 s | 35.29 dB |
| mi2i4 1776x2368, 4 refs | 13.072 s | 219.58 s | 37.52 dB |

1 MP FP16 step (TestStepTimings): 750.4 ms. Krea 2 DiT step 768x1024
(krea2 TestFullSizeTimings, second step): 1.005 s.

## Profile (TestStepProfile4MP, old attention)

Submissions of a t2i4 step (16544 tokens): head and early blocks 692 ms, the
single blocks 139 ms (1 block) and 556 ms (4 blocks), tail 1 ms. One single
block with a stamp after every dispatch:

| family | t2i4, 16544 tokens | mi2i4, 32755 tokens |
|---|---:|---:|
| attention | 68.8 ms (50.0%), 49 TFLOPS | 281.3 ms (67.4%), 47 TFLOPS |
| MLP in (5 / 8 chunks) | 40.7 ms (29.6%), 46 TFLOPS | 80.5 ms (19.3%) |
| q, k, v, attention out | 11.8 ms (8.6%), 106 TFLOPS | 24.1 ms (5.8%) |
| MLP out | 9.1 ms (6.6%), 103 TFLOPS | 17.9 ms (4.3%) |
| SwiGLU | 2.7 ms | 5.3 ms |
| rope | 1.5 ms | 2.8 ms |
| QK norm | 1.4 ms | 2.7 ms |
| gate | 1.1 ms | 2.0 ms |
| LayerNorm | 0.5 ms | 1.0 ms |

Attention dominates, as supposed. The MLP's input product runs at less than
half the rate of the others: its fp16 weight (18432 x 3072, 113 MB) is the
only one past the card's 64 MB of L3, see below.

## Attention (vk/shaders/krea2_flash.comp)

TestK2AttnThroughput, random data (zeros, a new arena's, run at a higher
clock and flattered the first figures by a quarter), best of five after a
second of warm-up:

| shape | old krea2_attn128 | flash, keys through LDS | flash, fp16 copy |
|---|---:|---:|---:|
| FLUX 3 1 MP, 4176 tokens | 45.9 TFLOPS* | 82.3* | 88.0 |
| FLUX 3 4 MP, 16544 | 47.9* | 93.2* | 98.5 |
| FLUX 3 4 MP + 4 refs, 32755 | 43.9* | 91.6* | 96.7 |
| Krea 2 768x1024, 3098, 48 heads over 12 | 51.7* | 82.0* | 95.1 |

(* on zeros.) On a real t2i4 block: 68.8 ms -> 31.7 ms (kvhalf 1.4 ms of it).

Tried and left: 16 queries a wave (LDS-bound, 47 TFLOPS), 64-key tiles (spill
at 32 queries a wave), 8 or 2 waves a workgroup (same), queries pre-scaled
into log2 units (+3% but 1.39% -> 1.76% on the peaked test), exp2 in fp16
(nothing), a lazy maximum (rescale only past 2^8; nothing: the rescale is not
what costs).

## Products

TestK2Flux3MMThroughput (zeros):

| product | main | bands of 8 row tiles | + fp16 tiles of 256 |
|---|---:|---:|---:|
| MLP in 18432x3072x4096 | 47.9 TFLOPS | 112.0 | 116.4 |
| MLP out 3072x9216x4096 | 110.4 | 111.9 | 107.8 (so stays 128) |
| 3072x3072x16544 | 110.4 | 110.4 | 116.7 |

Krea 2 fp8 (TestK2Throughput): 66.5 / 62.5 / 64.2 / 49.4 -> 67.8 / 63.3 /
64.0 / 50.7 TFLOPS. mlpChunk 8192: no faster than 4096 (109.5 / 115.5).
q, k and v as one 9216-row product: 7.58 ms against 3 x 2.77, 0.7 ms a
block (1%), not done.

## Steps through the work (TestDiT4MPMatchesComfyUI / TestStepProfile4MP)

| change | t2i4 step | mi2i4 step | t2i4 block |
|---|---:|---:|---:|
| main | 4.63 s | 13.07 s | 137.5 ms |
| flash attention, fp16 copy | 3.39 s | 8.68 s | 101.7 ms |
| products in bands | 2.70 s | 7.28 s | 80.3 ms |
| fp16 tiles of 256 | 2.64 s | 7.13 s | 78.5 ms |
| gate in the products, no out buffer | 2.62 s | 7.09 s | 78.1 ms |
| SwiGLU in the MLP input's epilogue | 2.46 s | 6.81 s | |
| QK norm and RoPE in one pass | 2.42 s | 6.73 s | 72.2 ms |

Last profile, one t2i4 block: attention 31.7 ms, MLP in 15.7, q/k/v/out
13.0, MLP out 9.7, qkrope 1.5, LayerNorm 0.5. The rms against ComfyUI's
first 4 MP call stayed at 0.68% (t2i4) and 1.27-1.28% (mi2i4).

