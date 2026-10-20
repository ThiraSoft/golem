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

