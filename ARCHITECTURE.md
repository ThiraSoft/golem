# Golem Architecture & Performance Details

This document covers the technical details of Golem's implementation, how it achieves its performance, and how it is tested.

## The Method (Testing & Verification)

**No layer is deemed correct until its intermediate activations match the reference implementation.**

Scripts load the real weights, inject a deterministic input, and write every intermediate quantity into `testdata/`. The Go tests read those files back, so they need neither Python nor llama.cpp at test time — `ref/` says what wrote each fixture and how to write it again.

For `gemma/` the reference is not PyTorch but llama.cpp itself, instrumented: the weights on disk are quantized, and a bf16 reference would bury a mistake under its own quantization error. `ref/gemma/dump_layers.cpp` records every intermediate under ggml's own names, and the engine is checked against those recordings — including the parts of ggml that are not the arithmetic anyone would write, its tabulated GELU and its fp16 caches.

The same rule holds for speed. Every number in these READMEs is a benchmark in the repository, run on the machine named beside it; nothing is estimated.

## The Shared Layer

| package | holds |
|---|---|
| `tensors/` | safetensors and GGUF: metadata, and views on the bytes |
| `nn/` | quantized matrix products with AVX2 kernels, norms, activations, the three rotations three architectures turn by, convolutions, ggml's antialiased resize, and the worker pool they are spread over |
| `token/` | tokenizers, one package per family |
| `audio/` | sound: reading and writing WAV, decoding MP3 and FLAC, resampling to mono at the model's rate, and the log-mel front end a speech tower reads |
| `imageio/` | decoding an image and putting it in the shape a vision encoder reads: planar RGB, one float a channel a pixel |
| `sample/` | top-k, top-p, temperature, and a seeded draw over a row of logits |
| `chat/` | a conversation's shape — messages, tools, calls — and the interface an engine implements to write one out |
| `vk/` | Vulkan compute, bound through `purego` rather than cgo: devices, buffers, pipelines, and the three dozen kernels the blocks of three architectures are made of — with a vision tower of its own, whose fp16 weights and mean-subtracting norms share nothing with the text kernels. |

Nothing is promoted into this layer on the strength of a guess. Code moves here once two engines are shown to want it, in the same commit that makes them both use it. `chat/` is the newest of them: the conversation types lived in `gemma/` until a second engine needed them.

`engine/` is the one package that sits above the engines rather than under them. It reads `general.architecture` out of a GGUF, opens whichever engine implements it, and hands back one shape — the forward pass, the vocabulary, the chat template, and the numbers a startup line prints. It exists so that a command names no engine, and nothing but a command imports it.

## Hardware & Tuning Limitations

Worth knowing before diving into the code:

- **x86-64 with AVX2 is the only tuned target.** `nn/*.s` is where the speed comes from. arm64 has NEON kernels for the quantized generation path, written and tested under QEMU on an x86 machine and never once timed on real ARM hardware — correct, and tuned by nobody. The vision tower's interleaved kernel and the audio decoder's are portable Go there. An Apple or a Graviton runs; it will not see the numbers above.

## GPU / Vulkan Architecture

**Most of a token runs on a GPU, if you ask.** This began as a CPU engine and the CPU path is still the one every test is written against. But a token of the 26B A4B reads about 2.3 gigabytes and almost all of it is four kinds of matrix: the logit head, 0.6 gigabytes, which is the input embedding read the other way round; the expert stacks, 0.8, eight matrices at a time out of a hundred and twenty-eight; the shared branch beside them, 0.3; and the attention's four projections, 0.5. None of it shortens with CPU work, because the bytes are the cost — the kernels already run at 37 GB/s on a bus whose ceiling is about 43.

`-vulkan` moves all four, and then everything between them. On the 26B A4B, **13.4 tokens a second becomes 133.5**, against llama.cpp's Vulkan build at 124.8 on the same card, and **the prompt goes from 40 a second to 4541** — against their 4038. It costs those matrices being resident — 12.8 gibibytes, which is why a card with sixteen is the smallest that can do this — and about nine seconds of upload.

A dense checkpoint goes the same way, because a dense block is a mixture block with one branch: the shared branch of a mixture and an ordinary feed forward are the same three matrices under the same norm, and what differs is the end of the block — one post-norm instead of three, and no routing. On the 12B, **5.0 tokens a second becomes 65.4**, against llama.cpp's 64.6 on the same card.

Qwen3 runs on the same stack. Four things differ and they are all the file being read rather than a second path: an ordinary pre-norm block, where neither half is normed on its way back into the stream; a SiLU on the gate where Gemma looks ggml's GELU up in a table; scores scaled by one over the square root of the head, which Gemma leaves at one because its query norm holds them in range; and a value handed to the attention unnormed, which Gemma norms. On the 4B, **14.6 tokens a second becomes 167.0**, against llama.cpp's 169.7; on the 0.6B, 83.4 becomes 359.4 against 365.4. The head is Q4_0 on those checkpoints rather than Q6_K, and reads through `shaders/matvec.comp` — the kernel the attention's projections already use.

The whole of a block goes: the norms, the rotation, the keys and values in fp16, the scores, the softmax and the mix, the router, the experts and the three post-norms that make a mixture block. Thirty blocks and the logit head are two submissions a token, and the first of those is a recording made once and submitted again — a token's four hundred dispatches cost more to write down than the card takes to run some of them. What crosses the bus is the embedding in, the position, and the logits back.

### Vulkan Kernel Tuning

The kernels are written against the card's four-byte integer dot product, which is one instruction for what the unpacked loop spends eight on. A Q4_0 word holds eight weights as nibbles and a single mask puts four of them in the four bytes the instruction reads; the accumulator is integer, so the answer does not move. It is what took the product kernels from about 250 gigabytes a second to between 360 and 530, and the logit head to 355. Those figures for the block kernels were taken before the benchmark read cold weights, and a matrix reread inside one submission on a card with 64 MB of last-level cache flatters itself; read them as the size of a change rather than as a rate. The head's 355 is a cold measurement of the whole tensor. A card without `VK_KHR_shader_integer_dot_product` gets an error where it would get a device, and `-vulkan` fails on it rather than falling back — a model half on a card the caller believed it was wholly on is a model whose speed nobody can explain.

A model whose attention is on the card reads its prompt in stretches of five hundred and twelve positions, because the keys and values are the card's and the two caches must not part. Five hundred and twelve of them in one pass is what a batch was always for: every matrix of the model read once for all of them instead of once for each, which is the whole of the difference between a prompt at the memory ceiling and a prompt five hundred times over it. A mixture is no exception any more: its expert stack is read by expert rather than by column, so the eight matrices a position routes to are read once for the positions that wanted them. Which way round is cheaper is the width's business, and the two cross where the pass is about a sixteenth of the stack wide — eight matrices a column against every expert the model has. Measured on the 26B, positions a second by width with each path forced: by column 184, 328, 557, 778, 990, 1240, 1370, 1516 at widths 2, 4, 8, 12, 16, 24, 32, 64, and by expert 106, 207, 441, 720, 943, 1228, 1734, 2765. So a pass narrower than thirty-two answers a column at a time, each dispatch reading that column's own eight matrices; a wider one turns the routing inside out. It used to switch at two, which made the narrowest useful batch — a token drawn for each of two conversations — the worst of both, a hundred and twenty-eight experts read to answer sixteen pairs. Generation reads the same binaries it always did: the column count is compiled into the kernel rather than pushed, so there are several of each and a token draws the narrowest.

A pass need not belong to one conversation. A block's key-value cache on the card is one buffer holding as many rings as the server was given slots, laid end to end, and the position buffer every column already carries has a fourth number in it saying which ring this column writes and reads — the context is cut between the conversations rather than multiplied, so four of them cost what one of four times the context did. The scores kernel is what constrains the rest: it answers thirty-two columns to a workgroup off the keys of the union of their ranges, and two conversations cannot share that scratch, because a position means a different entry of the cache in each. So the pass is cut into runs of columns that share a slot before the tiles are laid out, and a tile never straddles two. A prompt is one run and tiles exactly as it did; a token drawn for each of four conversations is four runs of one column, which is the shape generation has anyway. What that buys is one read of a gigabyte of weights answering four clients: the 26B A4B draws 82.1 tokens a second for one conversation and 255.0 for eight.

Which kernel a pass runs is the width's business. A token and a short stretch draw the mat-vec, one row of the answer to a team of eight lanes; a stretch above eight draws a tiled product instead, which stages both operands in shared memory and keeps a tile of the answer in registers. Both read their operands sixteen bytes at a time, and both fetch a step of the walk into registers before the step they are computing, so that the card waits for memory with a step of arithmetic in hand rather than at a barrier.

A matrix with few rows gives the tiled product few workgroups — the Qwen3 4B's down projection gives eighty, against sixty-four compute units that hold two of these each — so for those the shared dimension is cut into four slices, one workgroup apiece, and a pass over the answer adds them back. The row count decides: a matrix wide enough to fill the card is left alone, and measured on one that already is, the same split is a fifth slower. It takes the down projection of a Qwen3 4B block from 6.36 milliseconds over thirty-six blocks to 4.40, and the output projection from 2.78 to 2.07.

Before any of those, the largest single thing was not a kernel. Six buffers carrying the stream from one kernel to the next were allocated host-visible, left over from a per-block API the stack replaced, so every workgroup of every projection was reaching across the bus for its operand. In device memory the same prompt runs half again as fast.

It is not bit-identical to the CPU path and cannot be: the two sum the same products in different orders, and a mixture amplifies that because its intermediate is quantized on the way into the second projection. The Vulkan path is held to the reference tests the CPU path is held to, at the same tolerances, and `gemma/vulkan_test.go` measures where it sits — nearer llama.cpp than this engine's own portable Go path, which is also not the AVX2 one.

There is no cgo: `vk/` opens `libvulkan.so.1` through `purego`, and `CGO_ENABLED=0 go build ./...` still passes. A machine with no Vulkan loader is one where the flag fails and everything else works.

### Generation Speed vs Prompt Speed

On a card, golem is ahead of llama.cpp on both Gemma models, generating and reading, and within two hundredths of it generating on both Qwen3 ones. Qwen3.8 came later and sits apart: level reading a prompt, behind generating, for a reason that is the architecture's rather than a kernel's.

| tokens a second | golem gen | llama gen | golem pp64 | llama pp64 | golem pp256 | llama pp256 | golem pp512 | llama pp512 |
| --------------- | --------: | --------: | ---------: | ---------: | ----------: | ----------: | ----------: | ----------: |
| Gemma 4 26B A4B | **133.5** |     124.8 |   **2061** |        833 |    **3951** |        2918 |    **4541** |        4038 |
| Gemma 4 12B     |  **65.4** |      64.6 |   **1499** |        983 |    **2611** |        2471 |        2858 |        2976 |
| Qwen3 4B        |     167.0 |     169.7 |   **3950** |       2952 |    **5854** |        4545 |        5895 |        6125 |
| Qwen3 0.6B      |     359.4 |     365.4 |  **13915** |       9966 |   **23787** |       19159 |   **23995** |       22306 |
| Qwen3.8 27B     |      30.1 |      32.8 |    **799** |      628.7 |    **1134** |      1127.8 |    **1236** |      1234.7 |

golem reads a prompt faster than llama.cpp does on every model here at 64 and 256 positions, and on the 26B A4B, the 0.6B and Qwen3.8 at 512 as well.

Generation was the side that was left, at 0.83 of llama.cpp on the 26B A4B and 0.93 on the 12B against 0.98 on both Qwen3 models, and the shape of that table was the answer: **the models with a gap were exactly the models with a logit softcap.**

Gemma caps its logits at thirty. `Logits` did it after the product, on the CPU, in float64, one entry at a time — two hundred and sixty-two thousand hyperbolic tangents, 3.25 ms on an i7-9700K, a third of a token spent with the card already idle. Qwen3 has no softcap and returns straight from its product, which is why it had no gap. llama.cpp does the same cap as a graph node on the GPU, and its own profiler puts that node at 6.16 microseconds.

It is a push constant on the head's shader now: the last line of the product writes `limit*tanh(total/limit)` instead of `total`, and `Logits` does not walk the vocabulary again. `nn.Softcap` still exists for the CPU path and takes the worker pool. On the 26B A4B the token went from 9.897 milliseconds to 7.489 — **101 tokens a second to 133.5**.

Two things are worth keeping from how it was found, because no profile showed it. A GPU timeline cannot see a CPU loop, and the loop was in the seam between them: `Forward` alone measured 6.322 ms, `Forward` and `Logits` together 9.897, and the head's own benchmark 1.72. The 1.85 that belonged to neither was the whole of it. And llama.cpp's own `GGML_VK_PERF_LOGGER`, run on generation rather than on a prompt, put their every operation beside ours in one pass; nothing else in their table was more than a few per cent from ours.

What is left is Qwen3, at 0.98 on both sizes, where the difference is small and is in the kernels rather than beside them.

Qwen3.8's own gap, at 0.92, is not that story. Forty-eight of its sixty-four blocks are gated delta nets, and a delta net rewrites a 128x128 state a head at every position: the work is per token and there is no batch to spread it over, which is exactly why the prompt side reaches llama.cpp's rate and the generating side does not. What answers it there is not a faster kernel but a second token in the same pass, which is what the checkpoint's prediction block gives — `qwen35/mtp.go`, and 46.0 tokens a second against 30.1.
