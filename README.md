<img src="assets/logo.svg" alt="golem" width="420">

[![test](https://github.com/ThiraSoft/golem/actions/workflows/test.yml/badge.svg)](https://github.com/ThiraSoft/golem/actions/workflows/test.yml)
[![Go Reference](https://pkg.go.dev/badge/github.com/ThiraSoft/golem.svg)](https://pkg.go.dev/github.com/ThiraSoft/golem)
[![MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**CPU inference engines in pure Go.** No Python, no cgo, no GPU required, no
runtime to install: `go build`, one static binary, a GGUF file, an answer. Three
dependencies outside the standard library, each a file format the standard
library does not read: `golang.org/x/image` for a WebP, `hajimehoshi/go-mp3`
for an MP3 and `mewkiz/flac` for a FLAC. On an eight-core
desktop CPU that binary keeps pace with llama.cpp — ahead on the models large
enough for memory bandwidth to be the limit, behind on the smallest, and every
number below is a benchmark in this repository rather than an estimate.

**And there is a GPU path when you want one.** `-vulkan` is the whole model on
the card, bound through `purego` rather than cgo, so `CGO_ENABLED=0 go build
./...` still passes. On an RX 9070 XT it **reads a prompt faster than
llama.cpp's own Vulkan build does** — on all four models tested up to two
hundred and fifty-six positions, and on the largest and the smallest beyond
that. On the Gemma 4 26B A4B the margin is a factor of two and a half at
sixty-four positions and an eighth at five hundred and twelve. Generation is
the side still behind, at 0.83 to 0.98 of their rate. [The table is
below](#what-it-does-not-do), with both sides measured the same evening on the
same card and the same files.

A golem is inert matter given a voice. That is what these engines do to a file
of weights.

## Run it

```bash
go build ./cmd/golem-cli
./golem-cli -model gemma-4-E2B-it-QAT-Q4_0.gguf -p "Explain a mutex in one sentence." -stats
```

Any GGUF the engines implement will do — the file declares its own architecture
and the right engine is opened for it, so no command here names a model family.
The same weights behind an OpenAI-compatible API, tool calls included:

```bash
go build ./cmd/golem-server
./golem-server -model Qwen3-4B-Q4_0.gguf -addr 127.0.0.1:8080
```

Gemma also looks at pictures — both checkpoints, though not through the same
encoder: E2B's projector is a sixteen-block vision tower, the 12B's is an
embedder whose blocks are the language model's own. The encoder is a second
GGUF, and both commands take it the same way:

```bash
./golem-cli -model gemma-4-E2B-it-QAT-Q4_0.gguf \
    -mmproj mmproj-gemma-4-E2B-it-QAT-BF16.gguf \
    -image photo.png -p "What is in this picture?"
```

Gemma also listens, and the same file carries both encoders: E2B's projector
holds a twelve-block conformer, the 12B's a single projection with no encoder
behind it — forty milliseconds of waveform a token — and the 26B's holds no
audio weights at all, so that checkpoint sees and does not hear.

```bash
./golem-cli -model gemma-4-E2B-it-QAT-Q4_0.gguf \
    -mmproj mmproj-gemma-4-E2B-it-QAT-BF16.gguf \
    -audio question.wav -p "Answer what is asked."
```

WAV, MP3 and FLAC, at any rate and any number of channels; the front end
downmixes and resamples to sixteen kilohertz mono before the encoder sees
anything.

The server reads the OpenAI content parts a client sends — `image_url` for a
picture, `input_audio` for a recording — as a `data:` URI, as base64, or as a
path on the machine it runs on. It does not fetch either over the network: a
server that fetches what a prompt names is a server that can be aimed.

Weights are not in this repository; [what is here, and what is not](#what-is-here-and-what-is-not)
says where each one comes from.

## The engines

| | what it runs | against its reference, on the same CPU |
|---|---|---|
| [`gemma/`](gemma/) | Gemma 4 E2B, 12B and 26B A4B, from a GGUF, pictures and sound included | E2B ×1.01 generating, ×1.24 reading a prompt. 12B ×1.04 and ×1.33. 26B A4B ×1.06 and ×1.06. A picture through the vision tower, ×1.50 — vs llama.cpp |
| [`qwen/`](qwen/) | Qwen3 dense, from a GGUF | 4B ×1.00 and ×1.13. 0.6B ×0.85 and ×0.99 — vs llama.cpp |
| [`pockettts/`](pockettts/) | [Kyutai Pocket TTS](https://github.com/kyutai-labs/pocket-tts), twelve languages, voice cloning included | ×2.22 and ×1.63 the speed of PyTorch, on the 24- and 6-layer models |

In absolute terms, on an i7-9700K with eight threads and Q4_0 weights: Gemma
E2B draws 22.6 tokens a second and reads 204; the 12B, 5.0 and 42; the 26B
A4B, whose feed forward is a mixture of a hundred and twenty-eight experts,
13.1 and 51. Qwen3 4B,
14.6 and 110. Pocket TTS speaks at ×2.94 real time in French, ×6.81 in English.
A 640×426 picture goes through E2B's vision tower in 0.72 seconds, against
llama.cpp's 1.08; through the 12B's embedder, which is one product against a
hundred-megabyte weight, in 35 milliseconds against llama.cpp's 22. A 17-second
audio clip goes through E2B's conformer in 0.58 seconds, matching llama.cpp's
0.58; its mel front end builds 1743 frames in 15 milliseconds against
llama.cpp's 20.

**The 0.6B is the one this engine loses**, and `qwen/README.md` says why: at
320 MB the weights fit close enough that the memory bus stops being the limit,
and what is left is arithmetic, where llama.cpp's kernels win. That is the
honest shape of the trade — this engine is built for the regime where reading
the weights is the cost, and it says so where it is not.

**Every number here is x86-64 with AVX2.** There are arm64 kernels — Q4_0, Q6_K,
the packed product, the float32 and fp16 ones the attention loop is made of, and
the bfloat16 products the speech engine runs on — and they are correct and
untuned. They were written and verified under
emulation, because the machine this was built on is not an ARM one, and no
timing taken under QEMU means anything at all. Where a choice could be read out
of llama.cpp's ARM kernels rather than guessed it was: the packed layout is four
rows there, not the eight AVX2 wants. Expect llama.cpp to win on Apple Silicon
anyway — it has i8mm, tuned tile sizes, and a scheduler that knows a
performance core from an efficiency one, and none of that can be answered
without a machine to measure on. Parts without ARMv8.2 `FEAT_DotProd` fall back
to portable Go for the quantized products, which is correct and slower still.
If you have an arm64 machine, [`benchmark-arm.sh`](benchmark-arm.sh) is how this
paragraph stops being a disclaimer and becomes a number.

Each engine is self-contained. They do not import one another, and nothing in
the shared layer knows they exist.

## The method

**No layer is deemed correct until its intermediate activations match the
reference implementation.**

Scripts load the real weights, inject a deterministic input, and write every
intermediate quantity into `testdata/`. The Go tests read those files back, so
they need neither Python nor llama.cpp at test time — `ref/` says what wrote
each fixture and how to write it again.

For `gemma/` the reference is not PyTorch but llama.cpp itself, instrumented:
the weights on disk are quantized, and a bf16 reference would bury a mistake
under its own quantization error. `ref/gemma/dump_layers.cpp` records every
intermediate under ggml's own names, and the engine is checked against those
recordings — including the parts of ggml that are not the arithmetic anyone
would write, its tabulated GELU and its fp16 caches.

The same rule holds for speed. Every number in these READMEs is a benchmark in
the repository, run on the machine named beside it; nothing is estimated.

## The commands

| | what it does |
|---|---|
| [`cmd/golem-cli`](cmd/golem-cli/) | a conversation with the model, streamed to the terminal |
| [`cmd/golem-server`](cmd/golem-server/) | an OpenAI-compatible API over the same weights, tool calls included |
| `cmd/pocket-tts` | text in, a WAV file out |

Both language commands run either engine, with tools on both. Neither names
one: `-model` takes a GGUF, the file declares its own architecture, and
`engine/` opens whichever implements it.

The server keeps each conversation's tokens between stateless requests and
prefills only what diverged, so a turn costs a turn rather than the whole
prompt. `-parallel` gives it several conversations at once, picks the slot for
a prompt by the longest prefix it already holds, and carries whatever is
waiting through the model in a single pass.

## The shared layer

| package | holds |
|---|---|
| `tensors/` | safetensors and GGUF: metadata, and views on the bytes |
| `nn/` | quantized matrix products with AVX2 kernels, norms, activations, RoPE, convolutions, and the worker pool they are spread over |
| `token/` | tokenizers, one package per family |
| `audio/` | sound formats: reading and writing WAV |
| `sample/` | top-k, top-p, temperature, and a seeded draw over a row of logits |
| `chat/` | a conversation's shape — messages, tools, calls — and the interface an engine implements to write one out |
| `vk/` | Vulkan compute, bound through `purego` rather than cgo: devices, buffers, pipelines, and the ten kernels a whole block is made of — the Q6_K product the logit head is, the plain Q4_0 product that turned out to be most of the rest of the model, the two a mixture's expert branch needs, the attention's cache and scores, the norms, the router, and the three post-norms that close a mixture block |

Nothing is promoted into this layer on the strength of a guess. Code moves here
once two engines are shown to want it, in the same commit that makes them both
use it. `chat/` is the newest of them: the conversation types lived in `gemma/`
until a second engine needed them.

`engine/` is the one package that sits above the engines rather than under
them. It reads `general.architecture` out of a GGUF, opens whichever engine
implements it, and hands back one shape — the forward pass, the vocabulary, the
chat template, and the numbers a startup line prints. It exists so that a
command names no engine, and nothing but a command imports it.

## What it does not do

Worth knowing before you clone it:

- **x86-64 with AVX2 is the only tuned target.** `nn/*.s` is where the speed
  comes from. arm64 has NEON kernels for the quantized generation path, written
  and tested under QEMU on an x86 machine and never once timed on real ARM
  hardware — correct, and tuned by nobody. The vision tower's interleaved kernel
  and the audio decoder's are portable Go there. An Apple or a Graviton runs; it
  will not see the numbers above.
- **Most of a token runs on a GPU, if you ask.** This began as a CPU engine and
  the CPU path is still the one every test is written against. But a token of
  the 26B A4B reads about 2.3 gigabytes and almost all of it is four kinds of
  matrix: the logit head, 0.6 gigabytes, which is the input embedding read the
  other way round; the expert stacks, 0.8, eight matrices at a time out of a
  hundred and twenty-eight; the shared branch beside them, 0.3; and the
  attention's four projections, 0.5. None of it shortens with CPU work, because
  the bytes are the cost — the kernels already run at 37 GB/s on a bus whose
  ceiling is about 43.

  `-vulkan` moves all four, and then everything between them. On the 26B A4B,
  **13.4 tokens a second becomes 103.7**, against llama.cpp's Vulkan build at
  124.8 on the same card, and **the prompt goes from 40 a second to 4541** —
  against their 4038. It costs those matrices being resident — 12.8 gibibytes,
  which is why a card with sixteen is the smallest that can do this — and about
  nine seconds of upload.

  A dense checkpoint goes the same way, because a dense block is a mixture
  block with one branch: the shared branch of a mixture and an ordinary feed
  forward are the same three matrices under the same norm, and what differs is
  the end of the block — one post-norm instead of three, and no routing. On the
  12B, **5.0 tokens a second becomes 60.0**, against llama.cpp's 64.6 on the
  same card.

  Qwen3 runs on the same stack. Four things differ and they are all the file
  being read rather than a second path: an ordinary pre-norm block, where
  neither half is normed on its way back into the stream; a SiLU on the gate
  where Gemma looks ggml's GELU up in a table; scores scaled by one over the
  square root of the head, which Gemma leaves at one because its query norm
  holds them in range; and a value handed to the attention unnormed, which
  Gemma norms. On the 4B, **14.6 tokens a second becomes 167.0**, against
  llama.cpp's 169.7; on the 0.6B, 83.4 becomes 359.4 against 365.4. The head is
  Q4_0 on those checkpoints rather than Q6_K, and reads through
  `shaders/matvec.comp` — the kernel the attention's projections already use.

  The whole of a block goes: the norms, the rotation, the keys and values in
  fp16, the scores, the softmax and the mix, the router, the experts and the
  three post-norms that make a mixture block. Thirty blocks and the logit head
  are two submissions a token, and the first of those is a recording made once
  and submitted again — a token's four hundred dispatches cost more to write
  down than the card takes to run some of them. What crosses the bus is the
  embedding in, the position, and the logits back.

  Not because a norm is expensive. A submission costs sixty-three microseconds
  whatever is in it, and a card handed one and then left alone drops to half
  its clocks; as long as one norm stayed here the block had to come back to
  have it done.

  What that costs is Gemma 4's particulars written twice: a query norm and a
  key norm, two rotation geometries whose heads are not even the same size, a
  value taken from the key before the key was rotated, fifteen blocks at the
  end that compute no keys at all and read what two earlier ones left behind,
  and three roundings to fp16 that are not optional because llama.cpp holds its
  cache that way. `gemma/attention.go` is the other copy and it is the one the
  tests are written against.

  The kernels are written against the card's four-byte integer dot product,
  which is one instruction for what the unpacked loop spends eight on. A Q4_0
  word holds eight weights as nibbles and a single mask puts four of them in
  the four bytes the instruction reads; the accumulator is integer, so the
  answer does not move. It is what took the product kernels from about 250
  gigabytes a second to between 360 and 530, and the logit head to 609. A card
  without `VK_KHR_shader_integer_dot_product` gets an error where it would get
  a device, and the engine falls back to the CPU as it does when there is no
  Vulkan at all.

  A model whose attention is on the card reads its prompt in stretches of five
  hundred and twelve positions, because the keys and values are the card's and
  the two caches must not part. Five hundred and twelve of them in one pass is
  what a batch was always for: every matrix of the model read once for all of
  them instead of once for each, which is the whole of the difference between a
  prompt at the memory ceiling and a prompt five hundred times over it. A
  mixture is no exception any more: its expert stack is read by expert rather
  than by column, so the eight matrices a position routes to are read once for
  the positions that wanted them. Generation reads the same
  binaries it always did: the column count is compiled into the kernel rather
  than pushed, so there are several of each and a token draws the narrowest.

  Which kernel a pass runs is the width's business. A token and a short
  stretch draw the mat-vec, one row of the answer to a team of eight lanes; a
  stretch above eight draws a tiled product instead, which stages both operands
  in shared memory and keeps a tile of the answer in registers. Both read their
  operands sixteen bytes at a time, and both fetch a step of the walk into
  registers before the step they are computing, so that the card waits for
  memory with a step of arithmetic in hand rather than at a barrier.

  A matrix with few rows gives the tiled product few workgroups — the Qwen3
  4B's down projection gives eighty, against sixty-four compute units that hold
  two of these each — so for those the shared dimension is cut into four
  slices, one workgroup apiece, and a pass over the answer adds them back. The
  row count decides: a matrix wide enough to fill the card is left alone, and
  measured on one that already is, the same split is a fifth slower. It takes
  the down projection of a Qwen3 4B block from 6.36 milliseconds over
  thirty-six blocks to 4.40, and the output projection from 2.78 to 2.07.

  On the 12B that took the prompt from 71 tokens a second to 2858, on the
  Qwen3 4B from 179 to 5895, and on the 0.6B from 420 to 23995. llama.cpp's own
  Vulkan build, same card and same files, reads them at 2976, 6125 and 22306.

  Four things closed that gap, and each of them is written up where it lives.
  The expert branch of a mixture reads its stack by expert rather than by
  column, so the eight matrices a position routes to are read once for every
  position that wanted them. The tiled product runs on the matrix cores, at a
  tile the waves divide in both directions — `vk/shaders/matmul_coop.comp`
  carries that whole measurement, including the shapes it could not reach. Its
  staging is double-buffered: the reads of step k+1 are issued before the
  multiplies of step k, so the card waits for memory with thirty-two matrix
  multiplies in hand. And the block of attention scores is a cooperative
  multiply rather than a hundred and twenty-eight shared reads — a wave to each
  sixteen by sixteen tile, the keys read column-major so that nothing is
  transposed anywhere.

  Before any of those, the largest single thing was not a kernel. Six buffers
  carrying the stream from one kernel to the next were allocated
  host-visible, left over from a
  per-block API the stack replaced, so every workgroup of every projection was
  reaching across the bus for its operand. In device memory the same prompt
  runs half again as fast.

  It is not bit-identical to the CPU path and cannot be: the two sum the same
  products in different orders, and a mixture amplifies that because its
  intermediate is quantized on the way into the second projection. The Vulkan
  path is held to the reference tests the CPU path is held to, at the same
  tolerances, and `gemma/vulkan_test.go` measures where it sits — nearer
  llama.cpp than this engine's own portable Go path, which is also not the AVX2
  one.

  There is no cgo: `vk/` opens `libvulkan.so.1` through `purego`, and
  `CGO_ENABLED=0 go build ./...` still passes. A machine with no Vulkan loader
  is one where the flag fails and everything else works.
- **The server is one process around one model.** `-parallel` answers several
  conversations at once and batches them into one pass, the way llama.cpp's
  does — on E2B, 20.7 tokens a second for one client, 34.7 for two, 54.2 for
  four and 73.8 for eight, against llama-server's 20.3, 33.1, 60.5 and 72.9 on
  the same machine — but there is no second model, no distribution and no
  scheduler beyond that.
- **Two language architectures.** Gemma 4 — E2B, E4B, 12B and the 26B A4B,
  whose feed forward is a mixture of a hundred and twenty-eight experts — and
  Qwen3 dense. Qwen3's own mixture checkpoints are not read: the mixture here
  is Gemma 4's, which is not the usual shape and is not a general one.
- **The server speaks two endpoints.** `/v1/chat/completions`, images included
  when a projector is given, and `/v1/models`. No embeddings endpoint and no
  `/v1/completions`.
- **Q4_0, Q4_1, Q6_K, bf16 and float32.** The K-quants beyond Q6_K are not
  read. Q4_1 is where the published Q4_0 builds of Qwen3 keep `ffn_down`, and
  where every `ffn_down` of Qwen3.8-27B is; it has a CPU kernel and no shader
  yet, so a mixed file loads on the CPU path and is refused by `-vulkan`.
- **The prompt path on Gemma is a factor of one and two thirds behind ggml's
  best**, even where it beats the default build; `gemma/README.md` says where
  the remainder sits.
- **On a card, generation is still behind: two hundredths to a sixth,
  depending on the model.** The prompt is not — see the table below and the
  paragraph after it.

  Measured against llama.cpp's own Vulkan build on the same card, which is the
  fair comparison for a Vulkan engine. Both columns of every pair were taken
  the same evening, one benchmark at a time, on an RX 9070 XT against
  llama.cpp `ba1df050f`:

  | tokens a second | golem gen | llama gen | golem pp64 | llama pp64 | golem pp256 | llama pp256 | golem pp512 | llama pp512 |
  | --------------- | --------: | --------: | ---------: | ---------: | ----------: | ----------: | ----------: | ----------: |
  | Gemma 4 26B A4B |     103.7 |     124.8 |   **2061** |        833 |    **3951** |        2918 |    **4541** |        4038 |
  | Gemma 4 12B     |      60.0 |      64.6 |   **1499** |        983 |    **2611** |        2471 |        2858 |        2976 |
  | Qwen3 4B        |     167.0 |     169.7 |   **3950** |       2952 |    **5854** |        4545 |        5895 |        6125 |
  | Qwen3 0.6B      |     359.4 |     365.4 |  **13915** |       9966 |   **23787** |       19159 |   **23995** |       22306 |

  **golem reads a prompt faster than llama.cpp does on every model here at
  sixty-four, a hundred and twenty-eight and two hundred and fifty-six
  positions**, and on the 26B A4B and the 0.6B at five hundred and twelve as
  well; on the 12B and the 4B it is within a twenty-fifth there. On the 26B the
  margin is a factor of two and a half at sixty-four positions and an eighth at
  five hundred and twelve, and it holds at a thousand and twenty-four: 4247
  against 4013.

  Generation is the side that is left. It is 0.83 of llama.cpp on the 26B A4B,
  0.93 on the 12B, 0.98 on the 4B and 0.98 on the 0.6B — a gap that closes as
  the model gets smaller, which is the opposite shape from the one the prompt
  used to have, and it says where the work is. A token is bound by reading the
  weights, so what is left there is the order the weights are read in and how
  the dispatches are scheduled around them, not a kernel to rewrite.

  **Both sides draw a token from the prompt they read.** llama.cpp's
  `test_prompt` hands the whole prompt to one `llama_decode` with a null
  logits pointer, which computes the last token's logit head and no other, so
  golem's prompt benchmarks compute one head too — the whole vocabulary, on
  the card, once for the stretch. Without it the comparison was a prompt
  nobody drew a token from against a prompt somebody did.

  A benchmark of three iterations reads a third low on the largest model: the
  card is at its idle clocks for the first of them and a prompt of sixty-four
  positions is over before it has left them. Every golem number above was
  taken with enough iterations for that to stop mattering.

  Where a pass of five hundred and twelve columns of the 26B A4B now goes,
  from `go test ./gemma -run TestVulkanPromptProfile -v`: the experts' gate and
  up projection 29.5 per cent, the attention scores 15.4, the experts' down
  projection 14.7, the attention's four projections 8.7, and the router 7.7.
  Nothing else is above three.

## What is here, and what is not

No weights, no vocabularies, no voices. They belong to the people who published
them, and they are between three hundred megabytes and three gigabytes each.

- **Gemma 4** weights: a GGUF from Hugging Face, whatever quantization you like
  — the engine reads Q4_0, Q6_K, bf16 and float32, and the tests want the
  QAT Q4_0 build. Point `GOLEM_MODEL` at the file, `GOLEM_MODEL_12B` at a 12B
  one and `GOLEM_MODEL_26B` at a 26B A4B, with `GOLEM_MMPROJ` and
  `GOLEM_MMPROJ_26B` at their projectors: the checkpoints are tested
  separately, because they agree on the architecture's name and on very little
  else. A machine with only some of them skips the others' tests. Google's
  Gemma terms apply to them.
- **Qwen3 dense** weights: a GGUF from Hugging Face declaring `qwen3`. The
  tests want two of the same checkpoint — `GOLEM_MODEL_QWEN` at a bfloat16
  build and `GOLEM_MODEL_QWEN_Q4` at a Q4_0 one made with `llama-quantize
  --pure`, because the kernels and the architecture are two independent places
  for a mistake to hide and bfloat16 removes one of them. Alibaba's Qwen terms
  apply.
- **Pocket TTS** weights, tokenizer and voice states: Kyutai publishes them on
  Hugging Face, and `pockettts/README.md` says which repositories and where the
  engine looks. A voice can also be cloned from any recording, which needs
  nothing but the weights.

Neither are the recordings of the models themselves. `testdata/gemma/layers`,
`layers12`, `quants` and `window`, and `testdata/qwen/layers`, `long` and
`layers_q4`, hold activations, logits and slabs of quantized weights as ggml
computed them; `testdata/layer0`, `pipeline` and `pipeline_en` hold what
PyTorch computes inside Pocket TTS. Those are the models' output, not this
repository's, and each is one command away — `ref/README.md` and
`pockettts/README.md` have them.

What *is* versioned is what the models did not write: tokenizations and chat
templates, which are text and integers. Tests that need a file which is not
there skip rather than fail, so a fresh clone with no model at all still runs
164 of them green, 73 skipped — and a machine with both Gemma checkpoints, a
Qwen3 in two quantizations, the voices and one run of the recorders runs 236.

## Building

```bash
go build ./...
go test ./...
```

Go 1.23 or later, and nothing else. No C compiler, no toolchain, no wheels.

## Feedback wanted

This is one person's reading of two model architectures, checked against their
references and measured on one machine. Both of those are narrow.

What would help most:

- **Numbers from other hardware.** Everything here was tuned on an i7-9700K with
  AVX2 and eight cores:

  ```bash
  GOLEM_MODEL=<a Gemma 4 GGUF> go test ./gemma -run xxx -bench . -benchtime 20x
  ```

  A Zen, an Apple, a machine with more memory channels or none of AVX2 would
  each say something the tuning cannot know.
- **A parity failure.** If a test that compares against llama.cpp or PyTorch
  fails on your setup, that is the most useful bug report this repository can
  get, and the fixtures are exactly what makes it legible.
- **The kernels.** `nn/*.s` is where the time goes. The NEON side has never run
  on real silicon: a timing from an actual Apple or Graviton part, or a tuning
  that a machine confirms, is worth more here than anywhere else in the
  repository. [`benchmark-arm.sh`](benchmark-arm.sh) takes that measurement —
  run it on any arm64 machine and post the `golem-arm.txt` it writes. Most of it
  needs no model files.
- **Judgement calls.** Where the code chose one thing and explained itself in a
  comment, the explanation is a claim; if it is wrong, say so.

Issues and pull requests are both fine. So is a note that says only what you ran
and what happened.

## Standing on

- [llama.cpp and ggml](https://github.com/ggml-org/llama.cpp) — the reference
  `gemma/` is measured against, and the source of the Q4_0 and Q6_K block
  formats the kernels here read.
- [Kyutai Pocket TTS](https://github.com/kyutai-labs/pocket-tts) — the model
  `pockettts/` ports, and the daemon that encodes its voices.
- [Gemma](https://ai.google.dev/gemma) — Google DeepMind's models, of which E2B
  is what `gemma/` runs.
- [GGUF](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md) and
  [safetensors](https://github.com/huggingface/safetensors) — the two file
  formats `tensors/` reads.

## License

MIT. See [LICENSE](LICENSE).
