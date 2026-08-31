<div align="center">
  <img src="assets/logo.svg" alt="golem" width="420">

**Gemma, Qwen and Pocket TTS in a single static Go binary.**  
_No Python. No cgo. No GPU required — but with `-vulkan` it matches or exceeds llama.cpp's Vulkan performance on AMD hardware._

[![test](https://github.com/ThiraSoft/golem/actions/workflows/test.yml/badge.svg)](https://github.com/ThiraSoft/golem/actions/workflows/test.yml)
[![Go Reference](https://pkg.go.dev/badge/github.com/ThiraSoft/golem.svg)](https://pkg.go.dev/github.com/ThiraSoft/golem)
[![MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

</div>

---

A golem is inert matter given a voice. That is what these engines do to a file of weights.

**Golem** is a set of inference engines written in pure Go. Run Gemma 4, Qwen3, Qwen3.8 and Kyutai Pocket TTS locally with `go build`, a GGUF file, and your CPU — or your Vulkan GPU.

## ✨ Features

- **Zero Friction**: Compiles to a single static binary. No Python environment, no `cgo`, no runtime to install.
- **Pure Go, four dependencies**: `purego` for the Vulkan loader, and three file formats the standard library does not read — WebP, MP3, FLAC. `CGO_ENABLED=0 go build ./...` passes.
- **OpenAI Compatible**: Drop-in replacement for OpenAI API clients, tool calls included.
- **Multimodal**: Text, Vision (images) and Audio (WAV/MP3/FLAC) via Gemma 4; Qwen3.8 sees too, its tower checked against llama.cpp waypoint by waypoint.
- **Verified, not asserted**: no layer is deemed correct until its intermediate activations match llama.cpp or PyTorch, waypoint by waypoint.
- **Fast on CPU**: keeps pace with `llama.cpp` on tuned AVX2 kernels — ahead reading prompts, level generating, except on the smallest model, where the weights stop being the cost and it says so.
- **Vulkan GPU**: bound through `purego` rather than cgo. Measured on AMD against `llama.cpp`'s own Vulkan build: ahead of it reading prompts on all five models, and generating on both Gemma ones. The table below says where it is behind, and by how much.
- **Its own weight format**: `.golem` is 18 % smaller than llama.cpp's Q3_K_M on Qwen3-4B and ahead of it on every measure — a trellis codebook with no lookup table, converted on the card. [What it costs](#-golem--the-engines-own-weight-format).
- **Serves several clients at once**: `-parallel N` holds N conversations and carries a token for each of them through one read of the weights, on the card as well as on the processor — Qwen3.8 on the card excepted, for a reason [written below](#-several-conversations-one-pass).

## 🚀 Quickstart

Download a GGUF and run. No configuration file — the engine reads `general.architecture` out of the file and opens whichever engine implements it, so no command here names a model family.

### CLI (chat in the terminal)

```bash
go build ./cmd/golem-cli

./golem-cli -model gemma-4-E2B-it-QAT-Q4_0.gguf -p "Explain a mutex in one sentence." -stats
```

### Server (OpenAI-compatible API)

```bash
go build ./cmd/golem-server

./golem-server -model Qwen3-4B-Q4_0.gguf -addr 127.0.0.1:8080
```

`-parallel N` cuts the context into N slots, each holding its own conversation.

## 👥 Several conversations, one pass

Whatever is waiting at the moment a pass is built goes into that pass. Four clients each wanting a token are four tokens in one read of a gigabyte of weights, rather than four reads — llama.cpp's continuous batching, and it now works on the card as well as on the processor. The context is cut between the slots rather than multiplied, so `-context 4096 -parallel 4` is four conversations of 1024 and the memory is what it was.

Gemma 4 26B A4B on an RX 9070 XT, drawing a token for each conversation from 448 positions of context:

| conversations | 1 | 2 | 4 | 8 |
| --- | ---: | ---: | ---: | ---: |
| tokens a second | 82.1 | **125.9** | **211.4** | **255.0** |

Through the API, with prompts of about five hundred and forty tokens: one client reads 4055 positions a second and draws 75.3; two clients read 4260 between them and draw 115.6.

What made it possible is that a column now says which conversation it belongs to as well as where it sits, so a block's cache on the card is one ring per conversation rather than one ring. `cmd/golem-server/README.md` has the rest.

**Qwen3.8 on the card is the exception**, and `-parallel` above 1 is refused there rather than fallen back from. Forty-eight of its sixty-four blocks are delta nets, and a delta net keeps a state matrix a head that every token rewrites rather than a ring indexed by position: there is no ring to cut into slots, and answering two conversations off one state is a wrong answer and not a slow one. On the processor it holds slots like the rest.

## ⚡ Vulkan GPU

`-vulkan` puts the whole model on the card. Measured on an RX 9070 XT, the same evening, against `llama.cpp`'s own Vulkan build on the same files:

| tokens a second | golem gen | llama gen | golem pp64 | llama pp64 | golem pp256 | llama pp256 | golem pp512 | llama pp512 |
| --------------- | --------: | --------: | ---------: | ---------: | ----------: | ----------: | ----------: | ----------: |
| Gemma 4 26B A4B | **133.5** |     124.8 |   **2061** |        833 |    **3951** |        2918 |    **4541** |        4038 |
| Gemma 4 12B     |  **65.4** |      64.6 |   **1499** |        983 |    **2611** |        2471 |        2858 |        2976 |
| Qwen3 4B        |     167.0 |     169.7 |   **3950** |       2952 |    **5854** |        4545 |        5895 |        6125 |
| Qwen3 0.6B      |     359.4 |     365.4 |  **13915** |       9966 |   **23787** |       19159 |   **23995** |       22306 |
| Qwen3.8 27B     |      30.1 |      32.8 |    **799** |      628.7 |    **1134** |      1127.8 |    **1236** |      1234.7 |

**Reading a prompt, golem is ahead on all five models at 64 and 256 positions**, and on the 26B A4B, the 0.6B and Qwen3.8 at 512 as well. On the 26B A4B that is a factor of two and a half at sixty-four positions; on Qwen3.8 the two engines land within a tenth of a per cent of each other at 512, which the [section below](#-reading-a-prompt-on-the-card) takes apart op by op.

**Generating, golem is ahead on both Gemma models** and within two per cent on both Qwen3 ones. Qwen3.8 generates slightly behind due to the heavy per-token state updates of its gated delta net, which cannot be amortised across a batch.

The card holds the whole model: 12.8 GiB for the 26B A4B, which is why sixteen is the smallest card that can run it, and about nine seconds of upload. `-vulkan` is all or nothing and says so: a card without `VK_KHR_shader_integer_dot_product`, a machine with no Vulkan loader, a model too large for the card — each is an error at startup rather than a silent half-move. Without the flag, everything runs on the CPU as before.

_(See [ARCHITECTURE.md](ARCHITECTURE.md) for the kernel work behind these numbers.)_

## 🔮 Qwen3.8 drafts its own next token

The Qwen3.8 checkpoint ships a sixty-fifth block: a multi-token-prediction head that guesses the token *after* the one just decided, from the state the trunk has already computed. Guess right and the next pass verifies two tokens for the price of one. Guess wrong and it costs the pass it rode on and nothing else — every token returned is drawn from the model's own distribution, so this needs none of the accept-reject correction that drafting with a *separate* model does. The answer is the same with drafting on and off, and a test asserts it.

Qwen3.8 27B, a seventeen-token prompt, greedy:

| | prompt | a token at a time | drafting | drafts accepted |
| --- | ---: | ---: | ---: | ---: |
| RX 9070 XT | 119.6 /s | 31.1 t/s | **46.0 t/s** | 70% |
| i7-9700K, 8 threads | 0.7 /s | 0.70 t/s | _refused_ | — |

Seventeen positions is a narrow pass, which is why the card reads this prompt at a hundred and twenty a second and a five-hundred-and-twelve-position one at 1236 — the [section below](#-reading-a-prompt-on-the-card) is that curve.

**Drafting is refused on the processor**, and the reason is one measurement. A speculative step costs a draft plus a pass of two columns, against the one-column pass it hopes to replace:

| cost, as a fraction of one token | on the card | on the processor |
| --- | ---: | ---: |
| the draft | 0.08 | 0.03 |
| the pass of two columns | **0.94** | **1.97** |
| drafts that must be accepted to break even | 2% | 99% |

The card reads a block's weights once whether the pass carries one column or two, so verifying two tokens costs what drawing one did. The processor at this size is bound by arithmetic rather than by reading the weights, so two columns cost two columns — and no acceptance rate can pay for that. `qwen35/cost_test.go` is where both tables come from.

## 📈 Reading a prompt on the card

The same bargain reads a prompt. A pass carries up to five hundred and twelve positions, and what one costs says how far that goes:

| positions in the pass | 1 | 16 | 32 | 64 | 128 | 256 | 512 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| the pass | 31.2ms | 70.0ms | 63.2ms | 80.1ms | 140.2ms | 225.8ms | 417.8ms |
| positions a second | 32 | 229 | 506 | 799 | 913 | 1134 | **1236** |

Thirty-two positions cost *less* than sixteen, and that is not a misprint: sixteen is the widest the mat-vec builds, and thirty-two is where the tiled products take over.

**A prompt reads at 1236 positions a second, against 62 when a pass carried two.** llama.cpp's Vulkan build reads the same file on the same card at 1234.67 ± 2.00 (`llama-bench -p 512 -r 3`). Both benchmarks do the same work — 512 tokens from an empty cache, both warm, both synchronised, load time excluded — with one difference, and it is ours: golem reads every position's hidden state back over the bus where llama.cpp keeps only the last.

### Where a pass goes

Both engines report their own GPU timers op by op — `vk.Timeline` here, `GGML_VK_PERF_LOGGER=1` there. One pass of 512 positions of the 27B, in milliseconds:

| | golem | llama.cpp |
| --- | ---: | ---: |
| feed forward — gate and up | **132.6** | 149.0 |
| feed forward — down | 97.9 | **95.6** |
| delta net — q, k, v, gate, α, β | **55.1** | 60.9 |
| attention | 32.4 | **2.1** |
| delta net — output projection (Q5_K) | **24.1** | 28.0 |
| SwiGLU | 17.1 | **11.2** |
| recurrence, with its normalisations | **15.9** | 23.2 |
| attention — q, k, v | **14.4** | 15.6 |
| everything else | **26.4** | 38.8 |
| **the pass** | **416.0** | 424.4 |

The card is busy 97.6% of that wall clock — the timer's ticks account for all but 2.4% of it — so nothing measurable is lost between dispatches, and every difference above is a kernel.

Attention is the one line where the gap is a shape rather than a margin: golem dispatches the scores, the softmax and the mix as three passes over memory, where llama.cpp fuses them and keeps the block of scores where it computed it. Sixteen of the sixty-four blocks attend at all, which is why thirty milliseconds buys the other engine so much here and why this is the next line to take.

## 🧠 Supported Models

| Family | What it runs | On CPU, vs its reference |
| --- | --- | --- |
| **Gemma 4** | E2B, 12B, 26B A4B (mixture of 128 experts). Text, Vision, Audio. | Reading a prompt ×1.24 (E2B), ×1.33 (12B), ×1.06 (26B A4B). Generating, a tie: ×1.01, ×1.04, ×1.06 — vs llama.cpp |
| **Qwen3** | Dense models, from a GGUF. | 4B: ×1.13 reading, ×1.00 generating. 0.6B: ×0.99 reading, ×0.85 generating — vs llama.cpp |
| **Qwen3.8** | 27B: Dense model featuring forty-eight gated delta nets and sixteen attentions (three to one ratio) — plus the checkpoint's own multi-token-prediction head, which drafts the second token of every pass. Text and Vision. | 0.72 t/s on an i7-9700K: a delta net rewrites a 128×128 state a head every token, and that is arithmetic no kernel makes cheaper. On a card, 30.1 a token at a time, **46.0 drafting** and **1236** reading a prompt, against llama.cpp's Vulkan build at 32.8 and 1234.7 |
| **Pocket TTS** | 12 shipped models across 6 languages, voice cloning included. | ×2.31 and ×1.69 the speed of the PyTorch reference, on the 24- and 6-layer models |

In absolute terms, on an i7-9700K with eight threads and Q4_0 weights: Gemma E2B draws 22.6 tokens a second and reads 204; the 12B, 5.0 and 42; the 26B A4B, 13.1 and 51; Qwen3 4B, 14.6 and 110. Pocket TTS speaks at ×2.94 real time in French, ×6.81 in English.

**The 0.6B is the one this engine loses**, and [`qwen/README.md`](qwen/README.md) says why: at 320 MB the weights fit close enough that the memory bus stops being the limit, and what is left is arithmetic, where llama.cpp's kernels win. This engine is built for the regime where reading the weights is the cost, and it says so where it is not.

## 🗜️ `.golem` — the engine's own weight format

golem reads GGUF like everyone else. It also writes a format of its own, in two
tiers: on the model below, T4G reads closer to the original than llama.cpp's
four-bit quantization while being 16 % smaller, and T3G undercuts llama.cpp's
three-bit quantization by 18 % while beating it on every column.

Qwen3-4B, 4088 tokens of wikitext, every row measured against the same bf16
reference:

| | size | perplexity | KL divergence | top-1 agreement |
| --- | --- | --- | --- | --- |
| bf16 | 7.5 GiB | 19.29 | — | — |
| Q4_K_M | 2.33 GiB | 20.04 | 0.0715 | 90.1 % |
| **`.golem` T4G** | **1.96 GiB** | 20.38 | **0.0526** | **90.6 %** |
| Q3_K_M | 1.93 GiB | 24.03 | 0.2453 | 79.8 % |
| **`.golem` T3G** | **1.57 GiB** | **21.11** | **0.1771** | **83.1 %** |

Read the last two rows together: **T3G is 18 % smaller than Q3_K_M and ahead of it
on every column**, by nearly three points of perplexity and a quarter of the
divergence. The evaluation is paired — both models see the same tokens in the same
eight windows — so the gap is testable, and it is not noise: t = −2.94, T3G ahead
in six windows of eight.

Two things do the work, and neither is new:

- **A trellis instead of a table.** The codebook is a state machine, not a list of
  points: a weight is twelve bits of the code stream hashed by four instructions,
  so there is nothing to look up and nothing to keep in shared memory. That is
  what opens the four-bit tier, where a lattice's table would need 493 KiB against
  the 32 a GPU workgroup has. The structure is QTIP's bitshift trellis.
- **A rotation and a salience scale**, applied per calibration site rather than
  per matrix. This is most of the format's value: dropping it and keeping only
  signs and rotation costs twenty points of perplexity on Qwen3-0.6B, 39.80
  against 60.01.

Three widths — T3G at 3.25 bits a weight, T4G at 4.19, T5G at 5.19 for the logit
head, which is worth more bits than the layers before it. A three-bit file carries
a four-bit head by default, so the 1.57 GiB above is 3.35 bits a weight overall,
not 3.25.

```bash
go build ./cmd/golemquant
./golemquant -model Qwen3-4B-BF16.gguf -out Qwen3-4B.golem -bits 3 -calib wiki.txt
./golem-cli -model Qwen3-4B.golem -vulkan -p "..."
```

The file is a GGUF — same container, same vocabulary, same chat template — with a
tensor type llama.cpp does not know, which is why it is named `.golem` rather than
`.gguf`: the extension is the warning that only this engine reads it. The
conversion runs on the card, and [`compress/README.md`](compress/README.md) has
the method, the measurements, a list of what was tried and did not work, and
what this format is built on.

## 👁️ Multimodal (Vision & Audio)

Provide the projector weights, and Gemma can see and hear — and Qwen3.8 can see:

**Analyze images:**

```bash
./golem-cli -model gemma-4-E2B-it-QAT-Q4_0.gguf \
    -mmproj mmproj-gemma-4-E2B-it-QAT-BF16.gguf \
    -image photo.png -p "What is in this picture?"
```

**Transcribe and answer from audio:**

```bash
./golem-cli -model gemma-4-E2B-it-QAT-Q4_0.gguf \
    -mmproj mmproj-gemma-4-E2B-it-QAT-BF16.gguf \
    -audio question.wav -p "Answer what is asked."
```

**Qwen3.8 looks the same way**, with its own projector:

```bash
./golem-cli -model Qwen3.8-27B-Q4_0.gguf -mmproj mmproj-F16.gguf \
    -image photo.png -p "Describe this image in one sentence."
```

Its tower has no fixed input size: the picture keeps its aspect ratio, is scaled to a grid the token budget allows, and the learned position table is interpolated onto that grid — which is why an image of any shape gives a different number of rows. Every waypoint of it is held to llama.cpp's, worst gap 1.1e-4 at the patches and 6.4e-3 after all twenty-seven blocks. It sees and does not hear: the projector carries no audio encoder.

The tower goes to the card with the model. A 640×426 picture takes 33.6 seconds on an i7-9700K and **1.03 on an RX 9070 XT**, and every waypoint of the card's tower is held to llama.cpp's too — worst gap 9.9e-3 over the twenty-seven blocks, against the same 0.02 the processor's is held to. Reading the prompt and drawing the answer then run wherever the model does.

The projector is 884 MiB in fp16 and shares the card with the model, so it is resident when there is room and streamed one block at a time when there is not: 1.03 seconds an image against 1.41, rather than a fall back to the processor's thirty-three. `GOLEM_VISION_STREAM=1` forces the second path, which is how it is tested on a machine with room for the first.

WAV, MP3 and FLAC, at any rate and any number of channels; the front end downmixes and resamples before the encoder sees anything. The 26B's projector carries no audio weights, so that checkpoint sees and does not hear.

The server reads the OpenAI content parts a client sends — `image_url` for a picture, `input_audio` for a recording — as a `data:` URI, as base64, or as a path on the machine it runs on. It does not fetch either over the network: a server that fetches what a prompt names is a server that can be aimed.

## 🗣️ Speech (Pocket TTS)

```bash
go build ./cmd/pocket-tts

./pocket-tts -voice voice.safetensors -o hello.wav "Bonjour le monde."
```

**Clone a voice** from a recording, without training and without leaving Go — twenty to thirty seconds of mono 24 kHz is enough:

```bash
./pocket-tts -clone someone.wav -save-voice someone.safetensors
./pocket-tts -voice someone.safetensors -o answer.wav "And now I speak in that voice."
```

## 🔬 The Method

**No layer is deemed correct until its intermediate activations match the reference implementation.**

Scripts load the real weights, inject a deterministic input, and write every intermediate quantity into `testdata/`. The Go tests read those files back, so they need neither Python nor llama.cpp at test time. For `gemma/` the reference is llama.cpp itself, instrumented, because a bf16 reference would bury a mistake under its own quantization error; for `pockettts/` it is PyTorch, layer by layer, to a few parts in a million end to end.

Every number in this README is a benchmark in this repository, run on the machine named beside it. Nothing is estimated.

## 🛠️ Project Structure

- `cmd/golem-cli`, `cmd/golem-server`, `cmd/pocket-tts`, `cmd/golemquant` — the commands.
- `engine/` — reads the architecture out of a GGUF and opens the engine that implements it.
- `gemma/`, `qwen/`, `qwen35/`, `pockettts/` — standalone engine implementations; they do not import one another. `qwen35/` is Qwen3.8: a package is named for the architecture the GGUF declares, and this checkpoint declares `general.architecture = qwen35`, as llama.cpp's own `models/qwen35.cpp` does.
- `nn/` & `vk/` — the shared kernels: quantized AVX2 and NEON, and Vulkan compute.
- `compress/` — the `.golem` format: calibration, the trellis codec, and the conversion pipeline `golemquant` drives.
- `tensors/`, `token/`, `chat/`, `sample/`, `audio/`, `imageio/` — the rest of the shared layer.
- `ref/` — what recorded each test fixture, and how to record it again.

## 🤝 Contributing

We want to make Golem the best pure-Go inference engine available. We especially need:

1. **ARM benchmarks**: the arm64 kernels are correct and tuned by nobody — written and verified under emulation, never once timed on real hardware. Run [`./benchmark-arm.sh`](benchmark-arm.sh) on Apple Silicon or Graviton and share the results.
2. **Bug reports**: if a test comparing against PyTorch or llama.cpp fails on your setup, please open an issue.
3. **Kernel optimization**: help tune the NEON kernels for ARM64.

```bash
go build ./...
go test ./...
```

Weights are not in this repository, and every test that needs one skips cleanly when it cannot find it.

## 📜 License & Credits

Golem is [MIT Licensed](LICENSE).

Standing on the shoulders of giants: [llama.cpp & ggml](https://github.com/ggml-org/llama.cpp), [Kyutai Pocket TTS](https://github.com/kyutai-labs/pocket-tts), [Google Gemma](https://ai.google.dev/gemma), [QTIP](https://github.com/Cornell-RelaxML/qtip) — the bitshift trellis `.golem`'s codec is built on.
