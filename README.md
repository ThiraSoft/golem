<div align="center">
  <img src="assets/logo.svg" alt="golem" width="420">

**Gemma, Qwen and Pocket TTS in a single static Go binary.**  
_No Python. No cgo. Runs on your CPU, or on your GPU with `-vulkan`._

[![test](https://github.com/ThiraSoft/golem/actions/workflows/test.yml/badge.svg)](https://github.com/ThiraSoft/golem/actions/workflows/test.yml)
[![Go Reference](https://pkg.go.dev/badge/github.com/ThiraSoft/golem.svg)](https://pkg.go.dev/github.com/ThiraSoft/golem)
[![MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

</div>

---

A golem is inert matter given a voice. That is what these engines do to a file of weights.

**Golem** runs Gemma 4, Qwen3, Qwen3.8 and Kyutai speech models locally, on the processor or on a Vulkan GPU that is roughly ten times faster. It is one Go binary of about twelve megabytes, and there is nothing else to install.

```bash
go install github.com/ThiraSoft/golem/cmd/golem-cli@latest

golem-cli -model gemma-4-E2B-it-QAT-Q4_0.gguf -p "Explain a mutex in one sentence."
```

Without a Go toolchain, take a static binary from [the releases](https://github.com/ThiraSoft/golem/releases): Linux and macOS, amd64 and arm64. Windows does not build yet, because `tensors/` maps a checkpoint through a call that platform does not have.

There is no configuration file to write. The engine reads `general.architecture` out of the GGUF and opens whichever implementation matches, which is why no command below names a model family.

## Who this is for

If you ship Go and you need a model to run **on the machine your software already runs on**, this removes the part of that job that hurts. No Python runtime to install beside your binary. No four-gigabyte CUDA image. No cgo, so `GOOS=linux GOARCH=amd64 go build` still produces something you can scp to a box you do not control and run on a kernel you did not choose.

That covers on-premise deployments, edge and embedded boxes, air-gapped machines, CI, and any product where "please install Python 3.11 first" is not an acceptable sentence to put in front of a customer.

If you are running models on your own workstation and a package manager is fine, [Ollama](https://ollama.com) is friendlier and you should use it.

### Why not just bind to llama.cpp

Because a cgo binding is no longer a static binary. It brings back the C toolchain, the cross-compilation problems, and a build that breaks differently on every distribution. That is the entire cost this project is paying its way out of, and the price of admission was a full set of kernels: AVX2 and NEON in Go assembly, Vulkan compute bound through `purego` at runtime rather than linked.

The bet is that it is worth it only if the result is actually fast. See [the numbers](#speed).

## What you get for it

- **One binary, four dependencies.** `purego` loads Vulkan at runtime, and three packages read file formats the standard library does not (WebP, MP3, FLAC). `CGO_ENABLED=0 go build ./...` passes.
- **A library, not just a command.** `engine.Open` hands back a forward pass, a vocabulary and a chat template. The commands in `cmd/` are thin.
- **It is fast enough to be the real thing.** Tuned AVX2 kernels sit at llama.cpp's level on CPU. The Vulkan backend does too, on the card it was measured on. See [the numbers](#speed) and the caveats that come with them.
- **Nothing is asserted, everything is checked.** No layer here is considered correct until its intermediate activations match llama.cpp or PyTorch waypoint by waypoint. A model that answers fluently and wrongly is this project's named failure mode.
- **Every number in this file is a benchmark in this repository**, run on the machine named next to it. Nothing is extrapolated, and where golem loses it says so.

## What it does

| | |
| --- | --- |
| **Text** | Gemma 4 (E2B, 12B, 26B A4B), Qwen3, Qwen3.8 27B, and Prism's ternary Bonsai 2 27B |
| **Vision** | Gemma 4 and Qwen3.8, from a projector file |
| **Audio in** | Gemma 4 hears WAV, MP3 and FLAC. Kyutai STT transcribes English and French |
| **Audio out** | Kyutai Pocket TTS, 12 shipped models across 6 languages, plus voice cloning |
| **Images** | Krea 2 from ComfyUI's fp8 files, with its LoRA, the same picture as ComfyUI for the same seed, and on a colour to key out |
| **Serving** | OpenAI-compatible HTTP API, tool calls, continuous batching, JSON schemas and GBNF grammars |
| **Weights** | GGUF, every K-quant llama.cpp writes, Prism's ternary PQ2_0 and PTQ1_0, and golem's own `.golem` format |

## Quickstart

Every command below is `go install github.com/ThiraSoft/golem/cmd/<name>@latest`, or `go build ./cmd/<name>` from a clone.

**Chat in the terminal:**

```bash
golem-cli -model Qwen3-4B-Q4_0.gguf -p "Explain a mutex in one sentence." -stats
golem-cli -model Qwen3-4B-Q4_0.gguf -vulkan          # on the GPU
```

**Serve an OpenAI-compatible API:**

```bash
golem-server -model Qwen3-4B-Q4_0.gguf -addr 127.0.0.1:8080 -parallel 4
```

`-parallel N` cuts the context into N slots and holds N conversations at once. Whatever is waiting when a pass is built rides in that pass, so four clients wanting a token are one read of the weights instead of four. Details in [`cmd/golem-server/README.md`](cmd/golem-server/README.md).

**Look at a picture:**

```bash
golem-cli -model gemma-4-E2B-it-QAT-Q4_0.gguf \
    -mmproj mmproj-gemma-4-E2B-it-QAT-BF16.gguf \
    -image photo.png -p "What is in this picture?"
```

The same flags with `-audio question.wav` make Gemma listen. See [`gemma/README.md`](gemma/README.md).

**Speak, and transcribe:**

```bash
pocket-tts -voice voice.safetensors -o hello.wav "Bonjour le monde."
pocket-tts -clone someone.wav -save-voice someone.safetensors   # 20s of audio is enough

golem-cli -stt ~/models/stt-1b-en_fr -listen    # microphone in, words out
```

Voice cloning needs no training and no Python. Transcription streams at twice real time on eight CPU cores, and about ×3 on the card. See [`stt/README.md`](stt/README.md) and [`pockettts/README.md`](pockettts/README.md).

**Ask for JSON and get JSON:**

```bash
golem-cli -model Qwen3-4B-Q4_0.gguf -json-schema city.json -p "Give me the city of Lyon."
```

A model asked for JSON in prose usually obliges. A model drawing inside a grammar cannot do otherwise, because every token that would break the document is refused before the draw. `-json`, `-json-schema` and `-grammar` on the command line, `response_format` and `grammar` over the API. The engine is a port of llama.cpp's own, compared against it rule by rule. See [`grammar/README.md`](grammar/README.md).

**Or use it as a library:**

```go
m, err := engine.Open("Qwen3-4B-Q4_0.gguf", 4096)   // reads the architecture, opens the engine
defer m.Close()

ids := m.Vocab.Encode("Explain a mutex in one sentence.", true, true)
hidden := m.Forward.ForwardBatch(ids, 0)

logits := make([]float32, m.Vocabulary)
m.Forward.Logits(hidden[len(hidden)-1], logits)
```

`engine.Open` hands back one shape whichever family the file is: the forward pass, the vocabulary, the chat template the checkpoint carries, and what to sample with. The commands in `cmd/` are thin wrappers over it, and [pkg.go.dev](https://pkg.go.dev/github.com/ThiraSoft/golem) has the rest.

## Speed

On an RX 9070 XT against llama.cpp's Vulkan build (`ba1df050f`, b9603), same Q4_0 files, both sides warmed, measured on 2026-09-03 in one sitting. Tokens a second, higher is better:

| | golem gen | llama gen | golem pp512 | llama pp512 |
| --- | ---: | ---: | ---: | ---: |
| Gemma 4 26B A4B | **153.4** | 125.1 | **4434** | 4059 |
| Gemma 4 12B | **72.3** | 64.6 | 2901 | **2978** |
| Qwen3 4B | **180.5** | 172.0 | 5977 | **7117** |
| Qwen3 0.6B | 368.2 | **407.9** | 23445 | **23677** |

**Read the table as a whole: golem reaches llama.cpp's level here, and that is the entire claim.** This is one card, one driver, one afternoon. A kernel that wins on RDNA 4 at these shapes need not win on another architecture or another checkpoint, and nobody has run it there. The absolute rates do not travel either, so the column worth reading is the difference between the two engines rather than the rate itself.

Qwen3.8 27B is absent on purpose. At 15.65 GiB on a heap of 15.92 it leaves no room on a card that is also driving a desktop, and what a table would compare there is memory pressure rather than two engines. [`qwen35/README.md`](qwen35/README.md) has that measurement.

Bonsai 2 27B is Qwen3.8 27B in ternary weights, 6.7 GiB for two bits a weight and 5.5 for the dense 1.75. Stock llama.cpp does not read it, so the comparison is Prism's own fork (`PrismML-Eng/llama.cpp`, `bdc23b5`) built for Vulkan, on the same card, measured on 2026-09-23:

| | golem gen | fork gen | golem pp512 | fork pp512 |
| --- | ---: | ---: | ---: | ---: |
| Bonsai 2 27B PQ2_0 | **46.0** | 9.2 | **1253** | 924 |
| Bonsai 2 27B PTQ1_0 | **15.8** | 9.4 | **1128** | 423 |

Both packings hold the same trits. The dense one reads 17 % fewer bytes and decodes five weights a byte one at a time, which on this card costs more than the bytes save: take PQ2_0 unless memory is what is short. Held against the fork's logits at every position of a prompt, both answer within 0.0001 nats. Both prefill columns are warm, the median of three passes after one that is not counted, as `llama-bench` does it. [`qwen35/README.md`](qwen35/README.md) has the rotation these files carry and how drafting is grafted onto them.

On an i7-9700K with eight threads and Q4_0 weights: Gemma E2B draws 22.6 tokens a second and reads 204, the 12B does 5.0 and 42, the 26B A4B does 13.1 and 51, Qwen3 4B does 14.6 and 110. Against llama.cpp on the same machine those four are a tie on generation and between ×1.06 and ×1.33 reading a prompt.

The one model golem loses is Qwen3 0.6B, and [`qwen/README.md`](qwen/README.md) says why: at 320 MB the weights fit close enough that memory stops being the limit, and what is left is arithmetic. This engine is built for the regime where reading the weights is the cost, and it says so where it is not.

[ARCHITECTURE.md](ARCHITECTURE.md) has the kernel work behind all of this, the per-op breakdown of one pass against llama.cpp's, and what the next bottleneck is.

## Three things that are not in other engines

### A mixture's experts need not be on the card

Gemma 4 26B A4B keeps 12.85 GB of experts and reads eight matrices out of a hundred and twenty-eight per block. Eleven of those twelve gigabytes sit untouched on any given token, which means the cost of streaming a mixture is what a token *activates* rather than what the model *has*.

`GOLEM_MOE_EXPERTS_HOST=1` leaves the expert stacks in system memory that the card can address and lets the kernels read them where they lie. No shader knows the difference. That takes the model's footprint on the card from 13.6 GiB to 1.3, and a cache of the experts a token keeps asking for buys the speed back:

| experts kept in VRAM | that much VRAM | tokens/s |
| --- | ---: | ---: |
| 2 of 128 (the floor) | 0.2 GB | 7.8 |
| 32 of 128 | 3.2 GB | 30.2 |
| **51 of 128** | **5.1 GB** | **56.2** |
| all 128 (fully resident) | 12.9 GB | 131.6 |

Two fifths of the pool buys four fifths of the tokens, and the answers are identical in every row. The full curve, the block-by-block residency that runs a 27 GB checkpoint on a 16 GB card, and the reason the router had to move onto the GPU are in [`gemma/README.md`](gemma/README.md).

### Qwen3.8 drafts its own next token

The Qwen3.8 checkpoint ships a sixty-fifth block whose job is to guess the token *after* the one just decided. Guess right and the next pass verifies two tokens for the price of one, because the card reads a block's weights once whether the pass carries one column or two. Guess wrong and it costs nothing beyond the pass it rode on. Every token returned is drawn from the model's own distribution, so none of the accept-reject correction that drafting with a separate model needs applies here, and a test asserts the answer is the same either way.

On an RX 9070 XT: 27.6 tokens a second one at a time, **40.1 drafting**, 70% of drafts accepted. Bonsai 2 ships without the block, and the original model's grafts onto it unchanged: 46.0 tokens a second becomes 62.8, 57% of drafts accepted. On the CPU it is refused rather than offered, and [`qwen35/README.md`](qwen35/README.md) has the measurement that settles it.

### Its own weight format

golem reads GGUF like everyone else. It also writes `.golem`, which trades speed for size. On Qwen3-4B against the same bf16 reference, 4088 tokens of wikitext:

| | size | perplexity | KL divergence | top-1 agreement |
| --- | --- | --- | --- | --- |
| bf16 | 7.5 GiB | 19.29 | | |
| Q4_K_M | 2.33 GiB | 20.04 | 0.0715 | 90.1 % |
| **`.golem` H4G** | **1.96 GiB** | 20.07 | **0.0495** | **90.2 %** |
| Q3_K_M | 1.93 GiB | 24.03 | 0.2453 | 79.8 % |
| **`.golem` H3G** | **1.54 GiB** | **21.67** | **0.1715** | **84.6 %** |

H3G is 20 % smaller than Q3_K_M and ahead of it on every column. A weight is a path through a trellis found by Viterbi, QTIP's scheme: a window of the code stream is hashed into a small trained table, and each window gives two weights.

```bash
golemquant -model Qwen3-4B-BF16.gguf -out Qwen3-4B.golem -bits 3 -calib wiki.txt
```

On Qwen3.8-27B, which is the model the format is for, H3G is 10.45 GiB and closer to bf16 than Q3_K_M (KL 0.0646 against 0.0900), and it generates at 38.8 tokens a second against Q4_0's 34.1 on the same card, about 60 to 67 drafting. H4G is 13.34 GiB, KL 0.0235 from bf16, 33.5 tokens a second and 56.3 to 70.8 drafting. [`compress/README.md`](compress/README.md) has the method, the measurements, and the one-weight trellis these two replaced.

## How it is known to be right

**No layer is deemed correct until its intermediate activations match the reference implementation.**

Scripts load the real weights, inject a deterministic input, and write every intermediate quantity into `testdata/`. The Go tests read those files back, so they need neither Python nor llama.cpp at test time.

For `gemma/` the reference is llama.cpp itself, instrumented, because a bf16 reference would bury a mistake under its own quantization error. For `pockettts/` and `stt/` it is PyTorch in float32, layer by layer, to a few parts in a million end to end.

## Who wrote this

The code in this repository was written by an AI agent, [Claude](https://claude.com/claude-code), directed and reviewed by a human. Some of the Bonsai 2 work was written by Gemini under Claude's review. The git history says so on the commits themselves.

That is worth stating plainly rather than leaving to be discovered, and it is also the reason the section above exists. An agent will happily produce a layer that runs, returns plausible tokens and is quietly wrong in the fourth decimal place, and reading the diff does not catch that. Recorded activations do. Every claim in these files is either a test in the repository or a benchmark in it, because on this project that is the only kind of claim worth making.

Judge it the way you would judge any dependency you did not write: run the tests, check the numbers, read the parts you are about to trust.

## Project structure

- `cmd/golem-cli`, `cmd/golem-server`, `cmd/pocket-tts`, `cmd/golemquant` are the commands.
- `engine/` reads the architecture out of a GGUF and opens the engine that implements it.
- `gemma/`, `qwen/`, `qwen35/`, `pockettts/`, `stt/`, `nomic/` are standalone engines. They do not import one another.
- `nn/` and `vk/` are the shared kernels: quantized AVX2 and NEON, and Vulkan compute.
- `compress/` is the `.golem` format: calibration, the pair-trellis codec, and the conversion pipeline.
- `grammar/` is GBNF and the JSON Schema converter that feeds it.
- `internal/kyutai/` is the Mimi codec, shared by both directions of speech.
- `tensors/`, `token/`, `chat/`, `sample/`, `audio/`, `imageio/` are the rest of the shared layer.
- `ref/` is what recorded each test fixture, and how to record it again.

## Tests

```bash
go test ./...       # correctness on every model and the card, about 17 minutes
```

Weights are not in this repository, and every test that needs one skips cleanly when it cannot find it. That is why the same command is safe on a machine with no models and no card, and why CI runs it.

**Use `-p 1` if you have a GPU.** Three of these packages put whole models on the card, two at once want more than 16 GB, and what comes back is a lost device in whichever test binary happened to be second.

Anything slower than thirty seconds waits behind `GOLEM_FULL_TEST=1` and runs a package at a time. [CONTRIBUTING.md](CONTRIBUTING.md) says which tests those are and why.

## Contributing

The thing we need most is **ARM benchmarks**. The arm64 kernels are correct and tuned by nobody: written and verified under emulation, never once timed on real hardware. If you have Apple Silicon or a Graviton instance, run [`./benchmark-arm.sh`](benchmark-arm.sh) and share what it prints.

After that, bug reports (especially a parity test failing on your setup) and NEON tuning. [CONTRIBUTING.md](CONTRIBUTING.md) has the rest.

## License and credits

Golem is [MIT Licensed](LICENSE).

Standing on the shoulders of giants: [llama.cpp and ggml](https://github.com/ggml-org/llama.cpp), [Kyutai Pocket TTS](https://github.com/kyutai-labs/pocket-tts), [Google Gemma](https://ai.google.dev/gemma), [QTIP](https://github.com/Cornell-RelaxML/qtip) for the bitshift trellis under `.golem`, and [FreeToken](https://arxiv.org/abs/2608.16157), which asked the expert-caching question first and answered it differently.
