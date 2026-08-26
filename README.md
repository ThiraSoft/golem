<div align="center">
  <img src="assets/logo.svg" alt="golem" width="420">

**Gemma, Qwen and Pocket TTS in a single static Go binary.**  
_No Python. No cgo. No GPU required — but with `-vulkan` it outreads llama.cpp's own Vulkan build._

[![test](https://github.com/ThiraSoft/golem/actions/workflows/test.yml/badge.svg)](https://github.com/ThiraSoft/golem/actions/workflows/test.yml)
[![Go Reference](https://pkg.go.dev/badge/github.com/ThiraSoft/golem.svg)](https://pkg.go.dev/github.com/ThiraSoft/golem)
[![MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

</div>

---

A golem is inert matter given a voice. That is what these engines do to a file of weights.

**Golem** is a set of inference engines written in pure Go. Run Gemma 4, Qwen3 and Kyutai Pocket TTS locally with `go build`, a GGUF file, and your CPU — or your Vulkan GPU.

## ✨ Features

- **Zero Friction**: Compiles to a single static binary. No Python environment, no `cgo`, no runtime to install.
- **Pure Go, four dependencies**: `purego` for the Vulkan loader, and three file formats the standard library does not read — WebP, MP3, FLAC. `CGO_ENABLED=0 go build ./...` passes.
- **OpenAI Compatible**: Drop-in replacement for OpenAI API clients, tool calls included.
- **Multimodal**: Text, Vision (images) and Audio (WAV/MP3/FLAC) via Gemma 4.
- **Verified, not asserted**: no layer is deemed correct until its intermediate activations match llama.cpp or PyTorch, waypoint by waypoint.
- **Fast on CPU**: keeps pace with `llama.cpp` on tuned AVX2 kernels — ahead reading prompts, level generating.
- **Vulkan GPU**: reads prompts _faster_ than `llama.cpp`'s Vulkan build, bound through `purego` rather than cgo.

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

One model answers one request at a time; a second request waits. `-parallel N` cuts the context into N slots, each keeping its own conversation warm — what that buys is the prompt cache, not throughput.

## ⚡ Vulkan GPU

`-vulkan` puts the whole model on the card. Measured on an RX 9070 XT, the same evening, against `llama.cpp`'s own Vulkan build on the same files:

| tokens a second | golem gen | llama gen | golem pp64 | llama pp64 | golem pp256 | llama pp256 | golem pp512 | llama pp512 |
| --------------- | --------: | --------: | ---------: | ---------: | ----------: | ----------: | ----------: | ----------: |
| Gemma 4 26B A4B |     103.7 |     124.8 |   **2061** |        833 |    **3951** |        2918 |    **4541** |        4038 |
| Gemma 4 12B     |      60.0 |      64.6 |   **1499** |        983 |    **2611** |        2471 |        2858 |        2976 |
| Qwen3 4B        |     167.0 |     169.7 |   **3950** |       2952 |    **5854** |        4545 |        5895 |        6125 |
| Qwen3 0.6B      |     359.4 |     365.4 |  **13915** |       9966 |   **23787** |       19159 |   **23995** |       22306 |

**Reading a prompt, golem is ahead on all four models at 64 and 256 positions**, and on the 26B A4B and the 0.6B at 512 as well. On the 26B A4B that is a factor of two and a half at sixty-four positions.

**Generating, it is still behind** — 0.83 of llama.cpp on the 26B A4B, 0.93 on the 12B, 0.98 on both Qwen3 models. A token is bound by reading the weights; what is left there is the order they are read in and how the dispatches are scheduled around them.

The card holds the whole model: 12.8 GiB for the 26B A4B, which is why sixteen is the smallest card that can run it, and about nine seconds of upload. `-vulkan` is all or nothing and says so: a card without `VK_KHR_shader_integer_dot_product`, a machine with no Vulkan loader, a model too large for the card — each is an error at startup rather than a silent half-move. Without the flag, everything runs on the CPU as before.

_(See [ARCHITECTURE.md](ARCHITECTURE.md) for the kernel work behind these numbers.)_

## 🧠 Supported Models

| Family | What it runs | On CPU, vs its reference |
| --- | --- | --- |
| **Gemma 4** | E2B, 12B, 26B A4B (mixture of 128 experts). Text, Vision, Audio. | Reading a prompt ×1.24 (E2B), ×1.33 (12B), ×1.06 (26B A4B). Generating, a tie: ×1.01, ×1.04, ×1.06 — vs llama.cpp |
| **Qwen3** | Dense models, from a GGUF. | 4B: ×1.13 reading, ×1.00 generating. 0.6B: ×0.99 reading, ×0.85 generating — vs llama.cpp |
| **Pocket TTS** | 13 shipped models across 6 languages, voice cloning included. | ×2.31 and ×1.69 the speed of the PyTorch reference, on the 24- and 6-layer models |

In absolute terms, on an i7-9700K with eight threads and Q4_0 weights: Gemma E2B draws 22.6 tokens a second and reads 204; the 12B, 5.0 and 42; the 26B A4B, 13.1 and 51; Qwen3 4B, 14.6 and 110. Pocket TTS speaks at ×2.94 real time in French, ×6.81 in English.

**The 0.6B is the one this engine loses**, and [`qwen/README.md`](qwen/README.md) says why: at 320 MB the weights fit close enough that the memory bus stops being the limit, and what is left is arithmetic, where llama.cpp's kernels win. This engine is built for the regime where reading the weights is the cost, and it says so where it is not.

## 👁️ Multimodal (Vision & Audio)

Provide the projector weights, and Gemma can see and hear:

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

- `cmd/golem-cli`, `cmd/golem-server`, `cmd/pocket-tts` — the three commands.
- `engine/` — reads the architecture out of a GGUF and opens the engine that implements it.
- `gemma/`, `qwen/`, `pockettts/` — standalone engine implementations; they do not import one another.
- `nn/` & `vk/` — the shared kernels: quantized AVX2 and NEON, and Vulkan compute.
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

Standing on the shoulders of giants: [llama.cpp & ggml](https://github.com/ggml-org/llama.cpp), [Kyutai Pocket TTS](https://github.com/kyutai-labs/pocket-tts), [Google Gemma](https://ai.google.dev/gemma).
