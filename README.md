# golem

Inference engines in pure Go: no Python, no cgo, standard library only.
One `go build`, one binary.

A golem is inert matter given a voice. That is what these engines do to a file
of weights.

## The engines

| | what it runs | measured |
|---|---|---|
| [`gemma/`](gemma/) | Gemma 4 E2B and 12B, from a GGUF | E2B: 22.6 tokens/s generated, 204 read — llama.cpp on the same CPU: 22.4 and 165. 12B: 5.0 and 42, against 4.83 and 31.7 |
| [`pockettts/`](pockettts/) | [Kyutai Pocket TTS](https://github.com/kyutai-labs/pocket-tts), twelve languages, voice cloning included | ×2.82 real time on the 24-layer French model, ×6.55 on the 6-layer English one — PyTorch: ×1.27 and ×4.03 |

Both figures are from an i7-9700K with eight threads, on Q4_0 weights. Each
engine's README says what it is measured against, where it still differs from
its reference, and by how much.

Each engine is self-contained. They do not import one another, and nothing in
the shared layer knows they exist.

## The shared layer

| package | holds |
|---|---|
| `tensors/` | safetensors and GGUF: metadata, and views on the bytes |
| `nn/` | quantized matrix products with AVX2 kernels, norms, activations, RoPE, convolutions, and the worker pool they are spread over |
| `token/` | tokenizers, one package per family |
| `audio/` | sound formats: reading and writing WAV |
| `sample/` | top-k, top-p, temperature, and a seeded draw over a row of logits |

Nothing is promoted into this layer on the strength of a guess. Code moves here
once two engines are shown to want it, in the same commit that makes them both
use it.

## The commands

| | what it does |
|---|---|
| [`cmd/chat`](cmd/chat/) | a conversation with Gemma, streamed to the terminal |
| [`cmd/serve`](cmd/serve/) | an OpenAI-compatible API over the same weights, tool calls included |
| `cmd/pocket-tts` | text in, a WAV file out |

```bash
go build ./cmd/chat
GOLEM_MODEL=gemma-4-E2B-it-QAT-Q4_0.gguf ./chat
```

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

## What is here, and what is not

No weights, no vocabularies, no voices. They belong to the people who published
them, and they are between three hundred megabytes and three gigabytes each.

- **Gemma 4** weights: a GGUF from Hugging Face, whatever quantization you like
  — the engine reads Q4_0, Q6_K, bf16 and float32, and the tests want the
  QAT Q4_0 build. Point `GOLEM_MODEL` at the file, and `GOLEM_MODEL_12B` at a
  12B one if you have both: the two checkpoints are tested separately, because
  they agree on the architecture's name and on very little else. Google's Gemma
  terms apply to them.
- **Pocket TTS** weights, tokenizer and voice states: Kyutai publishes them on
  Hugging Face, and `pockettts/README.md` says which repositories and where the
  engine looks. A voice can also be cloned from any recording, which needs
  nothing but the weights.

Neither are the recordings of the models themselves. `testdata/gemma/layers`,
`layers12`, `quants` and `window` hold activations, logits and slabs of
quantized weights as ggml computed them; `testdata/layer0`, `pipeline` and `pipeline_en` hold what
PyTorch computes inside Pocket TTS. Those are the models' output, not this
repository's, and each is one command away — `ref/gemma/README.md` and
`pockettts/README.md` have them.

What *is* versioned is what the models did not write: tokenizations and chat
templates, which are text and integers. Tests that need a file which is not
there skip rather than fail, so a fresh clone with no model at all still runs 73
of them green — and a machine with both Gemma checkpoints, the voices and one
run of the recorders runs 123.

## Building

```bash
go build ./...
go test ./...
```

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
- **The kernels.** `nn/*.s` is where the time goes. The prompt path is still a
  factor of one and two thirds behind ggml, and `gemma/README.md` says why.
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
