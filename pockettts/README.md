# pocket-tts-go

An inference engine for [Kyutai Pocket TTS](https://github.com/kyutai-labs/pocket-tts)
in pure Go: no Python, no cgo, standard library only.
One `go build`, one binary, one WAV.

```bash
go build ./cmd/pocket-tts
./pocket-tts -voice testdata/voices/french_24l/<voice>.safetensors -o hello.wav "Bonjour le monde."
```

Or as a library:

```go
engine, _ := pockettts.Open(pockettts.Options{
    Weights:   ".../model.safetensors",
    Tokenizer: ".../tokenizer.model",
    Language:  "french_24l",
})
defer engine.Close()

voice, _ := engine.LoadVoice("testdata/voices/french_24l/<voice>.safetensors")
sound, _ := engine.Synthesize("Bonjour le monde.", voice, nil)
```

`sound` is a `[]float32` at 24 kHz.

`Synthesize` takes a `*Settings`, or nil for the model's own values. Build one
from `DefaultSettings(lang)` and change what you mean to change: the fields are
not interpreted, so a zero is a zero — an `EndThreshold` of 0 is a real setting,
one that makes the model far more reluctant to declare itself done.

`Settings.Frame` receives each frame as soon as it is ready, so the sound can be
played during generation. `Settings.Ctx`, when set, stops the generation as soon
as it is cancelled — at the next frame — and `Synthesize` then returns the
context's error.

The weights, the tokenizer and the predefined voices can be found where the
upstream daemon downloads them: `Locate(lang.WeightsPath())`,
`Locate(lang.TokenizerPath())` and `Locate(lang.EmbeddingPath(name))` each
return a path in the Hugging Face cache, or the empty string. `LocateVoices(lang)`
lists the voices that are actually there.

## Status

Two models, two answers. On an i7-9700K with eight threads, one sentence, the
voice `alba`, and the same end-of-speech threshold on both sides:

| | golem | pocket_tts, PyTorch |
|---|---:|---:|
| `french_24l` — 24 transformer layers | **×2.37** real time | ×1.27 |
| `english_2026-01` — 6 layers | **×5.87** | ×4.03 |
| first sound, `french_24l` | **210 ms** | 320 ms |
| first sound, `english_2026-01` | **94 ms** | 90 ms |
| ready to speak, warm | **30 ms** | 3.5 s |

```bash
go test ./pockettts -run xxx -bench 'Synthesis|FirstFrame' -benchtime 8x
python ref/bench_python.py french_24l alba 5
```

Both sides load the model and prepare the voice before the clock starts, and
both throw the first synthesis away. What is timed is what a daemon does all
day.

Both columns used to read the other way round on the six-layer model, and what
turned them is under *Throughput* below: the audio decoder was spending its time
entering kernels rather than computing in them, and one frame of it went from
17.8 ms to 6.3 ms. Where the transformer dominates — twenty-four layers,
memory-bound matrix-vector products — golem was already ahead and is now twice
PyTorch. Where it does not, the decoder was the whole difference.

What golem brings either way: a binary of a few megabytes instead of a gigabyte
of environment, thirty milliseconds to first readiness instead of three and a
half seconds, and a distribution that fits in a `go install`.

## The method

**No layer is deemed correct until its intermediate activations match PyTorch.**

The scripts in `ref/` load the real weights, inject a deterministic input, and
write every intermediate quantity into `testdata/`. The Go tests read these
files back, and no longer need Python.

Those recordings are not versioned here: they are what the model computes, and
the model is Kyutai's. The command that writes them is at the bottom of this
file, and every test that reads them skips until it has been run — the gaps in
the table below are what it produced on this machine.

| stage | maximum gap with PyTorch |
|---|---|
| one transformer layer | 2×10⁻⁷ |
| the 24 layers, on the real voice state | 2×10⁻⁷ |
| flow net | 4×10⁻⁶ |
| audio decoder, eight chained frames | 5×10⁻⁷ |
| tokenizer | identical segmentation on 18 sentences, in both languages |
| end to end, from text to sound | 5×10⁻⁶ on the first frame, 3×10⁻⁴ on the eighth |

Every stage above is checked on both `french_24l` and `english_2026-01`.

The end-to-end drift is not a divergence of computation: generation is a loop
that feeds back its own output, and the rounding gap accumulates there instead of
repeating.

Randomness is treated as an input: the flow net starts from Gaussian noise, so
the output is reproducible neither in Go nor in Python. The Python harness writes
the noise it drew, the Go test reads it back, and both integrate the same flow.

## What the engine does not do

**It does not clone voices.** Cloning requires encoding a reference WAV with the
Mimi encoder — half the model, for work that happens only once per voice. The
Python daemon already does it and caches the result on disk, in safetensors
format: the K/V caches of the 24 layers, exactly as the transformer left them
after listening to the reference. The Go engine reads that file and starts with
the voice already in memory.

This choice halves the work and costs one dependency: adding a voice requires a
trip through Python. Adding a voice is rare, synthesizing is constant.

No voice ships with this repository — the voice states published with Pocket TTS
belong to Kyutai, and redistributing them is not this project's call to make.
Put one under `testdata/voices/<language>/`, or encode your own with the Python
daemon; every test that needs a voice skips when there is none.

Also out of scope: training, batch greater than one, quantization, the GPU.

## Languages

The twelve models Kyutai ships are all supported. Pass `-language`; the default
is `french_24l`.

```bash
./pocket-tts -language english_2026-01 -voice voice.safetensors "Hello world."
```

Four things actually differ between them, and only four:

| | values |
|---|---|
| depth of the flow_lm transformer | 24 layers or 6 |
| `;` folded into `,` before synthesis | french_24l, german, german_24l |
| short inputs padded with spaces | english_2026-01 only |
| frames generated after the end is detected | 8 |

Two config keys look like they should matter and do not.
`insert_bos_before_voice` is applied when the voice state is built, on the
Python side, so it is already baked into the file this engine reads.
`mimi.inner_dim` sizes the encoder's downsampling and the speaker projection,
neither of which is on the decoding path.

Parity is checked against PyTorch on two languages — `french_24l` at 24 layers
and `english_2026-01` at 6, which is also the one language needing the padding.
The other ten share their geometry and rules with one of those two, but have not
been run.

## Throughput

In autoregressive generation at batch one, a transformer does not do large matrix
products: it does a long series of **matrix-vector** products. Every frame
therefore re-reads the whole of the weights. The limiting factor is not compute
power, it is **memory bandwidth** — and on that ground Go is not at a
disadvantage, which is what makes the project possible without a line of
assembly.

Measured breakdown, per frame (the budget is 80 ms):

| stage | time | what it cannot go below |
|---|---:|---:|
| flow_lm and flow net, 24 layers, alone | 19 ms | 16 ms — 604 MB of weights, at the 38 GB/s eight cores sustain |
| Mimi decoder, alone | 6.3 ms | ~1 ms — 162 MMAC over eight AVX2 cores |
| one frame of `french_24l`, the two pipelined | 34 ms | |

The two floors are floors for something having the machine to itself, which in
the pipeline neither has: the isolated times add up to 25 ms and a pipelined
frame costs 34, and the nine milliseconds between them are the two halves
sharing eight cores. The transformer is close to its floor and there is little
left in it. The decoder is six times above its own, and that is now a matter of
how much of its arithmetic is spent in kernels rather than around them.

```bash
go test ./pockettts/internal/mimi -run xxx -bench Frame -benchtime 40x
go test ./pockettts/internal/flowlm -run xxx -bench AdvanceLatent -benchtime 20x
```

Two things mattered more than all the micro-optimizations put together:

**Processing positions in blocks wherever they are available.** The audio
transformer receives sixteen positions together; processing them one by one made
it re-read its twenty-five megabytes of weights sixteen times per frame — more
than the whole flow_lm, for a tenth of the computation. The sweet spot is low:
with bfloat16 weights, a batch of size L gives L multiply-accumulates per two
bytes read, and the balance between memory and compute falls around L≈4. The text
prompt benefits from the same treatment.

**Pipelining.** Generating a latent and decoding it are independent from one
frame to the next. Decoding therefore runs on its own goroutine, and the two
complement each other — the transformer mostly waits on memory, the decoder
mostly on compute.

**Making the inner loop long enough to be worth entering.** The audio decoder
was seventeen milliseconds a frame for a hundred and sixty-two million
multiply-accumulates, which is more than an order of magnitude above what the
arithmetic costs, and the reason had nothing to do with the arithmetic. Its convolutions accumulate one scalar coefficient along one output
row at a time, which is the right shape when the row is a thousand samples long
and the wrong one at the top of the decoder, where the row is sixteen: the
512-channel input convolution was making 1.8 million kernel calls per frame to
multiply-add sixteen values each. Gathering each output position's window into
one contiguous vector turns the same layer into 57 000 dot products of length
512 — the same arithmetic, a thirtieth of the calls. `nn/conv.go` chooses
between the two forms per layer, on the length of the row and on whether the
gathered window still fits in cache. The two transposed convolutions at the top
of the decoder take the same treatment, a transposed convolution of stride S
being S ordinary ones interleaved. Attention in the audio transformer had the
same disease in a different place — a Go loop where a kernel belonged, and one
barrier per position where sixteen positions of eight heads make one section.

What is left to gain: the decoder is six times above its compute floor rather
than eighteen, which is now the same kind of gap as everything else here rather
than an outlier, and the flow_lm remains the memory-bound half — three
milliseconds above a floor set by the bus, and nothing to do about that one.

## One constraint not to lose sight of

The weights must **stay in bfloat16 in memory**, with the conversion happening in
the kernel. Converting them to float32 once and for all looks simpler but doubles
the amount re-read on every frame: 15.8 GB/s would be needed against 16.5 GB/s
available. The margin disappears.

## Layout

| path | role |
|---|---|
| `pockettts.go` | the API: open, load a voice, synthesize |
| `cmd/pocket-tts` | the command line |
| `internal/transformer` | the causal layer, shared by both models |
| `internal/flowlm` | the language model, the flow net, the voices |
| `internal/mimi` | the audio decoder |
| `internal/text` | text preparation |
| `internal/reference` | where the fixtures and the weights are found, for the tests |
| `languages.go` | what differs between the shipped models |
| `bench_test.go` | what a synthesis costs, end to end |
| `../tensors`, `../nn`, `../token/sentencepiece`, `../audio` | the shared layer: safetensors and `mmap`, kernels, the unigram tokenizer, RIFF writing |
| `testdata/voices` | where a voice state goes, by language; none ship here |
| `ref/` | the Python scripts that write the fixtures, and the one that times the reference |

## Reproducing the fixtures

From an environment where `pocket_tts` is installed:

```bash
python ref/dump_layer.py    <model.safetensors> testdata/layer0
python ref/dump_pipeline.py testdata/voices/french_24l/<voice>.safetensors testdata/pipeline
python ref/dump_pipeline.py testdata/voices/english_2026-01/<voice>.safetensors \
    testdata/pipeline_en english_2026-01 "Hello world."
python ref/dump_tokens.py   <french_24l/tokenizer.model>  testdata/tokenizer/cases.json    french
python ref/dump_tokens.py   <tokenizer.model>             testdata/tokenizer/cases_en.json english
```

The same environment runs `ref/bench_python.py`, which is what the table at the
top of this file compares golem against.

Nothing above is versioned here. The voice states go under
`testdata/voices/<language>/`, and the predefined ones can be fetched from
`kyutai/pocket-tts-without-voice-cloning`, which serves them already encoded.
The weights and the tokenizer are 672 MB and have no place in a git history
either — so every test that needs one of them skips cleanly when it cannot find
it; `POCKET_TTS_WEIGHTS`, `POCKET_TTS_TOKENIZER` and
`POCKET_TTS_VOICE` say where to look.
