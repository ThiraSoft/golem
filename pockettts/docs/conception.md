# pocket-tts-go — choices, implementation, results

*19 August 2026. This document tells what was decided and why. The README says
how to use it; this one says how it is built, and above all what could have been
done differently.*

---

## 1. The starting point

A spike had answered the one question that could have killed the project: can Go
run a transformer fast enough, without SIMD intrinsics?

The answer rested on a single observation. In autoregressive generation at batch
one, a transformer does not do large matrix products: it does a long series of
**matrix-vector** products. Every frame therefore re-reads the whole of the
weights — 604 MB for the 24 layers of the `flow_lm`. The limiting factor is not
compute power, it is memory bandwidth. The Go kernel reached 93 % of the
machine's hardware ceiling, and assembly would have brought nothing: the
computation was already entirely hidden behind the memory accesses.

What remained was to write the engine. What follows is that work.

---

## 2. The structuring decisions

### 2.1 Not porting the audio encoder

**The question.** A voice, in Pocket TTS, is obtained by encoding a reference
sound excerpt with the Mimi encoder, then making the transformer listen to the
result. The state it is left in *is* the voice. Porting that encoder required the
encoding SEANet, an extra transformer, and the downsampling — about half the
model.

**What was chosen.** Not to port it. The reference Python daemon already caches
that state on disk, in safetensors format: 66 MB holding the K/V caches of the 24
layers and their offsets. The Go engine reads that file and starts with the voice
already in memory; such a state goes under `testdata/voices/`, and none is
versioned here — they are Kyutai's to publish.

**Why.** Encoding a voice happens once per voice; synthesizing happens on every
sentence. Devoting half the porting effort to the first was disproportionate. The
format was already there, already readable by the existing safetensors code, and
the import verifies itself — if the caches were laid out wrong, the transformer's
output would diverge from the first position.

**The cost.** Adding a voice requires a trip through Python. It is a real
dependency, accepted, and documented.

**What it changed.** It is the decision that made the project feasible in one
session rather than several.

### 2.2 Keeping the weights in bfloat16

Converting them to float32 once and for all at load time looks simpler: the
kernel would have no conversion left to do. But that doubles the amount re-read
on every frame — 15.8 GB/s would be needed against 16.5 GB/s available on the
machine. The margin disappears.

The weights therefore stay the bytes of the memory-mapped file, and the
conversion happens in the kernel: a bfloat16 is a float32 stripped of its low
mantissa, so the shift is enough, with no loss and no rounding.

This rule has a pleasant consequence for the tests: PyTorch, which loads the same
bfloat16 values and converts them by the same shift, starts from exactly the same
values. The measured gaps are therefore pure accumulation noise, with no
quantization component.

**The exception.** The audio decoder converts its weights to float32 at load
time. They are small — ten million parameters against three hundred million — and
re-read sixteen times per frame rather than once: the trade-off flips.

### 2.3 Writing for one position, then generalizing to blocks

**The first choice.** The transformer layer was written for *one timestep*, not
for a `[B, T, D]` tensor that would then have to be sliced. At batch one, there
is never a tensor of rank higher than two to handle, and the indices stay
readable. The prompt fill was obtained by calling the same function as many times
as there were tokens, with exactly the same arithmetic — the KV cache making the
two paths equivalent by construction.

**What had to change.** That simplicity was expensive wherever several positions
are available at once. See §4.3: it became the main optimization of the project.
The layer now processes a block of positions, and the single-position case is
just a block of size one.

**The lesson.** Writing the simplest version that is correct first, then
generalizing it when measurement demands it, worked: the generalization was
mechanical because the simple version was correct and tested.

### 2.4 Treating randomness as an input

The flow net starts from Gaussian noise. The audio output is therefore
reproducible neither in Go, nor in Python, nor between the two. Parity could not
be checked on the final audio.

The rule adopted: **the noise is an input, not a side effect.** The Python
harness writes the noise it drew, the Go test reads it back, and both integrate
the same flow. In production, the Go draw uses `math/rand/v2` with an optional
seed.

Without that discipline there would be no end-to-end test at all — only component
tests and the hope that the assembly is correct.

### 2.5 Validating through intermediate activations

**The rule.** No layer is deemed correct until its intermediate activations match
PyTorch.

Comparing only the final output would say that there is an error, never where it
is. The scripts in `ref/` therefore write, for each module, the input, the
output, and the internal waypoints: after the normalization, after the Q/K/V
projection, after the rotation, after the attention, after the feed-forward.

**The result.** Of five stages written in the session, four came out green on the
first try. The only failure — the tokenizer — was located immediately because the
test printed the obtained and expected pieces side by side: the tab and the
newline were being treated as spaces, whereas NFKC does not touch them.

**The format.** Raw float32 files plus a JSON of shapes. No `.npy`, no protobuf:
the Go tests need no library to read them back. They are not versioned with the
code — 252 KB would fit, but they are the model's output, and one command
rewrites them.

### 2.6 Not porting the NFKC normalization

SentencePiece ships a precompiled character map that applies NFKC. Porting it
meant several hundred lines — a serialized normalization trie — for no effect on
already well-formed text, that is, everything coming out of a keyboard or a
modern file.

Only the rules that genuinely change the segmentation were written: unification
of Unicode spaces, collapsing of repeats, prefix, and escaping. Twenty lines.

**The limitation.** A text containing ligatures or compatibility characters (ﬁ,
①, ½) would be segmented differently than under Python. The test corpus covers
eighteen French sentences — elisions, accents, quotation marks, numbers, emoji,
URLs, repeated spaces — and deliberately excludes them. The limitation is
documented rather than hidden.

### 2.7 Splitting into packages along responsibility boundaries

`nn` does not know the model. `transformer` does not know which model uses it.
`flowlm` does not know about audio. `mimi` does not know about text.

That discipline paid off the moment the audio decoder needed the same causal
layer as the `flow_lm`, with two differences: a per-layer scale factor, and a
limited attention window. The layer moved up into its own package, parameterized
by a `Geometry`, and the two models share it. None of the existing parity tests
moved.

A `reference` package serves the fixtures to the tests of the four packages that
share them. It is not a `_test.go` file precisely because several packages depend
on it.

---

## 3. The implementation, stage by stage

### 3.1 Reading the weights — `internal/tensors`

A safetensors file is a JSON header describing each tensor, followed by the raw
data. The file is memory-mapped (`mmap`) rather than read: the 672 MB are never
copied, and the kernel only loads the pages actually touched. That is what gives
a startup of a few tens of milliseconds, against three and a half seconds for
the Python daemon on the same machine.

### 3.2 The transformer — `internal/transformer`

One layer: LayerNorm, projection to concatenated Q/K/V, RoPE rotation, KV cache,
causal attention optionally windowed, output projection, then LayerNorm, ×4
expansion, exact GELU, contraction. No bias on the projections.

Two details required attention:

- **The exact GELU.** `torch.nn.functional.gelu` uses by default the variant
  based on the error function, not the hyperbolic-tangent approximation. The
  latter would drift from the reference by the third decimal.
- **The attention window.** The `flow_lm` sees all of its past; the audio decoder
  is limited to 250 positions. The same code serves both, with `Context = 0`
  meaning "no limit".

### 3.3 The language model — `internal/flowlm`

The 24 layers, the projection from the latent to `d_model`, the output
normalization, the end-of-utterance head, and the text embedding table.

The voice state loads from a safetensors file whose layout — `[2, 1, positions,
heads, dimension]` — matches the Go cache exactly, up to the slicing. An explicit
check refuses NaNs in the useful positions: on the Python side the unwritten
positions carry them, and if they leaked into the useful part, attention would
produce silence without saying so.

**The flow net.** The transformer does not predict the frame's latent, it
predicts the direction leading from Gaussian noise to that latent. Six residual
blocks of width 512 give this velocity field; integrating it over one step is
enough — that is the point of the distillation the model was trained with.

The conditioning does not enter through the input but through the
normalizations: each block derives a shift, a scale and a gate from it. Hence the
profusion of small matrices and biases, where the transformer has none.

A trap was hiding there: the model's `RMSNorm` divides by an **unbiased**
variance — divided by n−1 — where LayerNorm divides by n. A factor that would go
unnoticed by eye and throw everything off.

### 3.4 The tokenizer — `internal/sentencepiece`

The `tokenizer.model` file is a protobuf message. The format can be read without
a library: every field is preceded by a varint giving its number and its
encoding. A hundred lines are enough, and the engine stays dependency-free.

The segmentation is a unigram model: the vocabulary gives each piece a
log-probability, and the chosen segmentation maximizes the sum of the scores. It
is a shortest path through the lattice of possible segmentations, which dynamic
programming solves in a single sweep. A character never seen before stays
traversable — otherwise a single exotic symbol would make the whole sentence
unencodable — and is then re-emitted as bytes, each by its fallback piece.

### 3.5 The audio decoder — `internal/mimi`

Every frame brings a latent of 32 values and must yield 1920 samples at 24 kHz.
The path climbs in stages: a projection brings the latent up to 512 channels, a
transposed convolution spreads it over sixteen internal steps, a two-layer
transformer puts them in context, then the SEANet decoder climbs back to the
audio rate through three successive expansions, ×6, ×5 and ×4.

Everything is in streaming mode: each convolution keeps the context the next
frame will ask for — the last input samples for a direct convolution, the
not-yet-complete output tail for a transposed one. That is what allows the sound
to be produced as generation goes, and it is the point where an error is heard
rather than seen: a break at the seam between frames, invisible on the first one.

The bias of a transposed convolution must count only once: it is subtracted from
the tail that is set aside.

### 3.6 The assembly — `pockettts.go`

The text is prepared (initial capital, final punctuation) then segmented under
fifty tokens, at the boundaries the voice would respect anyway: strong
punctuation first, commas next. Beyond that, the model starts skipping words.

Each segment restarts from the untouched voice. Not out of caution: the model was
not trained to chain segments, and making them share a context degrades the
diction. It is also what the Python daemon does.

---

## 4. The results

### 4.1 Parity

| stage | maximum gap with PyTorch |
|---|---|
| one transformer layer | 2×10⁻⁷ |
| the 24 layers, on the real voice state | 2×10⁻⁷ |
| flow net | 4×10⁻⁶ |
| audio decoder, eight chained frames | 5×10⁻⁷ |
| tokenizer | identical segmentation on 18 sentences |
| end to end | 5×10⁻⁶ (frame 1) → 3×10⁻⁴ (frame 8) |

The end-to-end drift is not a divergence of computation. Generation is a loop
that feeds back its own output: the rounding gap does not repeat, it accumulates.
The stages, tested in isolation, stay at 5×10⁻⁷.

Listening confirms it: the voice is the expected one.

### 4.2 Throughput

The figures below were taken while the engine was being written, on the sentence
and voice of the day. What is measured now, and reproducibly, is in the README:
a benchmark on both sides — `go test ./pockettts -bench Synthesis` and
`ref/bench_python.py` — which says ×1.54 against PyTorch's ×1.27 on the
24-layer French model, and ×2.97 against ×4.03 on the 6-layer English one.

Breakdown per frame at the time — the budget is 80 ms:

| stage | time | floor |
|---|---:|---:|
| `flow_lm`, 24 layers | 32 ms | 16 ms (604 MB to re-read at 38 GB/s) |
| flow net | 1 ms | |
| Mimi decoder | 26 ms | ~15 ms (162 MMAC) |
| **wall, pipelining included** | **36 ms** | |

The two big items run on separate goroutines and partially overlap; they also
compete for the eight cores, so that each is a little slower than in isolation —
the `flow_lm` alone does 27 ms, the decoder 21 ms.

### 4.3 What the optimization taught

The engine started at **×0.49**, reached ×0.88 through structure, then more than
doubled again through the instruction set.

**Processing positions in blocks.** The audio transformer receives sixteen
positions together. Processing them one by one made it re-read its twenty-five
megabytes of weights sixteen times per frame — 403 MB, more than the whole
`flow_lm`, for a tenth of the computation. The text prompt benefits from the same
treatment.

**Pipelining.** Generating a latent and decoding it are independent from one
frame to the next. Decoding runs on its own goroutine, and the two complement
each other — the transformer mostly waits on memory, the decoder mostly on
compute.

**Vectorizing.** This is the change that carried everything, and it first
required undoing a measurement error. The engine was believed to be glued to a
memory ceiling of 16 GB/s; that figure was in fact the throughput of the scalar
kernel, not that of the machine. Two measurements showed it: a single core
already reached 15.5 GB/s, and eight gave only 16.5 — parallelism that brings
nothing is the sign that something other than memory is the limit. The real
ceiling, measured since with a vectorized kernel on a matrix that fits in no
cache, is **38 GB/s**.

The matrix-vector kernel therefore moved to AVX2: `VPMOVZXWD` widens eight
bfloat16 weights at once, `VPSLLD` does the conversion — a bfloat16 is a
truncated float32, the shift is enough — and `VFMADD231PS` accumulates, over four
accumulators to cover the FMA latency. The matrix-vector product goes from 16.5
to 125 GB/s in cache, and synthesis from ×0.89 to ×1.47 in one step.

A `PREFETCHT0` two kilobytes ahead returned another 9 % on the `flow_lm`. Its
matrices are two to eight megabytes spread over eight cores: the hardware
prefetcher does not have time to get up to speed before the row ends.

**The same lesson, one layer down: the inner loop of the convolutions.** Once the
`flow_lm` was fast, the Mimi decoder became the dominant item, and the transposed
convolution alone 69 % of its time. Its inner loop wrote one value every `Stride`
floats — unusable by the vector unit. The polyphase form, written this time in
**separate phases** in a contiguous buffer, brings it back to an ordinary
`dst += a·src`: output position o only receives the weights congruent to o modulo
the stride, and within one phase the contribution of input t falls at
t + k/Stride. The final interleave costs only one pass over the output, against
one per coefficient before. With the same `Axpy` kernel in AVX2, the decoder's
frame goes from 31 to 21 ms.

One detail there was worth a third of the gain: `k % Stride` and `k / Stride` in
the inner loop. On a channel a few positions long — that is the regime of
streaming decoding, where the early stages have many channels and few samples —
two integer divisions cost more than the multiply-accumulate they serve. Two
incremented counters replace them.

**What brought nothing.** Before the vectorization: three rewrites of the inner
loop — accumulation in a register rather than in memory, decompressing the
weights outside the loop — for zero or negative gain; a parallelization
threshold; the removal of a four-hundred-megabyte copy per segment. All these
changes are correct, none moved the number: they were optimizing a scalar loop
whose problem was not its shape, but the instruction set.

The lesson fits in two sentences. **Measure stage by stage before optimizing**:
three timers around the three stages said, on every round, which of the two big
items had become the bottleneck. And **never take a ceiling for granted without
having measured it separately**: this engine's "memory ceiling" was a scalar
compute ceiling, and believing it cost three useless optimizations before the
only one that mattered.

---

## 5. What is left

- **Throughput.** The SEANet decoder is three times above its compute floor, the
  `flow_lm` nine milliseconds above its own. That is where the margin lies to get
  past the Python daemon.
- **The other languages.** Done, but only two of the twelve are checked against
  PyTorch — `french_24l` and `english_2026-01`. The claim that a geometry table
  would suffice was wrong: two of the four differences are behavioral, not
  geometric. See the README.
- **Publishing as a library.** The API is in place and the internals are under
  `internal/`. All that is missing is a real import path in `go.mod`.
- **Voice cloning**, should the dependency on Python become a nuisance. It is a
  big chunk: the SEANet encoder, its transformer, the downsampling.
