# gemma — Gemma 4, in Go

Runs Google's Gemma 4 from a GGUF file: no cgo, no Python, nothing outside the
standard library. The weights stay mapped and quantized; nothing is converted
at load time.

## What it does

`Open` maps the file and reads its geometry. `Forward` advances one token and
returns the hidden state; `Logits` scores the vocabulary. There is no tokenizer
here yet — the engine takes and returns identifiers.

```go
m, err := gemma.Open("gemma-4-E2B-it-QAT-Q4_0.gguf", 4096)
defer m.Close()

var hidden []float32
for pos, token := range prompt {
    hidden = m.Forward(token, pos)
}
logits := make([]float32, m.Cfg.Vocab)
m.Logits(hidden, logits)
next := gemma.Argmax(logits)
```

## The model, in the two places it is unusual

**Two attention geometries alternate.** Four blocks that see the last 512
positions, rotating with a base of 10⁴ over heads of 256, then one that sees
everything, over heads of 512 with a base of 10⁶ — and that one rotates only 64
of its 256 pairs, the rest switched off by frequency factors of 10³⁰.

**Twenty of the thirty-five blocks own no keys or values.** From block 15 on,
each block computes a query and reads the cache that block 13 or block 14 left
behind. That is what makes E2B small, and it is the easiest thing in the model
to get quietly wrong: read the wrong source and the output is fluent, plausible
and not the model's.

Besides that: five norms per block, per-head query and key normalization, a
gated-GELU feed forward that widens from 6144 to 12288 at block 15, logits
softcapped at 30, and a 256-wide vector per token *per block* folded into the
residual stream on the way out of each.

None of it is written into the code. Every number is read from the file, which
is why the 12B — no per-layer embeddings, no shared keys, a different window —
runs from the same lines.

## How it is known to be right

Not against PyTorch. The weights on disk are quantized, and comparing a Q4_0
forward pass against a bf16 one puts quantization error on top of everything
else until a mistake and a rounding stop being distinguishable.

The reference is llama.cpp, instrumented: `ref/gemma/dump_layers.cpp` runs the
same model on the same tokens, on the CPU backend with flash attention off, and
writes out every intermediate its graph names. Those recordings go to
`testdata/gemma/layers`, and the tests read them; no C++ runs at test time.

They are not versioned here — activations and logits are the model's output, and
slabs of its quantized weights are the model itself. `ref/gemma/README.md` is
one command, and the tests that read them skip until it has been run. The gaps
below are what that command produced on this machine.

| what is compared | worst gap |
|---|---|
| the token embedding | 0 — identical |
| the per-layer inputs, all 35 of them | 6e-6 |
| the query, key and value of a window block, after their norms and rotations | 2e-6 |
| the same for a global block | 6e-6 |
| attention, a window block | 3e-4, on heads reaching 3 |
| attention, a global block | 2e-6 |
| attention, a block that shares | 6e-3, on heads reaching 30 |
| a whole block, driven by the reference's own input | exact at most positions, 2e-3 relative at worst |
| the hidden state after 35 blocks | 1.9e-2 relative |
| the top 64 logits | 0.29, on logits near 15 |
| the reference's argmax | identical |
| sixteen greedy tokens | fifteen identical, one tie decided the other way |

Three things had to be reproduced rather than reimplemented, each of which
costs a day if guessed:

- **GELU is a lookup table.** ggml's CPU backend rounds to fp16, indexes 65536
  precomputed results and widens the answer back. Evaluating the formula
  instead leaves a gap of 10⁻³ on the feed-forward output.
- **The attention scores are not divided by the square root of the head
  dimension.** Gemma 4 sets that scale to one, and the query norm does the work.
- **Everything ggml holds in fewer bits, this holds in fewer bits too.** The
  key-value cache is fp16, so the queries and the attention probabilities are
  rounded to fp16 before their products; the per-layer projection is bf16, so
  its input is rounded to bf16. Keeping float32 through any of them is not more
  accurate than the reference, it is between one and three parts in a thousand
  away from it — and at that size a real mistake is invisible.

### Where the remaining gap comes from, and why it stops there

The first six blocks agree with llama.cpp to within a few units in the last
place — two parts in ten million. What breaks the tie is the Q4_0 matrix-vector
product, whose sums are ordered differently here than in ggml's AVX2 kernel:
the same integers, accumulated eight lanes at a time, with the recentring by
eight taken off once per block at the end instead of weight by weight. The two
agree to about 10⁻⁷, and once every few thousand activations that difference
decides which fp16 the GELU table is indexed by, or which integer a Q8_0
activation rounds to. Each such event moves one block's output by a part in ten
thousand, and thirty-five blocks of them add up to the two percent above.

That is the price of the kernel being fast rather than being ggml: the earlier
version reproduced ggml's summation order and stayed bit-identical through
twenty-six blocks, and it spent a tenth of a token doing it. The arbiter did not
change — the reference's argmax is still identical, and so are its greedy
tokens.

## Speed

On an i7-9700K, eight threads, Q4_0 weights:

| | per token |
|---|---|
| `Forward` — 35 blocks, a prompt already in the cache | 37 ms |
| `Logits` — the whole vocabulary | 10 ms |
| `ForwardBatch` — 64 positions of a prompt at once | 9.8 ms |

The first two are within a quarter of what the memory bus on this machine can
deliver: a token being generated reads 999 MB of block weights and 315 MB of
logit head, and 1.3 GB in 47 ms is 28 GB/s against the 38 GB/s eight cores can
sustain on a plain read. There is no arithmetic left worth removing there; what
remains is the waiting.

The third is a different regime. A batch reads the same 999 MB for all of its
positions, so the memory bus stops being the limit and the integer kernel
becomes it — which is why the figure is per token of a batch and not per pass.

### Against llama.cpp, on the same machine and the same file

Same i7-9700K, eight threads, the same `gemma-4-E2B-it-QAT-Q4_0.gguf`, llama.cpp
build 9603 held to the CPU (`-dev none -ngl 0`):

| | llama.cpp | golem | |
|---|---:|---:|---|
| prompt, 64 tokens | 172 t/s | 102 t/s | ×1.7 |
| generation, 32 tokens | 22.3 t/s | 21.3 t/s | ×1.05 |

```bash
llama-bench -m "$MODEL" -dev none -ngl 0 -t 8 -p 64 -n 32 -r 3
chat -model "$MODEL" -p "<fifty-nine tokens>" -temp 0 -n 32 -stats
```

**Generation** is where the two meet, and four things closed a gap that used to
be a factor of four and a half.

The logit head was two thirds of a token and had no kernel at all: it
dequantized each of 262144 rows into floats and multiplied those. It now runs
the product ggml runs — six-bit magnitudes kept unsigned, an activation cut in
superblocks of 256, integers all the way to one float per superblock — in
`nn/dot_q6_k_amd64.s`. 128 ms became 10, which is the tensor's size divided by
the memory bus.

The Q4_0 kernel stopped recomputing, once per row of every matrix, a recentring
that depends on the activation alone: a Q4_0 weight is stored shifted by eight,
and 8 × scale × sum(q) per block is now computed when the activation is
quantized. Half the integer products in the inner loop went away with it, and
what is left accumulates in a vector instead of being reduced horizontally
every thirty-two weights.

The parallel sections stopped being goroutines. A token is three hundred
sections long, and handing each to fresh goroutines made the runtime park and
wake its threads three hundred times; `nn/parallel.go` keeps the workers and has
them spin between sections. That alone was seven milliseconds a token.

And the sections themselves got fewer and wider. The query, key and value
projections share one; so do the gate and the up projection, together with the
activation, the multiply and the quantization that follow them — five passes
over the same twelve thousand values, four of which used to run on one core
while seven waited. Attention is a head to a core, which matters little at a
hundred positions and is what matters at four thousand.

**The prompt** was a factor of eleven and is a factor of one and two thirds. It
was eleven because every kernel here took a vector: sixty-four tokens meant
sixty-four passes over a gigabyte of weights for arithmetic that needed one.
`nn.Batch` is the answer — a set of activations quantized together and
interleaved block by block, so that a row of weights is read once and meets
every column of the batch while it is still in the first-level cache. What
`gemma.ForwardBatch` does to the model is carry a run of positions through it
together; the batch is causal within itself, which is why a block stores all of
its keys and values before any of its scores are computed.

Two things about that path are worth naming, because both were measured rather
than assumed:

- **The tiles matter as much as the kernel.** Four columns of a feed-forward
  output are forty-eight kilobytes of activation, which is past the
  first-level cache, and the down projection of this model is exactly that
  shape: it ran at 97 GMAC/s against 250 for every other matrix until the input
  was cut into stretches of two thousand. The cut is decided by the width alone,
  never by the batch, and the kernels carry their lanes across it — so a batch
  of sixty-four sums its products in the same order a batch of one does, and
  `TestBatchAgreesWithOneAtATime` holds it to that, bit for bit.
- **Thirty-two positions is where the gain stops.** At sixteen the weights have
  already stopped being the limit; past sixty-four the activations no longer fit
  in the caches and every position attends to every position before it, and the
  rate falls.

What is left between golem and llama.cpp on a prompt is the shape of the weights
themselves. ggml repacks Q4_0 into an interleaved layout at load time, so that
one pass of its kernel serves eight rows with contiguous nibbles; golem reads
the file's own layout and unpacks a row for every four columns. That is the next
thing to write, and it is worth about the factor that remains.

## The chat template

The GGUF carries the template as eighteen kilobytes of Jinja. golem does not
interpret Jinja. `chat.go` writes out the path a text conversation takes through
that template, which is short: a `<bos>`, an optional system turn, one
`<|turn>{role}\n…<turn|>\n` per message with `assistant` renamed `model`, and an
empty model turn at the end when the caller wants one generated.

Two details are not guessable from the shape of the output and are worth naming:

- Two assistant messages in a row are **one** model turn. The second opens no
  header — the template folds a continued answer into the turn already open.
- A past answer goes through `StripThinking` first: the template cuts it on
  `<channel|>` and keeps, from each piece holding a `<|channel>` marker, only
  what preceded it. An unclosed channel therefore swallows everything after it.

`testdata/gemma/chat/cases.json` holds what Jinja itself renders, from the
template read out of the model file, over fourteen conversations. The test
compares character for character. The recorder is `ref/gemma/dump_chats.py`.

Tools, tool calls, tool responses, content-part arrays and media markers are the
rest of that template, and `RenderChat` refuses a conversation that would need
them rather than rendering something close.

## Sampling

`Config.Sampling` is read from the file's own `general.sampling.*` keys — for
E2B, temperature 1, top-k 64, top-p 0.95 — and falls back to `sample.Defaults()`
when a checkpoint declares none. Feeding it to `sample.New` gives the draw;
`gemma.Argmax` remains the greedy choice the parity tests use.
