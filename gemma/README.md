# gemma — Gemma 4, in Go

Runs Google's Gemma 4 from a GGUF file: no cgo, no Python, nothing outside the
standard library. The weights stay mapped and quantized; nothing is converted
at load time.

Two checkpoints are known to run, and both are checked against llama.cpp here:
**E2B**, 35 blocks of 1536, and the **12B**, 48 blocks of 3840. They declare the
same architecture and share almost none of its numbers; the section below says
what differs, and none of it is a branch on the model's name.

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

## Sight

Gemma 4 can look at a picture, and the encoder that lets it is a second GGUF —
`mmproj-gemma-4-E2B-it-QAT-BF16.gguf`, whose `general.architecture` is `clip`.
`OpenVision` binds one to a model that is already loaded; a projector whose
output is not the model's width is refused rather than run.

```go
m, _ := gemma.Open("gemma-4-E2B-it-QAT-Q4_0.gguf", 4096)
_ = m.OpenVision("mmproj-gemma-4-E2B-it-QAT-BF16.gguf")

rows, _ := m.EncodeImage(png)              // one row per soft token
prompt, _ := m.BuildPrompt(ids, [][][]float32{rows})
hidden := m.ForwardPrompt(prompt, 0)
```

E2B's tower is sixteen blocks of 768, and four things separate it from the
language model beside it. **The image keeps its shape**: there is no fixed
input size — `clip.vision.image_size` says 224 and means nothing — so the
picture is scaled until the patch grid holds between 40 and 280 pooled tokens,
rounded to a whole number of them, and what the rounding leaves over becomes a
black border rather than a stretch. **The rotation is two-dimensional**: the
low half of each head turns with the patch's column and the high half with its
row, at base 100. **Every product is clipped** to the range it was quantized
under — the four scalars beside each weight in the file — because `gemma4v`
overrides `build_mm`, not only the final projection. And the feed forward is a
quick-GELU gate, which is the default `clip.cpp` falls to for a projector
declaring neither `use_gelu` nor `use_silu`.

Three numbers the file does not carry are written down in `visionconfig.go`
with a comment saying where they come from: the pooling kernel of 3, the RoPE
base of 100, and the token range of 40 … 280. They are what llama.cpp
hardcodes for this projector, and a file that disagreed with them would run
under llama.cpp with those values anyway.

### Two projectors, and only one of them is a tower

The 12B's projector is not the same architecture as E2B's. It declares
`clip.vision.projector_type = "gemma4uv"` — "unified" — carries eleven tensors,
and says `block_count = 0` and `head_count = 0` because it means them: there is
no tower. What the file holds is an embedder, and the blocks that would have
followed it are the language model's own, which is why its projection lands on
3840 and leaves it there.

It cuts patches of 48 rather than 16 — the merging of three by three is the
convolution's here, so nothing is pooled afterwards — normalizes the raw patch,
projects it with a bias, normalizes that, adds the same two position lookups,
normalizes again, and projects. The three norms are layer norms, mean subtracted
and bias added, where every norm in the tower is an RMS norm with a gain alone:
that part of the model is written in PyTorch's `nn.LayerNorm`, whose epsilon of
1e-5 is hardcoded in `clip.cpp` and here. And its pixels are not scaled to
-1 … 1: `gemma4v` scales them and `gemma4uv` does not.

What it costs is one product: 117 patches against a weight of 3840 by 6912,
which is three billion multiply-accumulates and a hundred megabytes read once.
Four output rows are taken at a time, by `nn.Scores4` — the kernel E2B's
attention already uses, four rows against two columns with every column loaded
once for four products. One row at a time would pull the three-megabyte grid
through the shared cache once per row, three thousand eight hundred times for
one picture; four rows at a time is a quarter of that, and it is what takes the
encoding from 76 milliseconds to 35. llama.cpp does the same work in 22.

Everything around it is shared. The image is sized by the same rule, the markers
are the same, the span is attended to in both directions the same way, and
`OpenVision`, `EncodeImage` and `BuildPrompt` do not know which of the two they
were given. `VisionConfig.Unified` is read from the file, and it is the only
place the difference is a branch.

The working memory is lent rather than made. One image's scratch is eighty
megabytes — twelve buffers of patches by width, and a feed-forward tile that is
four times as wide again — and the tower used to allocate it per picture and
throw it away, along with two small buffers per core *per block* for the
attention. Both are now taken from a reserve and given back: `sync.Pool` for
the scratch, because a server encodes several pictures at once and two
encodings sharing one buffer would corrupt each other, and a pool hands the
memory back to the collector when an idle server stops asking. A picture
allocates 22 megabytes where it allocated 103, and the tower went from 0.77
seconds to 0.72.

Reaching the language model costs two changes to it and no more. A position may
be handed the row itself instead of a token identifier — the per-layer input of
such a position is the padding token's, which is what llama.cpp uses for an
embedding batch — and **a picture is attended to in both directions**: every
token of one carries the last position of the span, and the window is measured
from there. It is a wider window, not a new mask.

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
is why the 12B runs from the same lines.

## The 12B, and what it does not have

The 12B declares `general.architecture = "gemma4"` like E2B and then disagrees
with it about nearly everything the code could have assumed:

- **No per-layer embeddings.** `embedding_length_per_layer_input` is zero, the
  file carries no `per_layer_*` tensor, and the whole branch is skipped — a
  zero read from the file, not a case on the checkpoint.
- **No sharing.** Every one of the 48 blocks computes its own keys and values;
  `attention.shared_kv_layers` is zero.
- **A different alternation.** Five window blocks of 1024 positions, then one
  global; sixteen query heads throughout, over eight key-value heads of 256 in a
  window block and *one* of 512 in a global one.
- **No value projection in a global block.** `blk.N.attn_v.weight` is simply
  absent there, and the key projection serves as both: the value is that
  projection normed without a gain of its own and never rotated, which is what
  llama.cpp does when it finds no `wv`. `Config.ValueIsKey` is read from the
  tensor's absence.
- **Two tokens it must not emit.** `tokenizer.ggml.suppress_tokens` names the
  image and audio markers, which no text decoder can turn into anything.
  llama.cpp adds minus infinity to those logits after the softcap; so does
  `Logits`. Without it the 12B answers a chat turn with one marker repeated to
  the token limit, which is the whole visible symptom of a missing line.
- **A template with one more rule.** See "The chat template" below.
- **A projector of another kind.** It sees, but not through a tower: its mmproj
  declares `gemma4uv` and holds an embedder alone. See "Two projectors, and only
  one of them is a tower" above.

Its widths are also what found a real bug in the shared kernel: 3840 and 15360
inputs, neither a multiple of the 2048-input tile the Q4_0 product cuts a wide
row into. The last stretch of such a row is shorter than a tile, and the kernel
used to be told a whole one — reading past both the row and the activation, and
returning sums with no bound at all. E2B never showed it, because 1536, 6144 and
12288 all divide. `nn.TestMatVecQ4_0PartialTiles` is that case, with both
operands padded so an overrun meets numbers rather than zeros.

## The 26B A4B, and the one thing it has

The 26B declares `gemma4` too, has no per-layer embeddings and no sharing, and
alternates five window blocks of 1024 for one global — sixteen query heads over
eight key-value heads of 256 in a window block and *two* of 512 in a global one,
where the 12B has one. Its global blocks publish no value projection either.
None of that needed a line: the geometry is read, and a number the code never
wrote down cannot be wrong about a checkpoint it has not met.

What it has that nothing else here does is a **mixture of experts**, and it is
not the usual one. The shared expert is the ordinary dense feed forward, and
the experts run *beside* it rather than after it:

    attn_out -> ffn_norm       -> dense FFN       -> post_ffw_norm_1 -\
                                                                      + -> post_ffw_norm -> + attn_out
    attn_out -> pre_ffw_norm_2 -> chosen experts  -> post_ffw_norm_2 -/

Three post-norms, not two. The sum of the branches goes through the block's own
`post_ffw_norm` — the same norm a dense block applies to its one branch —
before joining the stream. Leaving that out gives a block whose every inner
waypoint matches the reference to a part in a million and whose output does
not, which is exactly how it was found.

The router is the other place to read twice. It reads `attn_out`, the residual,
not either branch's normed input: an RMS norm with no gain, scaled by one over
the square root of the width, multiplied elementwise by `ffn_gate_inp.scale`
which stands in for that missing gain, then the router's matrix, then a softmax
over the largest eight of a hundred and twenty-eight, renormalized over those
eight alone. Routing on the branch's input instead gives a model that answers
fluently and wrongly, and nothing announces it — which is why
`ffn_moe_logits` is a recorded waypoint and not an inference from `l_out`.

Two things the file says that the reference source does not:

- **Gate and up are fused.** There is no `ffn_gate_exps` and no `ffn_up_exps`,
  one `ffn_gate_up_exps` of Q4_0 [2816, 1408, 128]: an expert's first 704 rows
  are its gate, the next 704 its up. So one product does both halves on an
  input read once.
- **A scalar per expert.** `ffn_down_exps.scale` multiplies that expert's whole
  output. llama.cpp applies it just before the routing weight, so this engine
  folds it into that weight — one multiplication instead of two thousand eight
  hundred and sixteen.

Its projector declares `gemma4v` and took no code at all: `LoadVisionConfig`
reads twenty-seven blocks of 1152 out of the file as it reads E2B's sixteen of
768. Its tower ends up twenty-eight thousandths of the range from ggml's
against E2B's eight, and that is depth rather than a missing step — every block
started from the reference's own input lands within two ten-thousandths, the
same bar E2B's blocks hold to, the pooling is exact to a part in ten million,
and the same instrument run over E2B draws the same curve eleven blocks
shorter. `TestMoEVisionBlocks` and `TestMoEVisionPoolAndProject` are that
measurement, kept so the next person does not have to take it on trust.

The one place the two projectors differ in arithmetic is the standardisation:
this one carries `v.std_bias` and `v.std_scale`, and subtracting a bias from a
value near it loses digits — a few parts in a hundred thousand, where E2B's
projection is a part in ten million. llama.cpp subtracts in float32 and loses
the same ones.

The expert stacks are the one kind of matrix `Repack` leaves alone. Only eight
of a hundred and twenty-eight are read per position, so the interleaved second
copy that pays for itself elsewhere would double the resident weights to speed
up six percent of the rows.

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

The same table for the 12B, from `testdata/gemma/layers12` and
`GOLEM_MODEL_12B`:

| what is compared | worst gap |
|---|---|
| a window block, driven by the reference's own input | 2e-3 relative |
| a global block — the kind whose keys are its values | 2e-3 relative |
| the hidden state after 48 blocks | 5.6e-2 relative |
| the top 64 logits | 0.86, on logits near 15 |
| the reference's argmax | identical |
| sixteen greedy tokens | identical |

Thirteen more blocks is what stands between the two hidden-state figures: the
per-block gaps are the same size in both checkpoints, and they compound.

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

On an i7-9700K, eight threads, Q4_0 weights. E2B first:

| | per token |
|---|---|
| `Forward` — 35 blocks, a prompt already in the cache | 35 ms |
| `Logits` — the whole vocabulary | 10 ms |
| `ForwardBatch` — 64 positions of a prompt at once | 5.0 ms |

And the vision tower, per picture rather than per token:

| | per image |
|---|---|
| `VisionTower.Encode` — a 640×426 picture, 1053 patches, 117 soft tokens | 0.78 s |
| llama.cpp's `mtmd_encode_chunk`, the same picture | 1.08 s |

**×1.38, and the comparison is if anything against this engine**: the figure
above includes resizing the picture, and llama.cpp's does not — its
preprocessing happens in `mtmd_tokenize`, before that timer starts. Both are
the best of four runs on eight cores with no accelerator.

A warning for anyone repeating this. `llama-mtmd-cli` prints two durations for
one picture, and only the first is the tower: `image slice encoded in …` is
`mtmd_encode_chunk`, and `image decoded (batch 1/1) in …` is the *language*
model reading the embeddings the tower produced. They are different
computations, and the second is the smaller one.

Where this began, before any of the work below, `Encode` took 3.2 s — three
times llama.cpp's tower rather than a third under it. The tower is 318 GFLOP
of matrix products and 54 of attention, and four fifths of what it costs now
is the one kernel that multiplies bfloat16 weights by float32 activations, at
785 GFLOP/s on the shapes the blocks are made of, against a machine peak near
1100. What got it there, in the order the measurements asked for it:

- **Whole matrices, not one patch at a time.** A thousand patches arrive
  together, so the weight is read once for all of them rather than a thousand
  times. This is the difference between a tower and a language model: one
  draws a token at a time and waits on the bus, the other is handed a grid and
  waits on the arithmetic.
- **Two rows against four columns.** One row against one column spends ten
  loads on eight multiply-accumulates, and this machine issues two of each per
  cycle: the loads finished last. Blocking both ways makes it six loads for
  eight. 260 GFLOP/s to 495.
- **An interleave instead of a shift.** A bfloat16 becomes a float32 by moving
  into the high half of a word pair, which interleaving it with zeros does on
  the shuffle port — where the shift it replaces ran on the two ports the
  multiply-accumulate itself needs. The order that comes out of the interleave
  is the order the activation is written in, which costs nothing.
- **The columns outside, the rows inside.** Four columns are twelve kilobytes
  and stay in the first cache while a core's whole share of the rows sweeps
  past them. That swap alone: 644 GFLOP/s to 785.
- **A panel that fits the second cache.** A batch wider than it streams past
  every row of weights; cut to 128 KB it is read once and answers them all.
  Measured at 64, 128 and 256.

The attention is blocked twice for the same reason: a head's keys and values
are a quarter of a megabyte each, so the queries sweep them in groups of eight
rather than one at a time, and each sweep is cut into blocks of 256 keys that
sit in a core's own cache.

And three loops that were the library's are ggml's own, eight values at a
time: the softmax's exponential — called two hundred million times for one
picture — the quick-GELU gate of the feed forward, and the clamp-and-round
that prepares every activation.

The first two are within a quarter of what the memory bus on this machine can
deliver: a token being generated reads 999 MB of block weights and 315 MB of
logit head, and 1.3 GB in 45 ms is 29 GB/s against the 38 GB/s eight cores can
sustain on a plain read. There is no arithmetic left worth removing there; what
remains is the waiting.

The third is a different regime. A batch reads the same 999 MB for all of its
positions, so the memory bus stops being the limit and the integer kernel
becomes it — which is why the figure is per token of a batch and not per pass.

### Against llama.cpp, on the same machine and the same file

Same i7-9700K, eight threads, the same `gemma-4-E2B-it-QAT-Q4_0.gguf`, llama.cpp
build 9603 held to the CPU (`-dev none -ngl 0`):

| | llama.cpp | golem | golem ÷ llama.cpp |
|---|---:|---:|---|
| prompt, 64 tokens | 165 t/s | 204 t/s | ×1.24 |
| generation, 32 tokens | 22.4 t/s | 22.6 t/s | ×1.01 |

```bash
llama-bench -m "$MODEL" -dev none -ngl 0 -t 8 -p 64 -n 32 -r 3
chat -model "$MODEL" -p "<fifty-nine tokens>" -temp 0 -n 32 -stats
```

The 12B on the same machine, the same way, `gemma-4-12B-it-QAT-Q4_0.gguf`:

| | llama.cpp | golem | golem ÷ llama.cpp |
|---|---:|---:|---|
| prompt, 64 tokens | 31.7 t/s | 42.0 t/s | ×1.33 |
| generation, 32 tokens | 4.83 t/s | 5.0 t/s | ×1.04 |

Per token of generation that is 176 ms in the blocks and 24 ms in the logit
head. Both engines are reading 6.5 GB of weights for every token here, and at
that size there is nothing between them but the memory bus, which is why
generation is a tie on both models. The prompt is not: there the bus stops
being the limit and the integer kernel becomes it, and that is arithmetic
rather than waiting.

**Generation** is where the two meet, and four things closed a gap that used to
be a factor of four and a half.

The logit head was two thirds of a token and had no kernel at all: it
dequantized each of 262144 rows into floats and multiplied those. It now runs
the product ggml runs — six-bit magnitudes kept unsigned, an activation cut in
superblocks of 256, integers all the way to one float per superblock — in
`nn/dot_q6_k_amd64.s`. 128 ms became 10, which is the tensor's size divided by
the memory bus.

When several conversations draw at once the head is read for each of their
states, and the six-bit magnitudes have to be shifted and masked out of the row
before anything can multiply them — work that belongs to the weights, not to
the activation, and that was being done once per state.
`nn/dot_q6_k_x2_amd64.s` unpacks a row once and multiplies it against two
activations: four states cost 13 ms of head instead of 16, eight cost 21
instead of 29, and the products are the same integers in the same order, which
`TestQ6KTwoColumnsAgreeWithOne` holds it to.

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

**The prompt** was a factor of eleven behind llama.cpp and is now a quarter
ahead of it. It
was eleven because every kernel here took a vector: sixty-four tokens meant
sixty-four passes over a gigabyte of weights for arithmetic that needed one.
`nn.Batch` is the answer — a set of activations quantized together and
interleaved block by block, so that a row of weights is read once and meets
every column of the batch while it is still in the first-level cache. What
`gemma.ForwardBatch` does to the model is carry a run of positions through it
together; the batch is causal within itself, which is why a block stores all of
its keys and values before any of its scores are computed.

Four things about that path are worth naming, because all four were measured
rather than assumed:

- **The tiles matter as much as the kernel.** A group of columns of a
  feed-forward output is tens of kilobytes of activation, which is past the
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
- **The width of a pass is what the row costs.** Unpacking the nibbles,
  converting the fp16 scale and folding the correction happen once per block
  whatever the width of the group, so at four columns they were a third of the
  instructions in the loop and at eight they are a seventh. `dotQ4_0x8AVX2`
  reads a row once for eight positions; a batch that is not a multiple of eight
  finishes on the four-column and one-column kernels, which sum in the same
  order, so the bit-for-bit agreement holds. That, and folding the multiply and
  the add of each block into one `VFMADD231PS`, took the prompt from 102 t/s to
  134 on E2B and from 20.7 to 27.0 on the 12B.
- **What the width could not amortize was the row's scale**, and that took a
  second layout of the weights. One conversion of a block's integer sum to a
  float, one multiply by the two scales and one add: three instructions per row,
  per column, per block, which no grouping of columns removes because the scale
  belongs to the row. Unless the rows are in the lanes of one register.
  `nn/pack_q4_0.go` interleaves eight rows so that they are — one horizontal add
  a block, and eight rows' sums come out in ascending lanes — and
  `dotPackedQ4_0x4AVX2` reads a group once for four positions. It also sums a
  block's four chunks in int16 before widening them, which is exact here (a lane
  holds at most fifteen thousand) and saves three instructions more. The prompt
  went from 134 t/s to 204 on E2B and from 27.0 to 42.0 on the 12B, which is
  where it passed llama.cpp; generation gained a little too, because fewer
  instructions still help when the bus is the limit.

The price is the one thing the engine did not do before: `Weights.Repack` builds
that layout at load time, in a third of a second on E2B and under two on the
12B, and holds
it in memory — 960 MB beside the mapping for E2B, 6.5 GB for the 12B. The
mapping's own pages are clean and the kernel can drop them under pressure, so
what the machine has to find is one copy plus what it is using. It is the same
trade ggml makes with its repacked buffer types, and it is what the prompt is
worth here.

### What the mixture costs

Eight threads, Q4_0, prompt of 64, warm page cache — the second measurement,
because the first is a measurement of the disk:

| | golem | llama.cpp `-ngl 0` |
|---|---|---|
| generating | 13.1 tokens a second | 12.4 |
| reading a prompt | 51.0 | 48.1 |

A token is 76.6 ms: 58.9 in the thirty blocks and 17.7 in the logit head, which
reads a Q4_0 matrix of 262144 rows by 2816 and is the same fifth of the cost
for either engine.

### Why the unit of work is one expert and not one position

`ExpertFFN` splits its work by *(position, expert)* — eight units per position
rather than one. That is the difference between 121 ms a token and 59, and the
reason is that decoding is a batch of one: split by position, seven cores of
eight spin while the eighth does the majority of the arithmetic in the model,
forty-seven million products a block against the dense branch's eighteen. The
profile said so plainly before the change — two thirds of all samples were in
`spinPause`.

Eight units cannot accumulate into one vector, so each writes its own and a
second section sums them. That reduction is `Dim` adds per unit against `Dim`
times `ExpertFFN` products, a part in seven hundred.

The doubling is not eightfold, and the profile says why that too: with the
cores busy, the expert path costs about twice the CPU it did on one core for
the same arithmetic. Eight cores each pull a different expert's 3.3 MB of
weights at once, and what they wait on afterwards is memory rather than each
other. A token reads about 1.7 GB — 800 MB of it experts, 415 MB the logit
head — so this is now a bandwidth number, and the next honest gain would have
to read fewer bytes rather than spread them better.

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

The two checkpoints do not carry the same template. The 12B's, asked for a
generation prompt with thinking off, closes it with a thought channel opened and
shut at once — `<|channel>thought\n<channel|>` — and E2B's has no such line.
`Config.EmptyThought` is read from the Jinja in the file, `ChatOptions` carries
it, and a 12B answered without it drifts off distribution within a token or two.

`testdata/gemma/chat/cases.json` and `chat12/cases.json` hold what Jinja itself
renders, from each template read out of its own model file, over fourteen
conversations each. The test compares character for character, and the fixture
says which of the two spellings it came from. The recorder is
`ref/gemma/dump_chats.py`.

Tools, tool calls and tool responses are the rest of that template, and
`toolrender.go` and `toolcall.go` write and read them. Content-part arrays and
media markers are not: `RenderChat` refuses a conversation that would need them
rather than rendering something close.

`gemma.Template` is how a command reaches all of this. It is the `chat.Template`
this engine implements — render, parse the calls, say where a call opens — and
it carries `Config.EmptyThought` so that a caller holding one never has to know
which of the two checkpoints it opened.

## Sampling

`Config.Sampling` is read from the file's own `general.sampling.*` keys — for
both checkpoints, temperature 1, top-k 64, top-p 0.95 — and falls back to
`sample.Defaults()` when a checkpoint declares none. Feeding it to `sample.New` gives the draw;
`gemma.Argmax` remains the greedy choice the parity tests use.
