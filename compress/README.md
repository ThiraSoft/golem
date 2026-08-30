# compress

A checkpoint in, a `.golem` out. Two codebooks, one scheme around them:

| | block | bits/weight | tensor type |
|---|---|---|---|
| `D4G` | 64 weights in 26 bytes | 3.26 | 1000 |
| `T4G` | 128 weights in 67 bytes | 4.19 | 1003 |

The file is a GGUF. Same container, so the vocabulary, the rope base and the
chat template travel unchanged; what is new is a tensor type llama.cpp does not
know and one vector per calibration site beside the matrices.

Everything below about the scheme — the salience, the bound of 24×, the rotation
by 128, `nn.D4GVectorNames` and the `A·(q ⊙ W)` convention with its two rules —
is shared by both, unchanged, and is where most of the format's value is. Only
the codebook differs.

## What a block holds

A row is two planes, the steps then the codes, because a 26-byte block would
otherwise start on a two-byte boundary and a shader reads words.

Per block of 64 weights:

- **two step codes**, one for each 32 weights. Eight bits each, naming powers of
  two spaced a sixteenth apart.
- **sixteen 12-bit codes**, each naming a point of the D4 lattice — the integer
  points of even coordinate sum — in canonical order by squared norm then
  lexicographically. Everything of norm ≤ 40 fits, which is 3961 points, and
  the first 135 of norm 42 fill the code out to exactly 4096.

There is a wide tier, `D4G16`, at sixteen bits a code. Its shell reaches norm
162 and its table is a quarter of a mebibyte rather than sixteen kibibytes,
which no longer fits a workgroup's shared memory.

`golem.d4.hadamard_group`, `golem.d4.code_bits` and `golem.d4.scale_block` in
the metadata say what the file was written with. `general.file_type` is 1000.

## The other codebook: a trellis, with nothing to look up

`-codec trellis` writes `T4G` instead. 128 weights in 67 bytes, 4.1875 bits
each, and no decode table at all.

D4 was chosen because its table fits a workgroup: 3961 points at r²=40 is
thirty-one kibibytes of shared memory. That choice bought a lookup and paid for
it with dimension four, and it closed the four-bit tier, whose table is 493 KiB
— which is where a file meaning to reach Q4_K_M's quality has to live.

A trellis has no table. The state is the last twelve bits of the code stream, so
weight *t* reads the twelve bits at offset 4·*t* and hashes them:

    weight t  =  step[t/64] · 1MAD(the twelve bits at 4t within the sequence)

    1MAD(s):  x   = 34038481·s + 76625530   (mod 2³²)
              sum = the four bytes of x, added
              v   = (sum − 510) · 0.00676589971

The effective dimension is the whole 128-weight sequence rather than four; every
weight decodes independently of its neighbours, so no product changes shape; and
the codebook is four instructions, the same for every tensor of every model. The
structure is QTIP's bitshift trellis (Tseng et al., NeurIPS 2024,
arXiv:2406.11235).

Per sequence of 128 weights:

- **two step codes**, one for each 64 weights. Eight bits each.
- **520 bits of path**: twelve for the first weight, which needs a whole window
  before there is any history to shift, and four for each of the 127 after it.
  Exactly 65 bytes.

      path    520 bits / 128 weights = 4.0625 bpw
      steps     8 bits /  64 weights = 0.125
                                       ------
                                       4.1875

A row is two planes, steps then codes, for D4G's reason. 128 is not free choice:
the Viterbi's backpointers have to sit in a workgroup's shared memory beside the
two cost planes, and 32 KiB + 16 KiB is what fits in the 64 KiB this card gives.
In device memory they would be 128 bytes a weight written and read back against
half a byte of output.

`mad1Scale` is written as a multiply and not a division on purpose. Twenty of
1MAD's 1021 values differ by one unit in the last place between the two forms,
and a Viterbi is a chain of comparisons over a codebook with 1021 distinct
values for 4096 states — near ties are everywhere and one bit moves whole paths.
It was 7 % of the weights.

### The step, and where its window sits

Eight bits naming powers of two a sixteenth apart, which is D4G's spacing and
for D4G's reason. The window is not D4G's: it runs from 6.1e-5 to 3.83 rather
than from 7.6e-6 to 0.48, because a lattice step is a *fraction* of its block's
RMS — the block is scaled up into a shell several units across — and a trellis
step is the block's RMS itself. Getting that wrong is not subtle in hindsight
and is invisible in advance: a unit-variance source clipped at D4G's ceiling and
reconstructed at **5.5 dB instead of 23**, with every shape agreeing.

An eight-bit step against an fp16 scale is 0.125 bits a weight, three percent of
the file, and whether it is free is not a question squared error can answer —
see the trap below. Measured through the offline bench on Qwen3-0.6B at k=4,
everything else equal:

| | bpw | PPL | KL | top-1 |
|---|---|---|---|---|
| fp16 scale | 4.3125 | 29.7845 | 0.0658 | 84.2 % |
| eight-bit step | **4.1875** | 29.8677 | 0.0702 | 84.0 % |

A twentieth of a point of perplexity and some percent of the divergence, for
three percent of the file. The format takes the step.

### What it costs the weights

The quantizer reads **23.1 dB on every tensor to ±0.4 dB**, head included,
against D4G's 16.05 — the rotation makes them statistically identical, so one
number describes the model. Shannon's bound at four bits is 24.08 and this codec
reads 23.04 on a Gaussian, so what the file loses to the codec is a tenth of a
decibel and what the codec loses to theory is one.

### What the file reads

Qwen3-4B, calibrated on 2048 tokens of `wiki.train.raw` in one pass, evaluated
on 4088 tokens of `wiki.test.raw` — disjoint, and neither is this README:

| | size | perplexity | KL | top-1 | top-5 |
|---|---|---|---|---|---|
| bf16 | 7.5 GiB | 19.2936 | — | — | — |
| Q4_K_M | 2.33 GiB | **20.0352** | 0.0715 | 90.1 % | 99.6 % |
| `.golem` T4G | **1.96 GiB** | 20.3757 | **0.0526** | **90.6 %** | 99.6 % |

Sixteen percent smaller than Q4_K_M, twenty-six percent closer to the original's
opinion, half a point better at naming the same word, and 1.7 % behind on
perplexity.

**That last figure is not a difference.** The evaluation is 4088 tokens in eight
windows and the two models see the same ones, so the comparison is paired and
can be tested: the mean gap is 0.017 nats against a standard error of 0.012 —
t = 1.4 — and T4G is ahead in two windows of eight. Repeating it on four times
the text settles it rather than deepening it:

| 16352 tokens, 32 windows | perplexity |
|---|---|
| Q4_K_M | 16.0023 |
| `.golem` T4G | 16.1185 |

0.7 % apart, mean gap 0.0072 nats against a standard error of 0.0062, t = 1.2,
and T4G ahead in **13 windows of 32**. Four times the data halves the gap and
leaves it inside the noise. There is no measured perplexity loss; what there is
is a measured divergence win.

The lesson is the one this file already carries about reporting both numbers,
with a second half: a perplexity difference of one or two percent on four
thousand tokens is not a result, and the windows a run already prints are enough
to say so.

Qwen3-0.6B, same corpora, at 298.5 MiB and 4.201 bits a weight: 30.8201 against
bf16's 28.8521, KL 0.0812, top-1 83.4 %, top-5 99.0 %.

The bits a weight are 4.194 and 4.201 rather than 4.1875 because a file also
carries one F32 vector per calibration site and leaves the norms in bf16.

### The head, which is the one tensor worth more bits

`-head 5` writes `token_embd` — the tied logit head — in the wide tier and
everything else in the ordinary one. It is not a hidden layer whose error the
layers after it absorb; it makes the logits, and llama.cpp's K-quant mixes have
always spent more there: Qwen3-4B's Q4_K_M gives it 6.56 bits and the rest 4.95.

On Qwen3-0.6B, where the head is a quarter of the weights:

| head | file | PPL | KL | top-1 |
|---|---|---|---|---|
| four bits | 298.5 MiB | 30.8201 | 0.0812 | 83.4 % |
| **five bits** | 317.1 MiB | 30.5240 | 0.0718 | 84.8 % |
| bf16 — the ceiling | 517.6 MiB | 30.3016 | 0.0662 | 85.2 % |

Five bits takes about three fifths of the way to an unquantized head for six
percent of the file rather than seventy-three. It is off by default because the
smallest file is the point, and because four bits already reads a quarter closer
to bf16 than Q4_K_M does at sixteen percent less. On Qwen3-4B the head is a
tenth of the weights rather than a quarter, so five costs 0.10 GiB.

### The offline bench is not a file, and the difference is one mechanism

`cmd/vqbuild` holds 32 columns per matrix out of the quantizer at 8 bits. **No
`.golem` can**: the format has nowhere to put them, and neither codec has ever
stored one. So every number the bench reports is of a scheme with an extra part,
and the bits it prints do not count it — `Opts.BPW` leaves the held-out columns
out of its own total.

What that part is worth, on Qwen3-0.6B at the same rate and codec:

| | PPL | KL | top-1 |
|---|---|---|---|
| bench, salience unbounded, 32 columns held out | 29.87 | 0.0702 | 84.0 % |
| bench, salience unbounded, none held out | **39.67** | **0.3105** | 69.4 % |
| the file: salience bounded to 24×, none held out | 30.82 | 0.0812 | 83.4 % |

Ten points, and the whole of it is the handful of columns whose salience scale
would otherwise dominate the group it is rotated with. The bench solves that by
paying 8 bits for them; the converter solves it by bounding the scale, which
costs no bits at all and recovers all but a point of the difference. They are
two answers to one problem and the file's is nearly as good — but it is not
quite, and a point of perplexity is what stands between this format and the
perplexity column of the table above.

### The step is chosen after the path

A block's RMS is the scale that makes it unit-variance, which is not the scale
that reconstructs it best. The lattice buys that difference with a grid of seven
candidate steps per block; a trellis cannot, because one path spans two blocks
and the search would have to be joint. Least squares takes it exactly and for
nothing once the path exists, and it is worth about 4 % of the error.

### The two encoders are held to a weaker contract than D4G's

`TestEncodeD4GMatchesCPU` demands byte equality because a D4 code is an index
into a shared enumeration: two encoders that disagree write different files for
the same weights. A trellis records the path it chose, and **any minimum-cost
path is an equally valid file**. `TestViterbiMatchesCPU` therefore holds the two
to the same *cost*, to a part in a hundred thousand, and reports agreement
(94.2 %) only as a collapse detector.

**This does not extend to the decoder.** A decoder is a pure function of the bits
it reads, so Go and the shader must agree exactly, and
`TestT4GDecodeMatchesCPUExactly` sweeps a one-hot activation over a whole row to
say so weight by weight. The step grid is uploaded as 256 floats rather than
recomputed with an `exp2` for the same reason: this card's `exp2` lands one unit
in the last place from Go's, which is 3e-7 of an output and is exactly the kind
of invisible this format cannot afford.

### On the card

`vk/shaders/viterbi_tcq.comp`. The Viterbi is 2^L operations a weight — four
thousand at L=12, against the lattice's eight — and it is the only half of a
conversion that cares where it runs: the 0.6B goes from 25 minutes on twelve
cores to 1 min 07.

## The scheme, and the one thing to understand about it

A matrix is not stored as its weights. It is stored as **A·(q ⊙ W)**: its
columns scaled by a per-column vector `q`, then rotated by a Hadamard transform
in groups of 128 with random signs folded in.

Nothing undoes that on the weight side. The activation meets **A·(x / q)**
instead — the same function with the reciprocal vector — and since AᵀA = I the
product is unchanged. `nn.PrepareD4G` is both directions; it runs once per site
rather than once per matrix, which is why the scheme is cheap at inference.

Two consequences, and both have cost this repository a day:

**A matrix reads the vector of its site, not its own.** The three projections
of an attention read one stream and so share one vector; a hybrid's four input
projections share the same one; the feed forward's gate and up share another.
`nn.D4GVectorNames` is where that convention lives, for the converter and every
reader alike — a copy of it anywhere else drifts, and a file whose matrices were
quantized against one vector and are read through another loads, agrees about
every shape, and answers nonsense.

**A row read on its own is not a product.** The embedding table is read a token
at a time, and that row has to be the embedding rather than the rotated form a
product would want. `nn.Matrix.Row` undoes the transform when the matrix carries
its vector; `MatVec` does not, because the activation already did.

## The salience, which is most of what the format is

`q` is not just signs. It carries an AWQ-style scale: the per-column power of
the activations that reach a site, raised to α = 0.5 and normalised by its
geometric mean. Converting without it — signs and rotation alone — costs twenty
points of perplexity on Qwen3-0.6B, 39.80 against 60.01. The lattice, which took
the most work, is worth 1.5 of those points.

The scale is **bounded to 24×**, and that is not a detail. It is applied before
a rotation that mixes 128 columns together: a column shrunk by two thousand is
mixed with one left alone, quantized as if it were the second, and multiplied
back by two thousand on the activation side. At α = 0.75 the spread reaches
eighteen thousand and the model reads at a perplexity of 246 rather than 40.

`cmd/golemquant` measures the sites by running the model — on a card when it can,
which is fifteen seconds for eight thousand tokens of a 27B against an hour and
a half of eight cores — and refuses to write a file it could not calibrate.

## What the numbers are

On Qwen3-4B, against the same corpus, held out from calibration:

| | size | perplexity | KL | top-1 | top-5 |
|---|---|---|---|---|---|
| bf16 | — | 21.27 | — | — | — |
| Q4_K_M | 2.33 GiB | 21.57 | 0.053 | 88.7 % | 99.3 % |
| `.golem` | 1.52 GiB | 23.80 | 0.175 | 78.2 % | 97.1 % |

Perplexity says twelve percent worse. The divergence says three and a third
times further from the original's opinion, and a different word chosen one time
in five where Q4_K_M chooses one in nine. Both are worth having and they are not
the same number — `cmd/vqdiff` reports both, and a format judged on perplexity
alone is a format flattered by it.

The quantizer reads **16.05 dB on every tensor to ±0.03 dB**, head included: the
rotation makes them statistically identical. Shannon's bound for a memoryless
Gaussian at three bits of code is 18.06 dB and a D4 lattice with a spherical
boundary tops out near 16.9, so 0.85 dB is what is left. Matching Q4_K_M's 4.95
bits at 3.26 would want 10.9 dB, and 1.7 bits is 10 dB: no published method
recovers that without end-to-end fine-tuning.

Converting from an already-quantized checkpoint costs little — 23.97 from
Q4_K_M against 23.80 from bf16 — so there is no need to fetch a 47 GiB bf16 when
a K-quant is at hand.

## What is measured and does not work

- **GPTQ compensation** (`gptq.go`): 0.15 points on a real model, where it halves
  the output error on synthetic data. The rotation whitens the Hessian and
  leaves nothing to redistribute. The code works; it just buys nothing here.
- **Non-uniform bit allocation**: per-weight sensitivities span 2.4× and
  `ffn_down` is indeed the most sensitive, but the arithmetic-to-geometric mean
  ratio is 0.17 dB. Three code widths and three shader variants for that: no.
- **Deltas between adjacent layers**: blocks differ by 1.30–1.37 against 1.414
  for two unrelated matrices. They share nothing.
- **Low rank**: the spectrum is not flat and the rotation does not touch it, but
  rank 64 in fp16 costs 46 % of the bits to remove 2.1 dB where 46 % more code
  gives 9.
- **A per-row gain**: γ = 1.0000 ± 1.7 %, which removes 0.5 % of the error.
- **A global gain on the trellis codebook**: g = 1 is optimal and any departure
  costs. The adaptation that pays is per block, and least squares takes it
  exactly.
- **The salience bound, and its exponent**, for the trellis. Six settings on
  Qwen3-0.6B — bounds of 8, 12, 24, 48 and none at α = 0.5, and α = 0.4 and 0.6
  at 24 — read 30.68 to 30.82 and KL 0.0800 to 0.0850. The default is as good as
  any of them, which is the same answer `-search` gets by a different route.
- **More calibration**, for the trellis. On Qwen3-4B in one pass, 2048 tokens
  read 19.7285 and KL 0.0510; 8192 read 19.7004 and **0.0550**. More calibration
  buys average likelihood and sells per-token agreement. Windowing it to match
  the evaluation's regime is worse on both: 19.7808 / 0.0566.
- **Choosing the salience per site** (`-search`, left off): six settings give a
  mean of 39.73 against 39.80 for one bound chosen for the whole model. The
  spread is the choosing, not the choice.

## The codebook is worth four times what theory gives it

At identical size on Qwen3-0.6B: the D4 lattice (16.21 dB) reads 39.80, Lloyd
with eight levels in registers (15.92 dB) reads 41.30, uniform eight levels
(15.34 dB) reads 43.22. The error gap between lattice and Lloyd is 0.29 dB,
which predicts half a point; the measurement is 1.50. Three explanations were
tested and all three are false — it is not tail clipping, not error cancellation
in the dot product, not a preference for a uniform step. `-codebook lloyd` is
kept to ask the question again in ten minutes.

## Traps

- The calibration corpus must be long **and** disjoint from the evaluation. An
  early one of 173 tokens made GPTQ overfit by sixteen points.
- A step quantized to an eighth rather than a sixteenth costs a full point of
  perplexity that the weights' squared error cannot see — 0.03 dB. Do not
  arbitrate the step's precision on MSE.
- Never rebuild a binary while a measurement is using it, and name each
  generation. Kill by explicit PID: `pkill -f` on a scratchpad path kills the
  calling shell.
