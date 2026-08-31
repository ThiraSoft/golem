# compress

A checkpoint in, a `.golem` out. One codebook, and a scheme around it:

| | block | bits/weight | tensor type |
|---|---|---|---|
| `T3G` | 128 weights in 52 bytes | 3.25 | 1000 |
| `T4G` | 128 weights in 67 bytes | 4.19 | 1001 |
| `T5G` | 128 weights in 83 bytes | 5.19 | 1002 |

`T3G` is the narrow body of a small model or a tight budget. The head defaults
to four bits whatever the body is — a logit head narrower than the body would
spend bits where they are worth least — but `-head` can still be set to match
the body; `golemquant` only refuses a head *narrower* than the body, not an
equal one.

The file is a GGUF. Same container, so the vocabulary, the rope base and the
chat template travel unchanged; what is new is a tensor type llama.cpp does not
know and one vector per calibration site beside the matrices.

Everything below about the scheme — the salience, the bound of 24×, the rotation
by 128, `nn.D4GVectorNames` and the `A·(q ⊙ W)` convention with its two rules —
is where most of the format's value is, and it is independent of the codebook:
a lattice carried it before the trellis did, and lost only the codebook question
when it was retired.

## What a block holds: a trellis, with nothing to look up

A sequence of 128 weights costs 67 bytes, 4.1875 bits each, and there is no
decode table at all — the whole reason this codebook exists. An earlier
codebook, a D4 lattice, decoded by table lookup: 3961 points at r²=40 fit a
workgroup's thirty-one kibibytes of shared memory at twelve bits a code, but
the four-bit tier its successor needed would have taken a table of 493 KiB,
which does not fit. The trellis was written to close that gap, and once it did,
carrying two codebooks bought nothing a measurement could find at any rate
either format reached — so the lattice was retired rather than kept beside it.

`golem.hadamard_group` and `golem.scale_block` in the metadata say what the
file was written with — both name the scheme, and are unchanged by which
codebook a block holds. `general.file_type` is 1000 for `T3G`, 1001 for
`T4G`, 1002 for `T5G` — the body's tier, since the body is what the file is
named for even when the head is written wider.

The state is the last twelve bits of the code stream, so
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

A row is two planes, steps then codes: a 67-byte block would otherwise start on
an odd boundary, and a shader reads words. 128 is not free choice:
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

Eight bits naming powers of two a sixteenth apart. A grid an eighth apart costs
a whole point of perplexity against an fp16 step at the same granularity, which
is more than the finer granularity wins back — the step sits at the bottom of a
curve, and nine percent is far enough up its sides to matter.

The window is 6.1e-5 to 3.83, not the wider range an earlier, lattice-based
codebook used. A lattice's step is a *fraction* of its block's RMS — the block
is scaled up into a shell several units across — while a trellis step is the
block's RMS itself, and the two windows are not interchangeable: a unit-variance
source clipped at the wider window's ceiling reconstructs at **5.5 dB instead of
23**, with every shape still agreeing.

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

The quantizer reads **23.1 dB on every tensor to ±0.4 dB**, head included — the
rotation makes them statistically identical, so one number describes the model.
Shannon's bound at four bits is 24.08 and this codec reads 23.04 on a Gaussian,
so what the file loses to the codec is a tenth of a decibel and what the codec
loses to theory is one. The retired lattice read 16.05 dB at the rate it spent,
14.97 at four bits — the numbers that closed the question of which codebook to
keep.

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

Qwen3-0.6B, same corpora, at 298.5 MiB and 4.201 bits a weight, calibrated on
2048 tokens (this format's default): 30.3744 against bf16's 28.8521, KL 0.0800,
top-1 84.2 %, top-5 99.0 %.

The bits a weight are 4.194 and 4.201 rather than 4.1875 because a file also
carries one F32 vector per calibration site and leaves the norms in bf16.

Converting from an already-quantized checkpoint costs little — 23.97 from
Q4_K_M against 23.80 from bf16, in the older, D4G-era regime this was first
measured under — so there is no need to fetch a multi-gigabyte bf16 checkpoint
when a K-quant is already at hand. This is a property of the scheme, not the
codebook: a converter reads whatever floats it is given back out of whichever
format they arrived in, and neither the salience nor the rotation cares.

### What three bits buys, and what is not three bits

`T3G` cuts the trellis's window in half: four bits a weight down to three,
520 bits of path down to 391 — twelve for the first weight in a window plus
three for each of the 127 after it — with the same two eight-bit steps per
128-weight row. The seven padding bits are what keep the row on a byte
boundary anyway: 2 (steps) + 391 (path) = 393 bits, and 52 bytes is 416, so
seven bits ride along unused rather than the block spilling into a 53rd byte
a shader would have to read and mask around. They cost nothing measured —
the row is 3.25 bits a weight, not 3.0555, and that eighth of a bit is the
price already visible in the table at the top of this file.

A `.golem` T3G file is not three bits throughout. `golemquant` defaults the
head to four-bit `T4G` there — the logit head is still worth more bits than
the body, for the reason above — though `-head 3` is accepted if asked for.
State the body's rate, the head's rate, and the weighted rate together, or a
file's size will get compared against a Q3_K figure that assumes a codec
this uneven never happens:

| model | body | head | weighted | file |
|---|---|---|---|---|
| Qwen3-0.6B | 3.25 (T3G) | 4.1875 (T4G) | 3.509 | 255.0 MiB |
| Qwen3-4B | 3.25 (T3G) | 4.1875 (T4G) | 3.347 | 1.573 GiB |

The 4B sits closer to the nominal 3.25 than the 0.6B because the head is a
smaller fraction of a bigger model's weights — about a tenth here, versus
about a quarter on the 0.6B — so the four-bit tax on it moves the weighted
rate less.

Measured on Qwen3-4B, same corpus and eight windows as the T4G table above,
against llama.cpp's own floor at three bits — Q3_K_M at 1.93 GiB, Q3_K_S at
1.76 GiB:

| | size | PPL | KL | top-1 | top-5 |
|---|---|---|---|---|---|
| Q3_K_M | 1.93 GiB | 24.0253 | 0.2453 | 79.8 % | 97.7 % |
| Q3_K_S | 1.76 GiB | 24.9781 | 0.3336 | 77.9 % | 96.7 % |
| `.golem` T3G | **1.573 GiB** | **21.1078** | **0.1771** | **83.1 %** | **98.3 %** |

Smaller than either K-quant tier and ahead on every axis — and unlike the
T4G-vs-Q4_K_M gap above, this one clears the paired test rather than falling
inside it. The same eight windows, tested the same way: T3G's mean per-window
gap against Q3_K_M is −0.1295 nats/token, standard error 0.0441, t = 2.94 on
seven degrees of freedom, above the two-tailed 5 % bound, and T3G wins six
windows of eight (seven of eight against Q3_K_S, t = 5.20). Three bits is a
cliff for the K-quants — 0.17 GiB between Q3_K_S and Q3_K_M buys them less
than a point of perplexity, a quarter of what the next 0.4 GiB up to Q4_K_M
buys — and the trellis clears that cliff instead of sliding down it with them.

Qwen3-0.6B, same corpora, calibrated on 2048 tokens: 34.8552 against bf16's
28.8521, KL 0.2498, top-1 73.6 %, top-5 96.4 %, at 3.509 bpw weighted and
255.0 MiB. `golemquant`'s conversion log reports how much of that ran where —
4022 M weights through the card and 0 M through the processor's k=3
fallback on the 4B — which is what a shader landing the new codebook should
show; a large processor figure would mean the kernel was never reached.

### Tail-biting: built, measured, and not taken

The seven padding bits and the twelve priming bits are what a **tail-biting**
layout would give back. Close the path on itself — weight *t*'s window becomes
bits `[3t, 3t+12)` modulo 384 — and a sequence is 128 three-bit symbols and
nothing else: 48 bytes of path, 50 of block, **3.125 bits a weight** against
3.25. It is legal at this tier alone, because L = 4k only at k = 3, so a state
is four whole symbols and the file can store the top three bits of each state
at bit offset 3*t*. It was built, measured against this layout on the same
corpus and the same eight windows, and **not taken.** The reason is not the one
that was expected, so it is worth writing down properly.

**The constraint is cheap.** Solving a tail-biting path exactly means
minimising over all 2^(L−k) start prefixes with the true cycle condition. Done
that way, on i.i.d. N(0,1):

| | SNR |
|---|---|
| padded, ML | 17.25 dB |
| tail-biting, exact | 17.01 dB |

**0.24 dB** — *less* than the 0.41 dB nine fewer bits of freedom would predict,
because the padded layout spends twelve of its bits priming a single weight and
tail-biting spends none. Closing the loop is close to free.

**The two-pass encoder is what is expensive.** The standard approximation — one
unconstrained pass whose terminal state names the tail, then one forced to start
where that tail leads — reads **16.31 dB**, 0.70 dB below the exact path. Pass
one picks its terminal state to fit the *last* weight with no regard for the
first, and pass two then pins nine of weight 0's twelve state bits to a value
chosen for the other end of the sequence. Iterating it to a fixed point does not
help: 2, 3 and 8 passes all settle at 16.30, which shows the heuristic converges
and says nothing about whether it converges to the right place. Searching eight
candidate prefixes instead of one — nine Viterbi passes rather than two, at
conversion time only — recovers 0.63 of the 0.70 and reads **16.94 dB**.

**So the subject is parked, not closed** — but the prize is thin. Even at
best-of-eight the residual deficit is about 0.32 dB against 0.125 bits a weight
saved, which nets roughly two percent of file size, and it is paid against a
measured generation regression (see below) whose cause was never established.
End to end, with the two-pass encoder, Qwen3-4B read 22.0596 and KL 0.2129 at
1.520 GiB, against 21.1078 and 0.1771 at 1.573; Qwen3-0.6B read 36.7061 and
0.3215 at 248.4 MiB against 34.8552 and 0.2498 at 255.0.

**And the premise the change was built on does not hold.** Tail-biting was
justified by 3.8 % fewer bytes read per token, on the reasoning that generation
reads every weight once and reuses nothing, so bytes convert to tokens per
second nearly one for one. The traffic saving is real — 1000 bytes a row against
1040, and every byte of a padded sequence *is* read, including its last, because
half-block 1's eighth group reads bytes 45 through 49. It simply does not
matter: **`matvec_t4g` moves about 77 GB/s against roughly 640 GB/s of GDDR6 on
this card, an eighth of the memory ceiling.** Generation here is not
bandwidth-bound, so trimming a few percent of traffic buys nothing measurable.
Measured back to back, five alternating runs each, tail-biting read 47.20 t/s
(sd 0.23) against the padded layout's 48.94 (sd 0.56) — a real regression, five
runs against five with no overlap, and its cause is not established.

That last paragraph is the part to carry forward. **Any future change argued for
on the grounds that it reads fewer bytes has to clear this first**: at an eighth
of the bandwidth ceiling, bytes are not what the kernel is short of.

### The head, which is the one tensor worth more bits

`-head 5` writes `token_embd` — the tied logit head — in the wide tier and
everything else in the ordinary one. It is not a hidden layer whose error the
layers after it absorb; it makes the logits, and llama.cpp's K-quant mixes have
always spent more there: Qwen3-4B's Q4_K_M gives it 6.56 bits and the rest 4.95.

On Qwen3-0.6B, where the head is a quarter of the weights:

| head | file | PPL | KL | top-1 |
|---|---|---|---|---|
| four bits (2048-token calibration) | 298.5 MiB | 30.3744 | 0.0800 | 84.2 % |
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
| the file: salience bounded to 24×, none held out | 30.37 | 0.0800 | 84.2 % |

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

### The two encoders are held to a weaker contract than byte equality

A codebook that is a table has one right index for a point — an index into a
shared enumeration — so two encoders that disagree write different files for
the same weights, and byte equality is the contract to hold them to. A trellis
records the path it chose instead, and **any minimum-cost path is an equally
valid file**: `TestViterbiMatchesCPU` therefore holds the two encoders to the
same *cost*, to a part in a hundred thousand, and reports agreement (94.2 %)
only as a collapse detector.

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
points of perplexity on Qwen3-0.6B, 39.80 against 60.01. The codebook, which
took the most work, was worth 1.5 of those points when it was the D4 lattice —
a comparison against the codebook doing nothing, not against the trellis that
replaced it.

The scale is **bounded to 24×**, and that is not a detail. It is applied before
a rotation that mixes 128 columns together: a column shrunk by two thousand is
mixed with one left alone, quantized as if it were the second, and multiplied
back by two thousand on the activation side. At α = 0.75 the spread reaches
eighteen thousand and the model reads at a perplexity of 246 rather than 40.

`cmd/golemquant` measures the sites by running the model — on a card when it can,
which is fifteen seconds for eight thousand tokens of a 27B against an hour and
a half of eight cores — and refuses to write a file it could not calibrate.

## What is measured and does not work

- **GPTQ compensation**: 0.15 points on a real model, where it halves the
  output error on synthetic data. The rotation whitens the Hessian and leaves
  nothing to redistribute. It worked; it just bought nothing here, so the
  factoring machinery (`Comp`, `NewComp`, the Cholesky routines) was removed
  from `gptq.go` along with `cmd/golemquant`'s `-gptq` and `-damp` flags —
  what is left is the windowed-Hessian accumulator, which the salience search
  still reads.
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
  the evaluation's regime is worse on both: 19.7808 / 0.0566. **This does not
  generalise.** On Qwen3-0.6B, 2048 tokens read 30.3744 and 8192 read 30.8680 —
  a gap of 0.49 nats the *other* way round: more calibration makes perplexity
  worse on the smaller model. Whatever the 4B's pair of numbers said about
  calibration buying average likelihood at the cost of per-token agreement, it
  is a fact about that model's size, not about calibration in general — the two
  models disagree on which direction more calibration even moves perplexity, so
  a reader should not assume either row predicts a third model's.
- **Choosing the salience per site** (`-search`, left off): six settings give a
  mean of 39.73 against 39.80 for one bound chosen for the whole model. The
  spread is the choosing, not the choice.

## Traps

- The calibration corpus must be long **and** disjoint from the evaluation. An
  early one of 173 tokens made GPTQ overfit by sixteen points.
- A step quantized to an eighth rather than a sixteenth costs a full point of
  perplexity that the weights' squared error cannot see — 0.03 dB. Do not
  arbitrate the step's precision on MSE.
- Never rebuild a binary while a measurement is using it, and name each
  generation. Kill by explicit PID: `pkill -f` on a scratchpad path kills the
  calling shell.
