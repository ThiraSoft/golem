# compress

A checkpoint in, a `.golem` out: 64 weights in 26 bytes, which is 3.26 bits
each.

The file is a GGUF. Same container, so the vocabulary, the rope base and the
chat template travel unchanged; what is new is a tensor type llama.cpp does not
know — `D4G`, type number 1000 — and one vector per calibration site beside the
matrices.

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
