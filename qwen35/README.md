# qwen35: Qwen3.8, in Go

The package is named for what the checkpoint declares. `general.architecture =
qwen35`, as in llama.cpp's own `models/qwen35.cpp`.

Qwen3.8 27B is a dense model of sixty-four blocks: forty-eight gated delta nets
and sixteen attentions, three to one. It carries a vision tower, and a
sixty-fifth block that predicts the token after the one being decoded.

## Speed, and where it runs

| | prompt | a token at a time | drafting |
| --- | ---: | ---: | ---: |
| RX 9070 XT | 1283 /s at 512 positions | 27.6 t/s | **40.1 t/s** |
| i7-9700K, 8 threads | 0.7 /s | 0.70 t/s | refused |

The processor figure is not a tuning failure. A delta net rewrites a 128×128
state per head on every token, and that is arithmetic no kernel makes cheaper.
This model wants a card.

**It also wants a card to itself**, and that is a measurement rather than an
opinion. The Q4_K_M checkpoint is 15.65 GiB on a heap of 15.92, so on a card
that is also driving a desktop there is no room left. llama.cpp reads 476
positions a second at sixty-four and then 79 at two hundred and fifty-six,
which is a collapse rather than a rate. This is why the model is absent from
the comparison table in the root README: neither engine is being measured
there, the memory pressure is.

## Drafting

The prediction head guesses the next token from the state the trunk has already
computed. The next pass then carries two columns instead of one and verifies
both. The card reads a block's weights once whichever it is, so the second
token is close to free.

Every token returned is drawn from the model's own distribution, which is what
separates this from drafting with a separate small model: there is no
accept-reject correction to apply, and no way for the output to drift.
`TestSpeculativeGenerate` and
`TestVulkanGolemSpeculationDrawsWhatTheModelDraws` assert that the answers are
identical with drafting on and off. 70% of drafts are accepted on the run above.

**On a card with no room, drafting is dropped rather than paid for.** The
prediction block and a shadow of every delta net's recurrence are about four
hundred mebibytes. With them in place on a 15.92 GiB heap, the driver leaves the
logit head in system memory and the model reads a gigabyte across the bus for
every token: 3.5 tokens a second against 26.5 with the head resident.
Speculation is worth about thirty per cent and a head on the bus costs
eighty-eight, so golem asks the heap before it uploads, drops the block when it
will not fit, and says so on standard error. The Q4_0 build is 14.94 GiB and
keeps both.

Turning it negative is what that looks like from outside: 21.1 tokens a second
with drafting against 27.6 without.

### Why it is refused on the processor

A speculative step costs a draft plus a pass of two columns, against the
one-column pass it hopes to replace.

| cost, as a fraction of one token | on the card | on the processor |
| --- | ---: | ---: |
| the draft | 0.08 | 0.03 |
| the pass of two columns | **0.94** | **1.97** |
| drafts that must be accepted to break even | 2% | 99% |

The card is bound by reading the weights, so two columns cost what one did. The
processor at this size is bound by arithmetic, so two columns cost two columns,
and no acceptance rate pays for that. `cost_test.go` is where both tables come
from.

## Reading a prompt on the card

The same bargain reads a prompt. A pass carries up to five hundred and twelve
positions, and what one costs says how far that goes.

| positions in the pass | 1 | 16 | 32 | 64 | 128 | 256 | 512 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| the pass | 33.2ms | 57.0ms | 52.4ms | 80.2ms | 137.7ms | 219.4ms | 399.2ms |
| positions a second | 30 | 281 | 611 | 798 | 929 | 1167 | **1283** |

Thirty-two positions cost less than sixteen, and that is not a misprint.
Sixteen is the widest pass the mat-vec builds, and thirty-two is where the
tiled products take over.

A prompt reads at 1283 positions a second here, against 62 back when a pass
carried two. llama.cpp's Vulkan build reads the same file on the same card at
1211.67 ± 1.63 (`llama-bench -p 512 -r 5`). Both benchmarks do the same work
(512 tokens from an empty cache, both warm, both synchronised, load time
excluded) with one difference in llama.cpp's favour: golem reads every
position's hidden state back over the bus where llama.cpp keeps only the last.

### Where a pass goes

Both engines report their own GPU timers op by op, `vk.Timeline` here and
`GGML_VK_PERF_LOGGER=1` there. One pass of 512 positions, in milliseconds:

| | golem | llama.cpp |
| --- | ---: | ---: |
| feed forward, gate and up | **132.6** | 149.0 |
| feed forward, down | 97.9 | **95.6** |
| delta net: q, k, v, gate, α, β | **55.1** | 60.9 |
| attention | 32.4 | **2.1** |
| delta net: output projection (Q5_K) | **24.1** | 28.0 |
| SwiGLU | 17.1 | **11.2** |
| recurrence, with its normalisations | **15.9** | 23.2 |
| attention: q, k, v | **14.4** | 15.6 |
| everything else | **26.4** | 38.8 |
| **the pass** | **416.0** | 424.4 |

The card is busy for 97.6% of that wall clock, so nothing measurable is lost
between dispatches and every difference above is a kernel.

Attention is the one line where the gap is a shape rather than a margin. golem
dispatches the scores, the softmax and the mix as three passes over memory
where llama.cpp fuses them and keeps the block of scores where it computed it.
Only sixteen of the sixty-four blocks attend at all, which is both why thirty
milliseconds buys the other engine so much here and why this is the next line
to take.

## Several conversations at once

`-parallel N` holds N conversations on the card, each with the context cut N
ways. It is not done the way gemma and qwen do it, by cutting the attention's
ring into slots: forty-eight of the sixty-four blocks are delta nets, whose
state is a matrix a head that every token rewrites, and there is nothing to cut.
So each slot owns a copy of every recurrence, its convolution window, its
shadows and its attention caches. On the 27B that is 151 MB of state a slot,
twice that where drafting keeps shadows.

The copies are laid end to end, one buffer a block, and the slot travels beside
the position in a buffer the pass writes, as gemma's does. A recording names no
slot, so a pass carries several conversations: the projections read the weights
once for all of them, and the convolution, the recurrence and the scores are
dispatched once a conversation, side by side. `TestVulkanSlotsShareAPass` holds
such a pass to the same conversations run one at a time, to the bit.

A state cannot be rewound, so golem-server continues a slot when the prompt is
what the slot holds and more. Otherwise it goes back to a checkpoint, a copy of
the slot's state taken where an earlier prompt parted or every thousand and
twenty-four positions (`SetCheckpoints`, `SaveCheckpoint`), or starts the slot
again. `TestVulkanCheckpointResumes` holds a resumed prompt to the same prompt
read from nothing, and `TestVulkanSlotsAreIndependent` checks that one
conversation run between two steps of another changes nothing, to the bit.

On the processor it holds slots like every other engine here.

## Sight

The tower has no fixed input size. A picture keeps its aspect ratio, is scaled
to whatever grid the token budget allows, and the learned position table is
interpolated onto that grid, which is why an image of any shape gives a
different number of rows.

Every waypoint is held to llama.cpp's: worst gap 1.1e-4 at the patches and
6.4e-3 after all twenty-seven blocks. It sees and does not hear, because the
projector carries no audio encoder.

```bash
./golem-cli -model Qwen3.8-27B-Q4_0.gguf -mmproj mmproj-F16.gguf \
    -image photo.png -p "Describe this image in one sentence."
```

## Bonsai 2, the same model in ternary

Prism ML's [Bonsai 2 27B](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf)
is this architecture with every projection, the embedding and the head in
weights of -1, 0 and +1 times one fp16 scale a hundred and twenty-eight. The
file says `qwen35` and this package opens it; what differs is two weight
formats and a rotation.

**The formats.** PQ2_0 is a trit in two bits, 34 bytes a block. PTQ1_0 packs
five trits a byte, 28 bytes a block. Both files hold the same trits, and a
tensor read from one is the same floats to the bit as read from the other.

**The rotation.** Every matrix was multiplied offline by a Walsh-Hadamard
transform of 1024 with a fixed sign vector, so its activation has to meet the
same transform before the product: `y = W·H(s ⊙ x)`. That is
`nn.PrepareGolem` with a group of 1024, which `.golem` checkpoints already
needed at 128. The embedding table is stored rotated too and is brought back
a row at a time. And the delta net's output is reordered from llama.cpp's tiled
head order to the grouped one before its rotation, which `BlockWeights.SSMOutGather`
carries. `prism.go` reads all of it from the file's `prism.hadamard.*` keys and
refuses a file whose ternary matrix it could not bind: a projection left
unrotated answers fluently and wrongly.

On the card each of the five sites a block is one dispatch of
`vk/shaders/rotate_q8.comp`, from the floats the kernel in front wrote to the
Q8_0 form the projections read. The floats stay unrotated, because the delta
net's two decay projections read them and those are not rotated in the file.

`TestVulkanBonsaiMatchesLlamaCpp` holds both files to the logits of Prism's own
llama.cpp fork at every position of a 31-token prompt: 0.0001 nats, the same
top-1 at every position.

| RX 9070 XT, 2026-09-23 | one at a time | drafting | pp512 |
| --- | ---: | ---: | ---: |
| PQ2_0 | 46.0 t/s | **62.8** | 1253 |
| PTQ1_0 | 15.8 t/s | 22.7 | 1128 |

Greedy, 128 tokens of a short English paragraph with 57 % of drafts kept, the
median of three runs; pp512 is the median of three passes after a first one
that is not counted. The card is held at `high` and the desktop at the
`performance` power profile. With the card on `auto` and the desktop on
`power-saver`, the same code draws 32.6 tokens a second.

PTQ1_0 reads 17 % fewer bytes and is three times slower, because its mat-vec
takes a weight at a time out of a byte of five where PQ2_0 spreads four with
two instructions. On this card the product is not waiting on memory.

A prompt read five hundred and twelve at a time goes through the int8 form of
`vk/shaders/matmul_coop.comp`: trits and Q8_0 magnitudes are exact in eight
bits, so the matrix cores take them as they are rather than as fp16. It is
worth 14 % on PQ2_0 and 10 % on PTQ1_0; the kernel's header has the shapes.

**Drafting is grafted.** Prism ships no prediction block. The original model's
`blk.64` works unchanged: the residual stream between blocks is not rotated,
only what each projection reads, so the block sees what it was trained on
within the ternary model's error. Copied into the file with `block_count` at
65 and `nextn_predict_layers` at 1, 79 to 86 % of drafts are accepted on
English and code and 48 % on a French prompt, and the text is the same with
drafting on and off. The grafted block's own matrices are Q4_0 and unrotated,
which is why the pipeline decides the rotation per block and not per model.

```bash
golem-cli -vulkan -model Ternary-Bonsai-2-27B-PQ2_0-mtp.gguf
```

`-vulkan` matters more than usual here: on eight cores this model draws 0.2
tokens a second.

