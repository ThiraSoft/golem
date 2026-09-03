<div align="center">
  <img src="assets/logo.svg" alt="golem" width="420">

**Gemma, Qwen and Pocket TTS in a single static Go binary.**  
_No Python. No cgo. No GPU required — and with `-vulkan` it keeps pace with llama.cpp's Vulkan build on the one AMD card it has been measured on._

[![test](https://github.com/ThiraSoft/golem/actions/workflows/test.yml/badge.svg)](https://github.com/ThiraSoft/golem/actions/workflows/test.yml)
[![Go Reference](https://pkg.go.dev/badge/github.com/ThiraSoft/golem.svg)](https://pkg.go.dev/github.com/ThiraSoft/golem)
[![MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

</div>

---

A golem is inert matter given a voice. That is what these engines do to a file of weights.

**Golem** is a set of inference engines written in pure Go. Run Gemma 4, Qwen3, Qwen3.8 and Kyutai Pocket TTS locally with `go build`, a GGUF file, and your CPU — or your Vulkan GPU.

## ✨ Features

- **Zero Friction**: Compiles to a single static binary. No Python environment, no `cgo`, no runtime to install.
- **Pure Go, four dependencies**: `purego` for the Vulkan loader, and three file formats the standard library does not read — WebP, MP3, FLAC. `CGO_ENABLED=0 go build ./...` passes.
- **OpenAI Compatible**: Drop-in replacement for OpenAI API clients, tool calls included.
- **Multimodal**: Text, Vision (images) and Audio (WAV/MP3/FLAC) via Gemma 4; Qwen3.8 sees too, its tower checked against llama.cpp waypoint by waypoint.
- **Verified, not asserted**: no layer is deemed correct until its intermediate activations match llama.cpp or PyTorch, waypoint by waypoint.
- **Keeps pace on CPU**: tuned AVX2 kernels reach `llama.cpp`'s level on this machine — reading prompts and generating both — except on the smallest model, where the weights stop being the cost and it says so.
- **Vulkan GPU**: bound through `purego` rather than cgo, and level with `llama.cpp`'s own Vulkan build on the one AMD card this has been measured on. The table below gives every number both ways, including where golem is behind and by how much.
- **Its own weight format**: `.golem` is 18 % smaller than llama.cpp's Q3_K_M on Qwen3-4B and ahead of it on every measure — a trellis codebook with no lookup table, converted on the card. [What it costs](#-golem--the-engines-own-weight-format).
- **A mixture's experts need not be on the card**: they can stay in system memory and be read across the bus where they lie, which takes the 26B A4B's footprint on the card from 13.6 GiB to 1.3. A cache of the ones a token keeps asking for buys the speed back — two fifths of the pool is four fifths of the tokens — and the answers do not change. [The measured curve](#-a-mixtures-experts-need-not-be-on-the-card).
- **Serves several clients at once**: `-parallel N` holds N conversations and carries a token for each of them through one read of the weights, on the card as well as on the processor — Qwen3.8 on the card excepted, for a reason [written below](#-several-conversations-one-pass).

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

`-parallel N` cuts the context into N slots, each holding its own conversation.

## 👥 Several conversations, one pass

Whatever is waiting at the moment a pass is built goes into that pass. Four clients each wanting a token are four tokens in one read of a gigabyte of weights, rather than four reads — llama.cpp's continuous batching, and it now works on the card as well as on the processor. The context is cut between the slots rather than multiplied, so `-context 4096 -parallel 4` is four conversations of 1024 and the memory is what it was.

Gemma 4 26B A4B on an RX 9070 XT, drawing a token for each conversation from 448 positions of context. These are from an earlier sitting than the table below and the two are not comparable — the section there says why:

| conversations | 1 | 2 | 4 | 8 |
| --- | ---: | ---: | ---: | ---: |
| tokens a second | 82.1 | **125.9** | **211.4** | **255.0** |

Through the API, with prompts of about five hundred and forty tokens: one client reads 4055 positions a second and draws 75.3; two clients read 4260 between them and draw 115.6.

What made it possible is that a column now says which conversation it belongs to as well as where it sits, so a block's cache on the card is one ring per conversation rather than one ring. `cmd/golem-server/README.md` has the rest.

**Qwen3.8 on the card is the exception**, and `-parallel` above 1 is refused there rather than fallen back from. Forty-eight of its sixty-four blocks are delta nets, and a delta net keeps a state matrix a head that every token rewrites rather than a ring indexed by position: there is no ring to cut into slots, and answering two conversations off one state is a wrong answer and not a slow one. On the processor it holds slots like the rest.

## ⚡ Vulkan GPU

`-vulkan` puts the whole model on the card. Measured on an RX 9070 XT in one
sitting, against `llama.cpp`'s own Vulkan build on the same Q4_0 files, both
sides warmed before timing:

| tokens a second | golem gen | llama gen | golem pp64 | llama pp64 | golem pp256 | llama pp256 | golem pp512 | llama pp512 |
| --------------- | --------: | --------: | ---------: | ---------: | ----------: | ----------: | ----------: | ----------: |
| Gemma 4 26B A4B |     105.4 |     115.6 |   **1858** |       1053 |    **3614** |        2704 |    **4011** |        3483 |
| Gemma 4 12B     |  **61.5** |      61.2 |   **1463** |       1209 |    **2557** |        2287 |    **2761** |        2683 |
| Qwen3 4B        |     130.4 |     139.0 |   **3528** |       3231 |    **5780** |        5585 |        5802 |        6686 |
| Qwen3 0.6B      |     259.6 |     270.3 |       7498 |      10188 |       14665 |       17297 |   **18756** |       18644 |
| Qwen3.8 27B     |      30.3 |      31.4 |    **798** |        655 |    **1170** |        1105 |    **1285** |        1212 |

**Read as a whole: golem reaches llama.cpp's level here, and that is the claim.**
Not more than that, and deliberately — this is one card, one driver, one
afternoon. A kernel that wins on RDNA 4 at these shapes need not win on another
architecture, another driver revision, or another checkpoint, and nobody has
run it there. The table above is what was measured; the conclusion to draw from
it is parity.

Reading a prompt, golem is the faster of the two on four of the five models at
every width, by a factor of one and three quarters on the 26B A4B at sixty-four
positions. The exception is the 0.6B, which is behind at sixty-four and two
hundred and fifty-six and level at five hundred and twelve: a model that small
spends a prompt in dispatch latency rather than in arithmetic, and there is
nothing in a kernel to win it back.

Generating, golem is level on the 12B and two to nine per cent behind on the
other four. Qwen3.8 is the closest of them because a gated delta net rewrites a
state matrix a head every token, which is arithmetic neither engine amortises;
the 26B A4B is the furthest.

And the absolute rates do not travel either. Earlier revisions of this file
carried a table from an evening when the same card ran a fifth faster on small
models — both engines by the same fraction, checked by re-running llama.cpp's
side as well — so the column to read is the difference between the two engines
and not the rate, and even that difference is this machine's.

**A K-quant runs on the card too.** Q4_0, Q4_1, Q4_K, Q5_K and Q6_K each have a
mat-vec and a tiled product here, and a checkpoint may mix them the way
llama.cpp's own quantizer does — a `Q4_K_M` gives the same role different
formats in different blocks, six bits on half its `ffn_down` and four on the
rest, and every matrix is asked what it is rather than told. A form with no
kernel is an error naming it, never a guess: eighteen bytes to a block of
thirty-two is Q4_0 and it is also Q4_K, so a reader that checked a length
instead of a type would answer fluently out of the wrong bits.

The card holds the whole model: 12.8 GiB for the 26B A4B, which is why sixteen is the smallest card that can run it at these speeds — though not all of that has to be on it, which is the next section. `-vulkan` is all or nothing and says so: a card without `VK_KHR_shader_integer_dot_product`, a machine with no Vulkan loader, a dense model too large for the card — each is an error at startup rather than a silent half-move. Without the flag, everything runs on the CPU as before.

_(See [ARCHITECTURE.md](ARCHITECTURE.md) for the kernel work behind these numbers.)_

## 🫙 A mixture's experts need not be on the card

**A dense model that does not fit cannot be rescued.** One column of a pass is
one multiply per weight, so every byte crossing the bus is used once: Qwen3.8
27B in BF16 is 54.8 GB a token, which is eight seconds a token on this machine's
link and no amount of engineering moves it. Making the weights smaller is the
answer there, and that is what [`.golem`](#-golem--the-engines-own-weight-format) is for.

**A mixture is different, and the difference is the whole opportunity.** The
26B A4B keeps 12.85 GB of experts and reads *eight matrices out of a hundred and
twenty-eight* per block — 0.8 GB a token. Eleven of its twelve gigabytes are
resident and untouched on any given token. So the cost of streaming a mixture is
what a token *activates*, not what the model *has*.

`GOLEM_MOE_EXPERTS_HOST=1` leaves the two expert stacks in system memory the
card addresses, and the kernels read them where they are. **No shader knows**: a
compute shader reads a storage buffer the same way wherever it lives, and only
the rate changes.

| 26B A4B on an RX 9070 XT | on the card |
| ------------------------ | ----------: |
| experts resident in VRAM  | 13.6 GiB   |
| experts in system memory  | **1.3 GiB**|

**What stays on the card is the shared branches, the attention and the head**;
the twelve gigabytes that leave are the experts. So what a mixture costs the card
no longer scales with how many experts it has, and what bounds it becomes host
memory instead — sixteen gibibytes of addressable system memory here. The speeds
are the table further down, which measures all of this on one continuation.

**The card stops being the limit, and the next one is measured.** The cache
sizes below hold as little as 1.6 GB of a 12.85 GB pool and answer the same
tokens, so what a mixture costs the card is now whatever you give it. What the
pool costs is a different ceiling: every submission of a token reaches every
block's share of it, and `vk/residency_test.go` walks that up a gibibyte at a
time — **fifteen submit and sixteen do not**, which is the heap's own size and
amdgpu's `gttsize` behind it. Allocating is not submitting: thirty gibibytes
allocate here without complaint, because the driver allocates lazily.

So the 26B A4B in Q4_0 fits beside the card and runs. The same model in Q8_0 —
`Q4_0` and `Q8_0` are both read by the routed kernels since v0.22.2 — puts 24 GB
of experts against that fifteen and is refused at the first submission. Running
*that* wants either a boot with a larger `gttsize` or a pool read from the mapped
file, and neither has been done here.

That arrangement is the floor for speed — every expert read across the bus,
nothing cached — and the bus is what binds it: this card sits behind a switch and
reaches the processor over eight lanes at 8 GT/s, 7.9 GB/s of payload. What
separates a resident token from a fully absent one is 802.9 MB of experts at
**6.37 GB/s**, which is the figure a mat-vec reads host memory at in isolation
and 95 % of what the copy engine manages across the same link.

**It answers the same tokens** — with the experts in system memory, and with any
size of cache in front of them. The reference test passes unchanged in all
three, down to the three logged tie margins being identical to the resident
run's. That control is what makes the arrangement worth building on: a mixture
that answers plausibly and wrongly is this project's named failure mode.

**A slice of the card buys most of it back.** `GOLEM_MOE_CACHE_SLOTS=N` keeps a
copy of N experts per block in device memory and fetches a missing one on the
way past. Measured on a continuation the model wrote itself — 160 tokens,
greedy, after a warm-up that is not counted:

| experts kept in VRAM | that much VRAM | tokens/s |
| -------------------- | -------------: | -------: |
| none — all in system memory | 0 | 7.3 |
| 16 of 128 | 1.6 GB | 15.1 |
| 32 of 128 | 3.2 GB | 27.3 |
| 40 of 128 | 4.0 GB | 35.3 |
| **51 of 128** | **5.1 GB** | **47.4** |
| 64 of 128 | 6.4 GB | 58.0 |
| all 128 — the resident path | 12.9 GB | 82.1 |

**Two fifths of the pool is four fifths of the tokens.** And the shape was known
before the cache existed: simulating one against the model's own routing — the
router is arithmetic, so it costs a map and no card time —
`gemma/expert_cache_test.go` predicted 34.5 tokens a second at 30 % of the pool
and 50.9 at 40 %, against the 35.3 and 47.4 measured here.

That simulation is also what says which cache to build. FreeToken reports that
one pool shared by every layer beats a split per layer by ten to fifteen points;
on this checkpoint the two sit under a point apart, so golem keeps one cache per
block — simpler to index, and the blocks do not compete.

**Nothing on the host decides any of this.** The experts are picked on the card,
one block at a time, inside a program recorded once and re-run per token, so a
host that had to choose would cost a readback between every pair of blocks.
`shaders/moe_admit.comp` turns each identifier into the slot holding a copy of
it, `shaders/moe_fill.comp` fetches whatever was missing, and the two product
kernels read a slot where they read an identifier — the same instruction. A
fetch costs what reading the expert where it lay would have cost, and buys every
later token that wants it again.

The prompt is unchanged: a wide pass reads every expert of a block at once, and
a cache of a few dozen has nothing to offer it, so passes above the cache's width
go by expert straight out of the pool.

**Q4_0 and Q8_0 both go through this.** A Q8_0 mixture is eight and a half bits a
weight against four and a half, which is the smallest form that puts a pool past
what a card can address — the reason to read it at all. A Q8_0 weight is a signed
byte against an activation that is already signed bytes, so the packed dot takes
both as they are, with no nibbles to unpack and no correction term. The two
routed kernels and `vk/quantproduct.go`'s door read it; the Q4_0 binaries they
share a source with are unchanged, instruction for instruction.

The pool has to fit in the memory the card can address, which is sixteen
gibibytes here against the 26B A4B's 12.85 — so that model fits and a much
larger one does not. Past that the source is the mapped file and the rate is the
disk's.

## 🔮 Qwen3.8 drafts its own next token

The Qwen3.8 checkpoint ships a sixty-fifth block: a multi-token-prediction head that guesses the token *after* the one just decided, from the state the trunk has already computed. Guess right and the next pass verifies two tokens for the price of one. Guess wrong and it costs the pass it rode on and nothing else — every token returned is drawn from the model's own distribution, so this needs none of the accept-reject correction that drafting with a *separate* model does. The answer is the same with drafting on and off, and a test asserts it.

Qwen3.8 27B, a seventeen-token prompt, greedy:

| | prompt | a token at a time | drafting | drafts accepted |
| --- | ---: | ---: | ---: | ---: |
| RX 9070 XT | 129.2 /s | 27.6 t/s | **40.1 t/s** | 70% |
| i7-9700K, 8 threads | 0.7 /s | 0.70 t/s | _refused_ | — |

**On a card with no room for it, drafting is dropped rather than paid for.** The
prediction block's weights and a shadow of every delta net's recurrence are
about four hundred mebibytes, and Qwen3.8-27B in Q4_K_M is 15.65 GiB on a card
whose heap is 15.92: with them the driver leaves the logit head in system
memory and the model reads a gigabyte across the bus for every token — 3.5
tokens a second against 26.5 with the head resident. golem asks the heap before
it uploads, drops the block when it will not fit, and says so on the standard
error. Speculation is worth about thirty per cent; a head on the bus costs
eighty-eight. The Q4_0 build is 14.94 GiB and keeps both.

Seventeen positions is a narrow pass, which is why the card reads this prompt at a hundred and thirty a second and a five-hundred-and-twelve-position one at 1283 — the [section below](#-reading-a-prompt-on-the-card) is that curve.

**Drafting is refused on the processor**, and the reason is one measurement. A speculative step costs a draft plus a pass of two columns, against the one-column pass it hopes to replace:

| cost, as a fraction of one token | on the card | on the processor |
| --- | ---: | ---: |
| the draft | 0.08 | 0.03 |
| the pass of two columns | **0.94** | **1.97** |
| drafts that must be accepted to break even | 2% | 99% |

The card reads a block's weights once whether the pass carries one column or two, so verifying two tokens costs what drawing one did. The processor at this size is bound by arithmetic rather than by reading the weights, so two columns cost two columns — and no acceptance rate can pay for that. `qwen35/cost_test.go` is where both tables come from.

## 📈 Reading a prompt on the card

The same bargain reads a prompt. A pass carries up to five hundred and twelve positions, and what one costs says how far that goes:

| positions in the pass | 1 | 16 | 32 | 64 | 128 | 256 | 512 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| the pass | 33.2ms | 57.0ms | 52.4ms | 80.2ms | 137.7ms | 219.4ms | 399.2ms |
| positions a second | 30 | 281 | 611 | 798 | 929 | 1167 | **1283** |

Thirty-two positions cost *less* than sixteen, and that is not a misprint: sixteen is the widest the mat-vec builds, and thirty-two is where the tiled products take over.

**A prompt reads at 1283 positions a second, against 62 when a pass carried two.** llama.cpp's Vulkan build reads the same file on the same card at 1211.67 ± 1.63 (`llama-bench -p 512 -r 5`). Both benchmarks do the same work — 512 tokens from an empty cache, both warm, both synchronised, load time excluded — with one difference, and it is ours: golem reads every position's hidden state back over the bus where llama.cpp keeps only the last.

### Where a pass goes

Both engines report their own GPU timers op by op — `vk.Timeline` here, `GGML_VK_PERF_LOGGER=1` there. One pass of 512 positions of the 27B, in milliseconds:

| | golem | llama.cpp |
| --- | ---: | ---: |
| feed forward — gate and up | **132.6** | 149.0 |
| feed forward — down | 97.9 | **95.6** |
| delta net — q, k, v, gate, α, β | **55.1** | 60.9 |
| attention | 32.4 | **2.1** |
| delta net — output projection (Q5_K) | **24.1** | 28.0 |
| SwiGLU | 17.1 | **11.2** |
| recurrence, with its normalisations | **15.9** | 23.2 |
| attention — q, k, v | **14.4** | 15.6 |
| everything else | **26.4** | 38.8 |
| **the pass** | **416.0** | 424.4 |

The card is busy 97.6% of that wall clock — the timer's ticks account for all but 2.4% of it — so nothing measurable is lost between dispatches, and every difference above is a kernel.

Attention is the one line where the gap is a shape rather than a margin: golem dispatches the scores, the softmax and the mix as three passes over memory, where llama.cpp fuses them and keeps the block of scores where it computed it. Sixteen of the sixty-four blocks attend at all, which is why thirty milliseconds buys the other engine so much here and why this is the next line to take.

## 🧠 Supported Models

| Family | What it runs | On CPU, vs its reference |
| --- | --- | --- |
| **Gemma 4** | E2B, 12B, 26B A4B (mixture of 128 experts). Text, Vision, Audio. | Reading a prompt ×1.24 (E2B), ×1.33 (12B), ×1.06 (26B A4B). Generating, a tie: ×1.01, ×1.04, ×1.06 — vs llama.cpp |
| **Qwen3** | Dense models, from a GGUF. | 4B: ×1.13 reading, ×1.00 generating. 0.6B: ×0.99 reading, ×0.85 generating — vs llama.cpp |
| **Qwen3.8** | 27B: Dense model featuring forty-eight gated delta nets and sixteen attentions (three to one ratio) — plus the checkpoint's own multi-token-prediction head, which drafts the second token of every pass. Text and Vision. | 0.72 t/s on an i7-9700K: a delta net rewrites a 128×128 state a head every token, and that is arithmetic no kernel makes cheaper. On a card, 27.6 a token at a time, **40.1 drafting** and **1283** reading a prompt, against llama.cpp's Vulkan build at 31.4 and 1211.7 |
| **Pocket TTS** | 12 shipped models across 6 languages, voice cloning included. | ×2.31 and ×1.69 the speed of the PyTorch reference, on the 24- and 6-layer models |

In absolute terms, on an i7-9700K with eight threads and Q4_0 weights: Gemma E2B draws 22.6 tokens a second and reads 204; the 12B, 5.0 and 42; the 26B A4B, 13.1 and 51; Qwen3 4B, 14.6 and 110. Pocket TTS speaks at ×2.94 real time in French, ×6.81 in English.

**The 0.6B is the one this engine loses**, and [`qwen/README.md`](qwen/README.md) says why: at 320 MB the weights fit close enough that the memory bus stops being the limit, and what is left is arithmetic, where llama.cpp's kernels win. This engine is built for the regime where reading the weights is the cost, and it says so where it is not.

## 🗜️ `.golem` — the engine's own weight format

golem reads GGUF like everyone else. It also writes a format of its own, in two
tiers: on the model below, T4G reads closer to the original than llama.cpp's
four-bit quantization while being 16 % smaller, and T3G undercuts llama.cpp's
three-bit quantization by 18 % while beating it on every column.

Qwen3-4B, 4088 tokens of wikitext, every row measured against the same bf16
reference:

| | size | perplexity | KL divergence | top-1 agreement |
| --- | --- | --- | --- | --- |
| bf16 | 7.5 GiB | 19.29 | — | — |
| Q4_K_M | 2.33 GiB | 20.04 | 0.0715 | 90.1 % |
| **`.golem` T4G** | **1.96 GiB** | 20.38 | **0.0526** | **90.6 %** |
| Q3_K_M | 1.93 GiB | 24.03 | 0.2453 | 79.8 % |
| **`.golem` T3G** | **1.57 GiB** | **21.11** | **0.1771** | **83.1 %** |

Read the last two rows together: **T3G is 18 % smaller than Q3_K_M and ahead of it
on every column**, by nearly three points of perplexity and a quarter of the
divergence. The evaluation is paired — both models see the same tokens in the same
eight windows — so the gap is testable, and it is not noise: t = −2.94, T3G ahead
in six windows of eight.

Two things do the work, and neither is new:

- **A trellis instead of a table.** The codebook is a state machine, not a list of
  points: a weight is twelve bits of the code stream hashed by four instructions,
  so there is nothing to look up. That is what opens the four-bit tier, where a
  lattice's table would need 493 KiB against the 32 a GPU workgroup has. The
  structure is QTIP's bitshift trellis. Nothing in a file points at a codebook —
  though the kernel, at a narrow pass, does build the whole image of that state
  machine in shared memory before it starts, because twelve bits of state is 4096
  floats and hashing each weight cost more than reading them.
- **A rotation and a salience scale**, applied per calibration site rather than
  per matrix. This is most of the format's value: dropping it and keeping only
  signs and rotation costs twenty points of perplexity on Qwen3-0.6B, 39.80
  against 60.01.

Three widths — T3G at 3.25 bits a weight, T4G at 4.19, T5G at 5.19 for the logit
head, which is worth more bits than the layers before it. A three-bit file carries
a four-bit head by default, so the 1.57 GiB above is 3.35 bits a weight overall,
not 3.25.

**What it costs is speed, and it is not a small cost.** A `.golem` file is
smaller and closer to the original than a K-quant of the same size, and it
generates more slowly than either. Qwen3-4B on an RX 9070 XT, same card, greedy:

| | size | tokens a second |
| --- | ---: | ---: |
| llama.cpp Q3_K_M | 1.93 GiB | 133.9 |
| llama.cpp Q4_K_M | 2.32 GiB | 128.7 |
| golem Q4_K_M | 2.32 GiB | 118.2 |
| golem Q4_0 | 2.11 GiB | 130.4 |
| **golem `.golem` T4G** | **1.97 GiB** | **80.3** |
| **golem `.golem` T3G** | **1.57 GiB** | **82.5** |

The reason is the codebook and it does not go away with tuning: a trellis weight
is decoded one at a time out of twelve bits of state, where a nibble format
decodes eight per instruction. About half of the trellis mat-vec is that decode.
[`compress/README.md`](compress/README.md) has the measurements, the thirteen
things that were tried to close it and the eleven that made it worse.

So the trade is memory for speed, and it is worth taking when memory is what is
short — a model that fits the card at T3G and does not at Q4_K_M generates
infinitely faster than one that does not fit — and not otherwise.

```bash
go build ./cmd/golemquant
./golemquant -model Qwen3-4B-BF16.gguf -out Qwen3-4B.golem -bits 3 -calib wiki.txt
./golem-cli -model Qwen3-4B.golem -vulkan -p "..."
```

The file is a GGUF — same container, same vocabulary, same chat template — with a
tensor type llama.cpp does not know, which is why it is named `.golem` rather than
`.gguf`: the extension is the warning that only this engine reads it. What says
the file is golem's is a `golem.format` key rather than that tensor type, which
lives in somebody else's enum; the loader checks the key, the type and the
declared block geometry, and refuses a file where any of the three disagrees. The
conversion runs on the card, and [`compress/README.md`](compress/README.md) has
the method, the measurements, a list of what was tried and did not work, and
what this format is built on.

## 👁️ Multimodal (Vision & Audio)

Provide the projector weights, and Gemma can see and hear — and Qwen3.8 can see:

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

**Qwen3.8 looks the same way**, with its own projector:

```bash
./golem-cli -model Qwen3.8-27B-Q4_0.gguf -mmproj mmproj-F16.gguf \
    -image photo.png -p "Describe this image in one sentence."
```

Its tower has no fixed input size: the picture keeps its aspect ratio, is scaled to a grid the token budget allows, and the learned position table is interpolated onto that grid — which is why an image of any shape gives a different number of rows. Every waypoint of it is held to llama.cpp's, worst gap 1.1e-4 at the patches and 6.4e-3 after all twenty-seven blocks. It sees and does not hear: the projector carries no audio encoder.

The tower goes to the card with the model. A 640×426 picture takes 33.6 seconds on an i7-9700K and **1.03 on an RX 9070 XT**, and every waypoint of the card's tower is held to llama.cpp's too — worst gap 9.9e-3 over the twenty-seven blocks, against the same 0.02 the processor's is held to. Reading the prompt and drawing the answer then run wherever the model does.

The projector is 884 MiB in fp16 and shares the card with the model, so it is resident when there is room and streamed one block at a time when there is not: 1.03 seconds an image against 1.41, rather than a fall back to the processor's thirty-three. `GOLEM_VISION_STREAM=1` forces the second path, which is how it is tested on a machine with room for the first.

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

- `cmd/golem-cli`, `cmd/golem-server`, `cmd/pocket-tts`, `cmd/golemquant`, `cmd/golemtune` — the commands.
- `engine/` — reads the architecture out of a GGUF and opens the engine that implements it.
- `gemma/`, `qwen/`, `qwen35/`, `pockettts/` — standalone engine implementations; they do not import one another. `qwen35/` is Qwen3.8: a package is named for the architecture the GGUF declares, and this checkpoint declares `general.architecture = qwen35`, as llama.cpp's own `models/qwen35.cpp` does.
- `nn/` & `vk/` — the shared kernels: quantized AVX2 and NEON, and Vulkan compute.
- `compress/` — the `.golem` format: calibration, the trellis codec, and the conversion pipeline `golemquant` drives.
- `tensors/`, `token/`, `chat/`, `sample/`, `audio/`, `imageio/` — the rest of the shared layer.
- `ref/` — what recorded each test fixture, and how to record it again.

## 🤝 Contributing

We want to make Golem the best pure-Go inference engine available. We especially need:

1. **ARM benchmarks**: the arm64 kernels are correct and tuned by nobody — written and verified under emulation, never once timed on real hardware. Run [`./benchmark-arm.sh`](benchmark-arm.sh) on Apple Silicon or Graviton and share the results.
2. **Bug reports**: if a test comparing against PyTorch or llama.cpp fails on your setup, please open an issue.
3. **Kernel optimization**: help tune the NEON kernels for ARM64.

```bash
go build ./...
go test ./... -short -p 1
```

Weights are not in this repository, and every test that needs one skips cleanly when it cannot find it — which is why the same command is safe on a machine with no models and no card, and why CI runs it.

**`-p 1` matters if you have a GPU**, and it is not a preference. `go test ./...` runs one test binary per package concurrently, and three of these packages put whole models on the card: two of them at once ask for more than a sixteen-gigabyte card has, and what comes back is `vkQueueSubmit failed (VkResult -4)` — a lost device — in whichever binary happened to be second. It reads like a driver fault, it is reproducible only by accident, and it cost an afternoon of looking in the wrong place here.

**`-short`** skips what streams a fifty-two-gigabyte checkpoint or decodes several hundred tokens on the processor. Without it the suite is hours, and `qwen35` alone wants more than `go test`'s ten-minute default: give it `-timeout 40m`.

## 📜 License & Credits

Golem is [MIT Licensed](LICENSE).

Standing on the shoulders of giants: [llama.cpp & ggml](https://github.com/ggml-org/llama.cpp), [Kyutai Pocket TTS](https://github.com/kyutai-labs/pocket-tts), [Google Gemma](https://ai.google.dev/gemma), [QTIP](https://github.com/Cornell-RelaxML/qtip) — the bitshift trellis `.golem`'s codec is built on — and [FreeToken](https://arxiv.org/abs/2608.16157), which framed the question the section above answers differently: it hides a miss behind a scheduled upload, where golem's router runs on the card and its blocks are one recorded program, so a miss is read where it lies instead. Its claim that one cache shared by every layer beats a per-layer split by ten to fifteen points does not reproduce on this checkpoint, where the two sit under a point apart.
