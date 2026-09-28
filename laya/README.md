# laya

Laya, the decision model Convai Innovations published under Apache 2.0 as
[convaiinnovations/laya](https://huggingface.co/convaiinnovations/laya), the
open counterpart of TypeSafe's Jev. It does not generate. A state goes in with
typed questions, and every option of every question comes back with a
calibrated probability, in one pass:

- **choice**: pick one of several options
- **score**: the expected level on an ordered scale, with the distribution
- **noul**: the probability that a statement holds

It is ModernBERT-large with a small head. Each question is its own sequence,
`[CLS] question [SEP] [MASK] option [MASK] option … [SEP] state [SEP]`, and
what the head writes at an option's `[MASK]` is scored into that option's
logit. The logits of one question are divided by a temperature fitted after
training and put through a softmax. The encoder is twenty-eight pre-norm
blocks without biases, a gated GELU on the error function, a NeoX rotation
whose base is 160000 in the global blocks and 10000 in the others, and two
blocks in three that see only sixty-four positions either side. The head is
two of PyTorch's `TransformerEncoderLayer`, pre-norm, with biases and a ReLU.

The checkpoint is read as Hugging Face ships it: `model.safetensors`, the two
configurations and `tokenizer/`, fp16, no conversion. All three checkpoints of
the repository open and pass the same tests against their own recording:

- `laya` (the root of the repository), English, ModernBERT-large, 512 positions
- `typed-decisions`, the same architecture fine-tuned for agent pipelines
- `multilingual`, mmBERT-base (twenty-two blocks of 768, 1024 positions, a
  vocabulary of 256000 across more than a hundred languages)

`tokenizer.json` says which tokenizer to build. The English ones are
ModernBERT's byte-level BPE with GPT-2's pre-tokenizer, read by
`token/bytebpe`. The multilingual one is mmBERT's SentencePiece-style BPE with
byte fallback, read by `token/bpe`: spaces become U+2581, one is put in front
of the text, and the text is cut before each of them, so no merge crosses a
word. The special tokens (`[CLS]` or `<bos>`, and so on) are read from
`tokenizer_config.json`.

## Against the reference

`ref/laya/dump.py` runs the checkpoint's own `rl_common.py` and
`rl_agent_api.py` in PyTorch, on the processor in float32, and records every
block of nine questions over three states: four about an email of about eighty
positions, two about a ticket of about two hundred and forty, which is past the
local window, and three about an email in French. `reference_test.go` replays them.

| | worst waypoint, relative | answers |
|---|---:|---|
| processor | 5.3e-5 | the same as `system_one` to its four decimals, give or take the last rounding |
| card | 5.8e-4 | within 5e-4 of `system_one` |

Those are the English checkpoint's. The typed one's worst are 8.5e-5 and
5.0e-5, the multilingual one's 4.3e-5 and 5.5e-5.

The sequences themselves (the state serialized as Python's `json.dumps` writes
it, the options rendered, the budget cut, the tokens) are identical to the
identifier. The tokenizer is also held to twenty texts Hugging Face segmented, five of them French.

The card's products run on the matrix cores, and an fp16 operand was not good
enough. ModernBERT grows activations in the thousands at a few positions from
its nineteenth block, and whether a position does is a threshold: with the
operand rounded to fp16, position 209 of the long fixture crossed it and ended
half a unit away from PyTorch at the last block. So every operand is carried
as two fp16 planes, its high part and the fp16 of what is left, and both meet
the weights (`vk/shaders/laya_mm.comp`). The kernel that writes an operand
(the norm, the attention, the gated GELU in the up projection's epilogue)
writes it split once. The high part is cut by a mask on the bits rather than
by a conversion, because RADV folds a conversion to fp16 and back into
nothing, which leaves the remainder zero and the result bit for bit the
rounded one. The matrix cores' own float32 accumulation was not good enough
either: over a whole shared dimension it moved the long fixture's worst
waypoint from 5e-4 to 1.6e-3, so the product sums chunks of 256 in
accumulators of their own and adds them in order, at a fifth of its speed.
Dropping the split from block 20 on passed the English fixtures (tried with
the kernels this replaced); it is not done, because block 19 is where this
fixture's position crosses, not where every input's does. The attention stays in float32
(`shaders/laya_attn.comp`): a kernel that rounds the queries and keys to fp16
moved that position by fifteen per cent.

A card without matrix cores runs the same planes through
`shaders/laya_mm_scalar.comp` in float32; forced onto this card it passes the
same test (worst waypoint 3.0e-4).

On the processor the matrices are laid out once, on the first pass, in panels
of sixteen rows (`nn.PackF16`, a second copy of the weights the card never
needs), and multiplied by an AVX2 tile of sixteen outputs by six positions
that widens each weight once for six positions (`nn/gemm_f16.go`). The gated
GELU and the attention's softmax run on float32 kernels of their own
(`nn/gelu_erf.go`): Numerical Recipes' erfc and an exponential by range
reduction, within a few parts in ten million of the float64 functions, where
`math.Erf` cost a tenth of a pass.

## Speed

`BenchmarkDecide`, on an i7-9700K and an RX 9070 XT, 2026-09-28, against the
checkpoint's own `system_one` in PyTorch 2.11 (ROCm 7.2) on the same machine
in the same sitting, `ref/laya/bench.py`. PyTorch runs the card under bf16
autocast, as `system_one` does on a GPU. golem's figure is the median of three
runs of `-benchtime 30x` on the card, `10x` on the processor.

English checkpoint (`laya`, ModernBERT-large):

| one request | golem, processor | PyTorch, processor | golem, card | PyTorch, card |
|---|---:|---:|---:|---:|
| 1 question, 82 positions | **90 ms** | 185 ms | **7.8 ms** | 19.9 ms |
| 4 questions, 325 positions | **342 ms** | 419 ms | **11.0 ms** | 22.1 ms |
| 32 questions, 2600 positions | **2.77 s** | 4.16 s | **51 ms** | 56 ms |

Multilingual checkpoint (`multilingual`, mmBERT-base):

| one request | golem, processor | PyTorch, processor | golem, card | PyTorch, card |
|---|---:|---:|---:|---:|
| 1 question | **35.6 ms** | 50.1 ms | **5.1 ms** | 16.6 ms |
| 4 questions | **130 ms** | 145 ms | **8.7 ms** | 17.9 ms |
| 32 questions | **1.03 s** | 1.48 s | **23 ms** | 35.8 ms |

golem wins every row, but not by much everywhere, and the margins move:

- 32 questions on the card, English: PyTorch measured 72 ms earlier the same
  day and 56 ms here, with nothing changed on its side. The card's clocks
  follow how busy it is kept: with a few milliseconds of host work between
  two passes, the same kernels took up to two thirds longer than submitted
  back to back.
- 4 questions on the processor, multilingual: one run in three came out at
  141 ms against PyTorch's 145. The processor's figures swing by five to ten
  per cent from run to run on this machine, both sides.
- The card does twice the arithmetic PyTorch does, for the split operands,
  and its products give up a fifth of their speed to the chunked sums. That
  is the English 32-question row's thin margin, and where it would go first.

## Using it

```go
m, err := laya.Open("path/to/convaiinnovations/laya")
m.UseVulkan() // optional
qs, err := laya.ParseQuestions(json.RawMessage(`{
    "department": {"type": "choice", "instructions": "Which team should handle this email?",
                   "criteria": {"billing": "payments, refunds", "support": "technical problems"}},
    "spam": {"type": "noul", "instructions": "This email is spam."}}`))
res, err := m.Decide(ctx, json.RawMessage(`{"subject": "Refund not received", "body": "…"}`), qs)
```

`res.Answers` are in the order of the questions; `json.Marshal(res)` writes
them as Jev and `system_one` do. `cmd/laya` is the same from the command
line:

```bash
laya -model path/to/laya [-vulkan] [-stats] request.json
```

where the request is `{"state": …, "questions": {…}}`. A state that is a JSON
string is read as the text it holds. `golem-server -laya DIR` answers the same
request on `POST /v1/systemone`, Jev's endpoint and laya-serve's.

The tests want `GOLEM_MODEL_LAYA` naming the checkpoint directory and the
fixtures `ref/laya/dump.py` writes into `testdata/laya`.
