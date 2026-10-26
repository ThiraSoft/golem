# systemone: options weighed against llama.cpp

`cmd/golem-server`'s `/v1/systemone`, answered by the chat model, weighs each
option of a closed question by the sum of the log-probabilities of its tokens
and of the end of the turn after them (`cmd/golem-server/weigh.go`). This is
what holds that sum to llama.cpp.

`cases.tsv` holds four questions as the server writes them for Gemma 4: a
system turn, a state, the question and its options, the template's own
markers around them. Two have a short state (149 and 122 positions), two a
combat log of about a thousand, which is past E2B's sliding window of 512.
The options are five orders of one to three tokens, Yes and No, and four
gestures of which two share their first two tokens and one is the first token
of another ("heal", "heal the tank"). Each ends in `<turn|>`.

`ref/dump_logprobs.cpp` decodes each prompt and option from an empty cache on
the processor, no product handed to a card, and writes every option token's
log-softmax:

```bash
mkdir -p testdata/systemone/{e2b,12b}
build/ref/dump_logprobs "$GOLEM_MODEL"     testdata/systemone/e2b ref/systemone/cases.tsv
build/ref/dump_logprobs "$GOLEM_MODEL_12B" testdata/systemone/12b ref/systemone/cases_thought.tsv
```

E2B answers straight after the question. The 12B does not: with thinking off
it still opens a thought and closes it at once, `<|channel>thought\n<channel|>`,
each of those tokens at a log-probability of -0.000, before every answer.
Weighed straight after the question, the options of the 12B shared less than
a thousandth of the mass between them, and their distribution was that of
answers the model does not give. Past the thought they share 88 to 99.9 per
cent of it, by llama.cpp's own record. So when the likeliest token after a question's words is the
first of that empty thought, `Weigh` writes it and weighs the options past it,
for that question and the ones after it. `cases_thought.tsv` is the same
questions with the thought written, which is what the 12B is held to; the
test weighs them without it, so that `Weigh` has to decide to write it.

`TestWeighMatchesLlamaCpp` reads them back and weighs the same tokens through
`Context.Weigh` as the server does: the question's words and its options
written after the state in one pass, each option turned as if it followed the
words and blind to the others. `TestWeighFixtureTokensAreGolems` checks
that golem's tokenizer reads the prompts and the options into the tokens
llama.cpp read.

Measured 2026-10-26, llama.cpp `8fe90e1fb`, i7-9700K and RX 9070 XT:

| | worst gap in an option's log-probability | worst gap in its probability | same pick |
|---|---:|---:|---|
| E2B (QAT Q4_0), golem on the processor | 0.219 nats | 0.023 | 4 of 4 |
| E2B (QAT Q4_0), golem on the card | 0.217 nats | 0.030 | 4 of 4 |
| 12B (QAT Q4_0), golem on the card | 0.394 nats | 0.007 | 4 of 4 |

The 12B's gap was 0.456 nats with the options weighed one after the other on
the rewound cache, each through the attention of one column, and 0.489 with
that attention split. In one pass an option is answered by the tiled
attention and products, which sum the same products in other orders. None of
the three is nearer to llama.cpp by design: the gap moves by a few
hundredths with the order of the sums. On the processor the order does not
change, and E2B's gap is the same to the last digit either way.

The gap is the engine's, not the weighing's. Each option weighed alone from
an empty cache gives the same numbers as the rewound cache to the fourth
decimal, and the options in one pass give the rewound cache's bit for bit on
the processor (`gemma`'s `TestBranchesAgreeWithOneAfterAnother`), and the logits themselves are that far apart: `gemma/README.md`
puts the top 64 logits of E2B at 0.29 from llama.cpp and the 12B's at 0.86.
An option's log-probability is a difference of logits, and its largest gaps
are on unlikely options, where a few tenths of a nat move a probability of a
few per cent by less than one. The test allows 0.3 nats and 0.05 in
probability, with the same pick, on E2B, 0.6 nats and 0.05 on the 12B.

The fixtures are the model's output and are not versioned (`.gitignore`); the
tests skip until they are recorded.
