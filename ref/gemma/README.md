# dump_quants — recording what ggml computes

A development tool. It opens a Gemma GGUF, takes real quantized tensors out of
it, and writes down both the raw quantized bytes and what ggml itself makes of
them. The Go kernels are then checked against those recordings, so llama.cpp is
needed here and nowhere else.

Three cases are written into `testdata/gemma/quants/`:

- `q4_0_matvec` — a 64×1536 slab of `blk.0.attn_q.weight`, a deterministic
  activation, and the matrix-vector product ggml computes from them. ggml
  quantizes the activation to Q8_0 internally, exactly as the Go kernel does.
- `q4_0_dequant` — four rows of the same tensor, dequantized by ggml.
- `q6_k_dequant` — four rows of `token_embd.weight`, dequantized by ggml.

`index.json` describes each case; `*.w.bin` holds the quantized bytes as they
sit in the GGUF, `*.x.bin` and `*.y.bin` little-endian float32.

## Building and running

`$LLAMA_DIR` is a built llama.cpp checkout and `$GOLEM_MODEL` a Gemma 4 GGUF.

```bash
cmake -B build -DLLAMA_DIR="$LLAMA_DIR"
cmake --build build
mkdir -p ../../testdata/gemma/quants
./build/dump_quants \
  "$GOLEM_MODEL" \
  ../../testdata/gemma/quants
```

`LLAMA_DIR` must point at a llama.cpp checkout that has already been built:
the tool links against `ggml`, `ggml-base` and `ggml-cpu` in its `build/bin`.

## The rule

The fixtures are not committed — they are Gemma's output, and this repository
does not redistribute it — so this tool is what puts them on a machine. It is
not run at test time, and the Go tests never need llama.cpp. Regenerating them is a deliberate act, and a fixture that
changes means the reference changed — which is exactly what should require a
commit of its own.

## dump_layers — recording a forward pass

`dump_quants` pins the kernels; `dump_layers` pins the engine. It runs Gemma 4
E2B on a fixed prompt with a scheduler callback attached, and writes out every
intermediate llama.cpp names, as float32 in ggml's own layout — dimension 0
fastest. Both the weights and the graph are kept on the CPU, and flash
attention is disabled: the reference is ggml's own arithmetic, and a fused
attention kernel has no waypoints to record.

Two runs:

- `short` into `testdata/gemma/layers/`: the prompt "The capital of France is",
  every waypoint of blocks 0, 4, 13, 14, 15 and 19, the final norm, the top 64
  logits and sixteen greedy tokens. Those six blocks cover the four kinds a
  block can be — owning a window cache, owning a global one, reading block
  13's, reading block 14's — and include the two sources, so that a sharing
  block can be tested without running the thirteen blocks before it. Every
  block's output is kept as well, which is what lets a test start any block
  from the reference's own input. Blocks 15 and 19 have no key or value
  waypoints, because they compute neither.
- `window` into `testdata/gemma/window/`: a sentence repeated past 512 tokens,
  keeping only the last position of blocks 0, 4 and 15 and the final norm. Its
  only job is to make the sliding window matter — on a short prompt a window
  block and a global one attend to the same positions, and a missing mask would
  go unnoticed.

```bash
cmake -B build -DLLAMA_DIR="$LLAMA_DIR"
cmake --build build
MODEL=/path/to/gemma-4-E2B-it-QAT-Q4_0.gguf
mkdir -p ../../testdata/gemma/layers ../../testdata/gemma/window
./build/dump_layers "$MODEL" ../../testdata/gemma/layers  short
./build/dump_layers "$MODEL" ../../testdata/gemma/window  window
```

The tool fails if a name it was asked for never appeared in the graph. That is
deliberate: llama.cpp renames waypoints from time to time, and a silently
missing fixture is a test that passes without testing anything.

## dump_tokens — recording a segmentation

`dump_tokens` pins the tokenizer. It loads the vocabulary alone — `vocab_only`
skips the weights — and writes, for a corpus of twenty-six cases, the
identifiers llama.cpp produces and the text it writes back from them.

Each case carries the flags it was recorded under, because `add_special` and
`parse_special` change the answer: the same sentence tokenizes differently with
`<turn|>` treated as a special token and as five ordinary characters. The
corpus covers escaping, runs of newlines and tabs, digits, punctuation, emoji,
CJK, byte fallback, and both sides of the special-token switch.

```bash
cmake -B build -DLLAMA_DIR="$LLAMA_DIR"
cmake --build build --target dump_tokens
mkdir -p ../../testdata/gemma/tokenizer
./build/dump_tokens "$MODEL" ../../testdata/gemma/tokenizer
```

## dump_chats.py — recording the chat template

The one recorder here that does not go through llama.cpp. The template is 18 KB
of Jinja carried in the GGUF metadata, and the question a fixture has to settle
is what *that template* renders — not what a second implementation of Jinja
believes it renders. So this script reads the template out of the file with a
small metadata reader of its own and hands it to Jinja2, which is the engine
the template was written for.

Fourteen cases: a bare user turn, a system or developer preamble, several turns,
two consecutive assistant messages (which the template folds into one model
turn), the thinking flag, the generation prompt withheld, content that needs
trimming, a thinking channel that has to be stripped out of a past answer,
an empty message and a non-ASCII one.

Tools, tool calls and content-part arrays are deliberately absent: golem's
renderer covers a text conversation, and a fixture for a path that is not
implemented would only say that it is not implemented.

```bash
python3 dump_chats.py "$MODEL" ../../testdata/gemma/chat/cases.json
```

Needs Jinja2 (`pip install jinja2`). Run by hand; the fixture is committed —
a rendered template is text, not weights.
