# ref — recording what the reference computes

Development tools. They run a model under llama.cpp, or under the engine a
model's template was written for, and write down what it computes. The Go tests
read those recordings back, so llama.cpp is needed here and nowhere else.

The three C++ recorders are model-neutral, and are meant to stay that way.
llama.cpp names the waypoints of its graph the same whatever the architecture,
so what has to change between one model and the next is not the recorder: it is
which waypoints to keep, on which prompt, with which tokenizer flags. That lives
in a file under `ref/<model>/`, named on the command line.

**A new architecture adds a directory here, not a recorder.**

| | pins | takes |
|---|---|---|
| `dump_layers.cpp` | the engine | a model, an output directory, a `.run` file |
| `dump_tokens.cpp` | the tokenizer | a model, an output directory, a `.tsv` corpus |
| `dump_quants.cpp` | the kernels | a model, an output directory |

The per-model files are in [`gemma/`](gemma/), alongside the one recorder that
is not shared — `dump_chats.py`, which goes through Jinja rather than llama.cpp
because the question a chat fixture settles is what *that template* renders.

## Building

`LLAMA_DIR` must point at a llama.cpp checkout that has already been built: the
tools link against `llama`, `ggml`, `ggml-base` and `ggml-cpu` in its
`build/bin`.

```bash
cmake -S ref -B build/ref -DLLAMA_DIR=/path/to/llama.cpp
cmake --build build/ref -j8
```

## Regenerating every fixture

From the repository root, with `GOLEM_MODEL` and `GOLEM_MODEL_12B` set:

```bash
mkdir -p testdata/gemma/{layers,layers12,window,tokenizer,quants}

build/ref/dump_layers "$GOLEM_MODEL"     testdata/gemma/layers    ref/gemma/short.run
build/ref/dump_layers "$GOLEM_MODEL"     testdata/gemma/window    ref/gemma/window.run
build/ref/dump_layers "$GOLEM_MODEL_12B" testdata/gemma/layers12  ref/gemma/short.run
build/ref/dump_tokens "$GOLEM_MODEL"     testdata/gemma/tokenizer ref/gemma/corpus.tsv
build/ref/dump_quants "$GOLEM_MODEL"     testdata/gemma/quants
```

## dump_layers — recording a forward pass

It runs the model on a fixed prompt with a scheduler callback attached, and
writes out every intermediate llama.cpp names, as float32 in ggml's own layout
— dimension 0 fastest. Both the weights and the graph are kept on the CPU, and
flash attention is disabled: the reference is ggml's own arithmetic, and a
fused attention kernel has no waypoints to record.

The tool fails if a name it was asked for never appeared in the graph. That is
deliberate: llama.cpp renames waypoints from time to time, and a silently
missing fixture is a test that passes without testing anything. A name the run
file declares `optional` is the exception, for the waypoints two checkpoints of
one architecture disagree about. `index.json` names the model each recording
came from, so a fixture cannot be mistaken for another checkpoint's.

### The run file

One directive per line, `#` starts a comment, a blank line is ignored.

| directive | |
|---|---|
| `prompt <text>` | the prompt; `\n` is a newline, `\s` a space, `\\` a backslash. A leading or trailing space must be written `\s`, because the line is trimmed. |
| `repeat <n>` | repeat the prompt n times (default 1) |
| `label <text>` | what `index.json` records as `"prompt"`; defaults to the prompt |
| `add_special <0\|1>` | `llama_tokenize`'s `add_special` (default 1) |
| `min_tokens <n>` | fail if the run tokenizes to fewer (default 0) |
| `blocks <i> <i> …` | the block indices the per-block names apply to |
| `all_blocks <name>` | a name recorded for every block, `0..n_layer-1` |
| `require <name>` | a per-block name that must appear |
| `optional <name>` | a per-block name that may be absent |
| `global <name>` | a whole-model name that must appear |
| `global_opt <name>` | a whole-model name that may be absent |
| `last_column <0\|1>` | keep only the last column of each recording |

A per-block name is expanded to `<name>-<index>` for each block listed.

## dump_tokens — recording a segmentation

It loads the vocabulary alone — `vocab_only` skips the weights — and writes,
for each case of a corpus, the identifiers llama.cpp produces and the text it
writes back from them.

Each case carries the flags it was recorded under, because `add_special` and
`parse_special` change the answer: the same sentence tokenizes differently with
a special token treated as special and as its own characters.

### The corpus

Four tab-separated columns, `#` starts a comment:

```
name <TAB> text <TAB> add_special <TAB> parse_special
```

The text column uses `\n`, `\t`, `\r`, `\s`, `\\` and `\xNN`, because a
tab-separated file cannot carry a literal tab, a line cannot carry a literal
newline, and a trailing space would not survive being read.

## dump_quants — recording what ggml makes of quantized bytes

It takes real quantized tensors out of a GGUF and writes down both the raw
bytes and what ggml itself computes from them, so the Go kernels are checked
against ggml rather than against a second implementation of the same idea.

Three cases, into `testdata/gemma/quants/`:

- `q4_0_matvec` — a 64×1536 slab of `blk.0.attn_q.weight`, a deterministic
  activation, and the matrix-vector product ggml computes from them. ggml
  quantizes the activation to Q8_0 internally, exactly as the Go kernel does.
- `q4_0_dequant` — four rows of the same tensor, dequantized by ggml.
- `q6_k_dequant` — four rows of `token_embd.weight`, dequantized by ggml.

`index.json` describes each case; `*.w.bin` holds the quantized bytes as they
sit in the GGUF, `*.x.bin` and `*.y.bin` little-endian float32.

## The rule

The recordings are not committed — they are the models' output, and this
repository does not redistribute it — so these tools are what put them on a
machine. They are not run at test time, and the Go tests never need llama.cpp:
each one skips until its fixture is there.

Regenerating a fixture is a deliberate act, and a fixture that changes means
the reference changed — which is exactly what should require a commit of its
own.
