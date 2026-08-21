# qwen

Qwen3 dense models from a GGUF, in Go, with no cgo and nothing outside the
standard library. Checked activation by activation against llama.cpp.

```bash
GOLEM_MODEL_QWEN=Qwen3-4B-Q4_0.gguf go test ./qwen
```

## What it runs

A GGUF that declares `general.architecture = "qwen3"`. Anything else is refused
by name at load rather than half-read.

Tested on Qwen3 0.6B and 4B. Nothing in the engine is written for a particular
size: the geometry is read from the file, and the 4B's thirty-six blocks of
width 2560 load and run against the same code the 0.6B's twenty-eight of width
1024 do.

Not here: `qwen35` and its state-space blocks, the mixture-of-experts variants,
the K quantizations, the multimodal projector.

## The block, mostly in the negative

Plain pre-norm — norm, attend, add; norm, feed forward, add — with two norms a
block and the residual outside both halves. [`gemma/block.go`](../gemma/block.go)
is the file to read beside [`block.go`](block.go): Gemma norms on *both* sides
of each half with the residual outside all four, then folds in a per-layer
embedding and scales the block. None of that is here.

Gone with it: the sliding window, blocks that read another block's cache, the
key projection that stands in for a value, the logit softcap, and the embedding
scale on the way in. Their absence is not a zero to branch on — the fields are
not in the configuration.

What it has and Gemma does not is a pair of per-head norms, `attn_q_norm` and
`attn_k_norm`, applied before the rotation; the `1/sqrt(head_dim)` llama.cpp
passes into the softmax; and SiLU where Gemma has a tabulated GELU.

Sixteen query heads read eight key-value ones, two queries to a key. Heads are
128 wide against a model 1024 wide in the 0.6B, so the query projection comes
out twice the residual stream — worth knowing before writing an index.

The head is the input embedding read the other way round. A checkpoint carrying
its own `output.weight` is refused rather than having two hundred megabytes of
it quietly ignored.

## What it is checked against

`ref/qwen/*.run` describe what llama.cpp records; `ref/README.md` says how to
run the recorders. The waypoint names come from `llama.cpp/src/models/qwen3.cpp`
rather than from a guess, and three of them are not what a reader of
`ref/gemma/short.run` would predict — there is no `inp_scaled`, the queries are
recorded on both sides of the rotation, and there is no waypoint between the
attention and its output projection.

```bash
cmake -S ref -B build/ref -DLLAMA_DIR=/path/to/llama.cpp && cmake --build build/ref -j8
mkdir -p testdata/qwen/{layers,long,layers_q4,tokenizer}

build/ref/dump_layers "$BF16"    testdata/qwen/layers    ref/qwen/short.run
build/ref/dump_layers "$BF16"    testdata/qwen/long      ref/qwen/long.run
build/ref/dump_layers "$Q4_PURE" testdata/qwen/layers_q4 ref/qwen/short.run
build/ref/dump_tokens "$BF16"    testdata/qwen/tokenizer ref/qwen/corpus.tsv
```

The bfloat16 checkpoint comes first on purpose. The kernels and the architecture
are two independent places for a mistake to hide, and bfloat16 removes one of
them; the Q4_0 run then proves the other half. Make that file with
`llama-quantize --pure` — the Q4_0 build published alongside the bfloat16 one
stores `ffn_down` as Q4_1, which `tensors/gguf.go` refuses and should go on
refusing.

Where it stands, on the 0.6B against a five-token prompt:

| | worst relative gap |
|---|---|
| `Qcur`, `Kcur`, `Vcur`, all four sampled blocks | 1e-5 |
| `ffn_out` | 1.4e-6 |
| `l_out-27`, bfloat16 | 4.2e-3 |
| `l_out-27`, Q4_0 | 2.7e-2 |
| top 64 logits | 0.26 absolute |

Two assertions came out exactly zero: a batch of positions and the same
positions one at a time give bit-identical hidden states, and a `Reset` followed
by the same prompt does too. And the sixteen tokens llama.cpp draws greedily
after the prompt are the sixteen this engine draws, from bfloat16 and from Q4_0
alike — which is the assertion the activation comparisons cannot make.

The Q4_0 stack drifts further than the bfloat16 one because every product there
quantizes its activation to Q8_0 in blocks of thirty-two, and a value near a
step rounds the other way from llama.cpp's now and then. That path is chaotic
rather than biased: `qwen/quantized_test.go` says what was measured about it and
why a change in that figure is not by itself a regression.

## Speed

On an i7-9700K, eight threads, pure Q4_0 weights, llama.cpp build 9603 held to
the CPU with `-dev none -ngl 0`.

| | | llama.cpp | golem | golem ÷ llama.cpp |
|---|---|---:|---:|---|
| **4B** | prompt, 64 tokens | 97.0 t/s | 109.9 t/s | ×1.13 |
| | generation | 14.56 t/s | 14.62 t/s | ×1.00 |
| **0.6B** | prompt, 64 tokens | 739 t/s | 729 t/s | ×0.99 |
| | generation | 93.9 t/s | 79.5 t/s | ×0.85 |

```bash
GOLEM_MODEL_QWEN_Q4="$Q4_PURE" go test ./qwen -run XXX -bench . -benchtime 8s -count 3
llama-bench -m "$Q4_PURE" -dev none -ngl 0 -t 8 -p 64 -n 64 -r 3
```

The 0.6B is the one this engine loses, and it loses it for a reason worth
stating. At 320 MB the weights sit close enough that the memory bus stops being
the limit; what is left is arithmetic and the cost of eight cores meeting at a
barrier, and llama.cpp's kernels win that. At 2.3 GB the bus is the limit again,
which is the regime this engine is built for and where it is ahead on both
figures. Generation of the 4B is a tie to three digits, because there both
engines are doing the same thing: reading 2.3 GB per token.

Reading a prompt went 301 → 579 → 699 → 729 tokens a second over four changes,
and none of them touched a kernel:

- `Cache.Reset` cleared every position the cache had been *sized* for — 940 MB
  for a context of 4096 — rather than the ones a conversation wrote.
- `perPosition` was guessed at four where `gemma/block.go` says twenty-four,
  which put three passes of every block just under the threshold below which
  `nn.InParallel` declines to split. Eighty-four passes of a prompt ran on one
  core while seven spun. Parallel efficiency went from 71% to 88%.
- `nn.SwiGLURange` replaced an exact float64 exponential and a separate multiply
  with `ggml_vec_swiglu_f32`: ggml's polynomial and its fusion, 1.71× on the
  activation.

The scaling curve is now ×1.88 at two cores, ×3.65 at four, ×7.03 at eight.
