# nomic

nomic-embed-text-v2-moe, which GGUF calls `nomic-bert-moe`: text in, one
768-wide vector out. It is the embedder ollama ships under the same name, and
`golem-server -embed` answers the endpoints a client of either speaks —
`cmd/golem-server/README.md` says which.

It is a BERT, which makes it unlike every other engine here. Nothing is
generated, so there is no cache and no logit head: a text goes through once,
every position sees every other, and the rows the last block writes are
averaged into one vector. Twelve post-norm blocks — the residual is added and
the sum is normed, twice a block — with a bias on every projection and a NeoX
rotation of the queries and keys. Every odd block's feed forward is a mixture
of eight experts, two of which answer, weighted by the router's softmax as it
stood: not renormalized over the two, because llama.cpp calls `build_moe_ffn`
with `norm_w` false for this model. The tokenizer is XLM-RoBERTa's
SentencePiece unigram, in `token/ugm`, character map and all.

## Against the reference

`ref/nomic/short.run` records every waypoint llama.cpp names, on the CPU
backend, for an f16 and a Q8_0 checkpoint; `reference_test.go` replays them.

| checkpoint | worst waypoint, relative | pooled vector, cosine |
|---|---:|---:|
| f16 | 9.5e-4 (`ffn_out-10`) | 0.99999991 |
| Q8_0 | blocks 0 and 1 bit for bit, then a couple of per cent | 0.99987 |

The Q8_0 gap is not a mistake and the test says why at length. Every
activation is quantized to eight bits on its way into a product, so an ulp
anywhere is a whole step somewhere. The norms are ggml's to the bit — the
squares summed eight at a time in SSE's order, the reciprocal of a float square
root — which is what makes the first two blocks exact; the router's softmax
takes its exponential from an AVX2 polynomial that is not `expf`, and from
block 2 the steps start to differ. llama.cpp's own f16 and Q8_0 recordings of
the same prompt are 0.99952 apart.

Against ollama itself, on two sentences through `/api/embed`, the vectors are
0.9999998 apart, with the same token counts.

One text in the benchmark's corpus lands at 0.99999 against both llama.cpp and
ollama, with the same tokens. Its router, at block 11, puts its second and third
experts 1.4e-5 apart in probability; golem's router logits differ from ggml's
by about 1e-4, and a mixture's top-k is hard, so one position is answered by
another expert. That is what a mixture does to any two implementations that do
not sum in the same order, and it is a hundred-thousandth of the cosine.

The tokenizer is held to 32 cases llama.cpp segmented (`ref/nomic/corpus.tsv`):
the newline and the tab the map turns into spaces, NFKC's ligatures, circled
digits and full-width letters, a combining accent composed, a run of unknown
characters answered by one `<unk>`, and nine scripts.

## Speed

On the CPU, since that is where ollama runs this model on this machine: an
i7-9700K, eight cores, the `performance` profile, 2026-09-14. The same
requests, through each server's HTTP API, against the same f16 file.

| positions a second | golem | llama.cpp | ollama |
|---|---:|---:|---:|
| 64 texts in one request, 3394 positions | **1837** | 1533 | 1051 |
| one text of 400 positions | **1757** | 1493 | 788 |
| 64 one-text requests, eight at a time | **1640** | 1515 | 303 |

llama.cpp is `llama-server --embeddings --device none` at `8fe90e1fb`; ollama is
the service the machine already ran. On the card, llama.cpp's Vulkan build reads
a 512-position text at 39061 positions a second — see the backlog.

A Q8_0 file is slower here than the f16 one, which is the wrong way round: it
still goes through `MatVecBatch`, 733 positions a second on the 64 texts against
llama.cpp's 1656 for the same file. The tile has not been written for it.

Where it came from, measured at each step:

- **315 → 740.** The fp16 product was 74 % of the time, in a kernel that widens
  a row for every column it meets. `nn.MatMulF16` holds four rows against three
  columns in registers instead.
- **740 → 1597.** The profile then said half the cores were spinning: the norms
  ran on one thread, and every product allocated its operand afresh. They are
  spread and kept now. The attention, which was a dot product a key, became two
  products a head on the float32 twin of the same kernel.
- **1597 → 1837.** The rounding to fp16 went through the eight-wide
  `nn.RoundHalfRange`, and the softmaxes through ggml's own exponential.
- **Concurrent requests** wait for the model and are carried by its next pass
  together — `cmd/golem-server/embeddings.go`'s batcher. Without it eight
  clients got half the rate one request of sixty-four gets.

A text's vector is the same float alone and in a batch. That is a property of
the kernels and not an accident: every entry of a product comes out of the tile
kernel, the ragged edges included, and each of its accumulators depends on its
own row and column only.

## Using it

```go
m, err := nomic.Open("nomic-embed-text-v2-moe.f16.gguf")
vecs, err := m.Embed([][]int32{m.Tokenize("search_query: what is golem?")})
nomic.Normalize(vecs[0])
```

The checkpoint wants its task prefixes, `search_query: ` and
`search_document: `, and nothing here adds them: the text is the caller's.
`Tokenize` cuts a text to the 512 positions the model was trained on and keeps
the closing `</s>`.

The tests want `GOLEM_MODEL_NOMIC` naming an f16 or a Q8_0 file; the benchmark
takes `GOLEM_NOMIC_CORPUS`, one text a line.
