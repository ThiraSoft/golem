# sample

A row of logits in, a token out.

The chain is llama.cpp's default one, in its order:

1. **top-k** keeps the `k` highest logits. With a vocabulary of 262144 and a `k`
   of 64, sorting the row would cost more than the draw it serves, so the
   selection runs through a heap of `k` entries — the root is the worst kept
   candidate, and anything that does not beat it is dropped on sight. Only the
   survivors are sorted.
2. **top-p** softmaxes the survivors and keeps the shortest prefix whose
   probabilities reach `p`, including the token that crosses the threshold. One
   candidate always survives: an empty distribution has nothing to draw from.
3. **temperature** divides what is left before the final softmax. Zero or less
   never reaches this point — `Pick` returns the highest logit straight away.
4. **the draw** walks the cumulative probabilities against one float from the
   generator.

Ties, everywhere, go to the lower identifier, which is what llama.cpp does and
what makes a greedy run reproducible.

The generator is `math/rand/v2`'s PCG, seeded from `Params.Seed`. A sampler owns
it, so two samplers in the same process do not disturb each other and a seed
plus a sequence of logit rows determines the tokens exactly.

```go
s := sample.New(sample.Params{Temperature: 1, TopK: 64, TopP: 0.95, Seed: 7})
id := s.Pick(logits)
```

`Defaults()` returns the values Gemma 4's file declares for itself; `gemma`
reads that file's own `general.sampling.*` keys and falls back to them.

What is not here: repetition and presence penalties, min-p, typical-p, locally
typical sampling, mirostat, grammars. They are additions to this chain rather
than changes to it, and nothing in golem asks for them yet.
