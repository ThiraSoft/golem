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
s := sample.New(sample.Params{Temperature: 1, TopK: 64, TopP: 0.95, Seed: 7,
	PenaltyLastN: 64, PenaltyRepeat: 1.1})
s.Seed(prompt)         // the penalties read the conversation, not only the answer
s.Constrain(g)         // optional: a grammar the draw has to stay inside
id := s.Pick(logits)
```

`Defaults()` returns the values Gemma 4's file declares for itself; `gemma`
reads that file's own `general.sampling.*` keys and falls back to them.

Two things sit around that chain.

**The penalties.** `PenaltyRepeat`, `PenaltyFreq` and `PenaltyPresent` over the
last `PenaltyLastN` identifiers, applied before top-k, with llama.cpp's own
arithmetic — a logit above zero is divided, one at or below it is multiplied,
which is llama.cpp's fix for a division that would make an unlikely token
likelier. `Seed` puts the prompt into that window, as llama-server does before
its first draw, and `Pick` puts whatever it draws there.

The window is what makes the shortcut exact. A penalty can only lower a logit,
so the k best penalised candidates are among the k+u best raw ones, u being the
identifiers in the window: the chain takes k+u through the heap, penalises
those, and cuts back to k without ever walking the row. Weights that could
raise a logit — a repeat below one, a negative frequency or presence — break
that argument, and the sampler notices and penalises a copy of the row instead.

**The constraint.** `Constrain` hands the sampler something that says which
tokens may come next; `grammar.Grammar` is what that is in practice. The chain
draws first and asks about the one token it drew, and only a refusal costs
anything: then it filters a prefix of the sorted row, widening it only while
too few candidates survive, and tests a token's lead byte against the grammar's
own table before asking the grammar anything.

What is still not here: min-p, typical-p, locally typical sampling, mirostat,
DRY. They are additions to this chain rather than changes to it, and nothing in
golem asks for them yet.
