# golem-cli

A conversation with a GGUF model, on the CPU. Gemma 4 or Qwen3: the file
declares its own architecture and `engine/` opens whichever implements it, so
this command names neither.

```bash
go build ./cmd/golem-cli
./golem-cli -model gemma-4-E2B-it-QAT-Q4_0.gguf
./golem-cli -model Qwen3-4B-Q4_0.gguf
./golem-cli -model … -p "Explain what a mutex is, in one sentence." -temp 0 -stats
```

`-p` answers once and exits. Without it, turns are read from the terminal until
end of file; `/reset` forgets the conversation and starts a new one against the
same mapped weights.

The sampling flags — `-temp`, `-top-k`, `-top-p` — default to the values the
model file declares for itself, which for E2B are 1, 64 and 0.95. A temperature
of zero is greedy. `-seed` fixes the draw; left at zero, each run differs.

## Rendered every turn, fed once

Every turn re-renders the whole conversation, because the template is the only
thing that knows how to write one and no two checkpoints write it the same way.
The cache is not rebuilt with it: the new render is encoded, compared against
the tokens the cache already holds, and only the tail that differs is fed.

That tail is small and stays small. Measured on an eight-thread i7-9700K, a
second turn and a tenth cost the same: fourteen tokens on Gemma, nineteen on
Qwen. The extra five are Qwen's empty `<think></think>` block, which lives in
the generation prompt and which the next render replaces with the answer — so
the divergence sits at the end of the context and does not walk back into it.

One consequence is worth stating: the token that ends a turn is drawn but never
fed. Feeding it would cost a forward pass for a marker the next render carries
anyway, and the prefix comparison puts it in at the right position.

`session.go` holds that loop behind three small interfaces — the engine, the
vocabulary, the template — so it is tested against a scripted model, a
vocabulary of whole words and a template of whole words, with no weights on the
machine and no engine named.
