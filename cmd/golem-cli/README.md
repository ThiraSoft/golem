# golem-cli

A conversation with Gemma 4, from a GGUF, on the CPU.

```bash
go build ./cmd/golem-cli
./golem-cli -model gemma-4-E2B-it-QAT-Q4_0.gguf
./golem-cli -model … -p "Explain what a mutex is, in one sentence." -temp 0 -stats
```

`-p` answers once and exits. Without it, turns are read from the terminal until
end of file; `/reset` forgets the conversation and starts a new one against the
same mapped weights.

The sampling flags — `-temp`, `-top-k`, `-top-p` — default to the values the
model file declares for itself, which for E2B are 1, 64 and 0.95. A temperature
of zero is greedy. `-seed` fixes the draw; left at zero, each run differs.

## The context is built once

The template's output grows by appending, so a turn encodes only what is new:
the closing of the model's last turn, the user's message, and the header of the
next answer. The cache is never rebuilt and positions run on across turns —
re-rendering the whole conversation each turn would re-read three gigabytes of
weights for every token already spoken.

One consequence is worth stating: the token that ends a turn is drawn but never
fed. Feeding it would cost a forward pass for a marker the next encoding carries
anyway, so `<turn|>\n` opens the following prompt instead.

`session.go` holds that loop behind two small interfaces — one for the engine,
one for the vocabulary — so it is tested against a scripted model and a
vocabulary of whole words, with no weights on the machine.
