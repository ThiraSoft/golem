# grammar

What a model may say next, and what it may not.

A grammar is written in GBNF, llama.cpp's notation. `Parse` reads it into flat
rules; `New` starts an automaton at `root`; `Allows` says whether a token may
come next and `Accept` advances over one that was drawn. `sample.Sampler` holds
one of these and asks. Nothing here reads a logit.

```go
rules, err := grammar.Parse(`root ::= "{" [0-9]+ "}"`)
toks := grammar.NewTokens(vocabSize, piece, isEOG) // once per model
g := grammar.New(rules, toks)

g.Allows(id) // may this token come next
g.Accept(id) // it was drawn
g.Done()     // the grammar is satisfied and the turn may end
```

This is a port of `src/llama-grammar.cpp`, and the fidelity is deliberate: the
identifiers rules are given, the names generated for a group or a repetition and
the order elements are emitted in are all observable, and `grammar/schema`'s
tests compare golem's output against llama.cpp's line for line.

What the port changes is one thing. A stack position is `{rule, index}` where
llama.cpp carries a pointer into the rules — `Allows` copies the whole state for
every candidate it tests, and two integers are what make that copy cheap.

Three details are the difference between a grammar that works and one that
looks like it does:

- **A piece is not always valid UTF-8.** A byte-level BPE splits a character
  across two tokens whenever it pleases, so the bytes are kept raw and the half
  character is carried between tokens and matched against the range it could
  still complete to. Without it every accent and every ideogram is refused.
- **A token that prints nothing is refused**, as llama.cpp does. A model let
  loose on control tokens loops on tokens nobody can see.
- **An end-of-turn token is refused while no stack is empty.** A draw at
  temperature would otherwise close the answer in the middle of an object, and
  what comes back is a document with its braces open — a failure that costs the
  most and shows the least.

Measured on an i7-9700K over a synthetic vocabulary of 262144 pieces, just
inside an object — the state that allows the least:

| | |
|---|---|
| one question, which is what nearly every token costs | 376 ns |
| the whole vocabulary swept, lead byte first | 0.72 ms |
| the whole vocabulary swept, every token walking the stacks | 40.7 ms |

`grammar/bench_test.go` is where those come from. The last row is what a
straight port of llama.cpp's own loop costs in Go, and it is why the table
below exists.

`FirstBytes` is the guard under a sweep: the lead bytes any allowed token could
start with, as one table of 256, so a state that refuses nearly everything — the
start of an object, just after a comma — costs one comparison a token rather
than a walk of the stacks. It is a superset by construction, and a test holds it
to that.

What is not here: the two token elements of GBNF (`<[id]>` and its negation),
lazy triggers for tool calls, and llguidance. The first is the only part of a
grammar that depends on which vocabulary it was written for; the others are
llama.cpp's answers to questions golem has not asked.
