# ref/gemma — what the recorders are asked for, for Gemma 4

The recorders themselves are in [`../`](../) and know nothing about Gemma. What
is Gemma here is which waypoints matter, which prompts reach them, and which
sentences a tokenizer is likely to get wrong — plus `dump_chats.py`, the one
recorder that does not go through llama.cpp at all.

## short.run — the parity recording

The prompt "The capital of France is", every waypoint of blocks 0, 4, 5, 13,
14, 15 and 19, the final norm, the top 64 logits and sixteen greedy tokens.

Those blocks cover every kind a block can be in either checkpoint — owning a
window cache, owning a global one, reading block 13's, reading block 14's, and
the 12B's first global block, which is block 5 and computes no value projection.
They include the two sources, so that a sharing block can be tested without
running the thirteen blocks before it.

Every block's output is kept as well (`all_blocks l_out`), which is what lets a
test start any block from the reference's own input, and what makes a divergence
in the full stack say which block it began in.

Five waypoints are `optional`, because the two checkpoints disagree about them:
the keys and values of a block that shares another's cache do not exist, and
nothing per-layer exists in the 12B, which declares no per-layer width.

The same file records both checkpoints, into `testdata/gemma/layers/` and
`testdata/gemma/layers12/`.

## window.run — making the sliding window matter

A sentence repeated past 512 tokens, keeping only the last position of blocks
0, 4 and 15 and the final norm. Its only job is the window: on a short prompt a
window block and a global one attend to the same positions, and a missing mask
would go unnoticed. `min_tokens 513` is what refuses to record a run that
failed to be long enough.

## corpus.tsv — twenty-six segmentations

Escaping, runs of newlines and tabs, digits, punctuation, emoji, CJK, byte
fallback, and both sides of the special-token switch. `<turn|>` appears twice,
once as five ordinary characters and once as a special token, because that is
the pair that catches a tokenizer which ignores `parse_special`.

## dump_chats.py — recording the chat template

The one recorder that does not go through llama.cpp. The template is 18 KB of
Jinja carried in the GGUF metadata, and the question a fixture has to settle is
what *that template* renders — not what a second implementation of Jinja
believes it renders. So this script reads the template out of the file with a
small metadata reader of its own and hands it to Jinja2, which is the engine
the template was written for.

Twenty-five cases. Fourteen for a text conversation: a bare user turn, a system
or developer preamble, several turns, two consecutive assistant messages (which
the template folds into one model turn), the thinking flag, the generation
prompt withheld, content that needs trimming, a thinking channel that has to be
stripped out of a past answer, an empty message and a non-ASCII one.

Eleven more for tools: a declaration with parameters, one with none, one holding
an array and a boolean, two declarations at once, declarations beside a system
message and beside the thinking flag, a call left unanswered (which the template
ends on a bare `<|tool_response>`), a call answered, two calls answered, a call
answered and then spoken to, and a call whose arguments run through every type
the format has.

Content-part arrays — images, audio, video — are still deliberately absent:
golem's renderer covers text, and a fixture for a path that is not implemented
would only say that it is not implemented.

```bash
python3 ref/gemma/dump_chats.py "$GOLEM_MODEL"     testdata/gemma/chat/cases.json
python3 ref/gemma/dump_chats.py "$GOLEM_MODEL_12B" testdata/gemma/chat12/cases.json
```

Two files, because the two checkpoints do not carry the same template: the 12B's
closes a generation prompt with a thought channel opened and shut at once when
thinking is off, and E2B's has no such line. The fixture records which of the
two it came from under `empty_thought`, and the Go renderer is told by the
fixture rather than by the model's name.

Needs Jinja2 (`pip install jinja2`). Run by hand; these fixtures *are*
committed — a rendered template is text, not weights.
