# token/bpe — Gemma 4's tokenizer

The file says `tokenizer.ggml.model = "gemma4"`, and that is a shape of its
own: BPE merges over raw UTF-8, wearing SentencePiece's clothes. Spaces become
`U+2581` before anything else happens, so the pieces keep track of word
boundaries; but there is no unigram lattice here, no GPT-2 byte alphabet, and
no word-level pre-splitting. The 262144 pieces and the 514906 merges both come
out of the GGUF metadata — there is no second file to find or to ship.

## What encoding does

1. The special tokens are cut out of the text first, longest first, so that a
   short one cannot bite into a long one. With `parseSpecial` off, only the
   user-defined tokens do this; the control tokens stay literal text.
2. Every ASCII space becomes `U+2581`. Nothing else is normalized: no NFKC, no
   collapsing of repeats, no leading space — the file sets `add_space_prefix`
   to false and it is honoured.
3. The text is cut into runs of newlines and runs of everything else. That is
   the only pre-splitting, and it exists because a merge may not straddle a
   newline.
4. Each run is merged: every adjacent pair with a known rank waits in a
   priority queue, lowest rank first and leftmost on a tie, and each pop merges
   one pair and queues the two new neighbours it creates.
5. What survives is looked up by text. A symbol the vocabulary does not know is
   re-emitted one byte at a time through the `<0xNN>` pieces, so no character
   can make a sentence unencodable.

A run made only of newlines that already exists as a piece is taken whole,
without merging. Rebuilding it from single newlines does not always reach the
same answer, and llama.cpp special-cases it for the same reason.

## What it shares with token/sentencepiece

The character `U+2581`, and nothing else. They are neighbours because they play
the same role, not because they have an implementation in common.

## The reference

`ref/gemma/dump_tokens.cpp` records what llama.cpp produces on a corpus of
twenty-six cases — escaping, runs of newlines and tabs, digits, punctuation,
emoji, CJK, byte fallback, and both sides of the `parse_special` switch — into
`testdata/gemma/tokenizer/cases.json`. The fixture is committed; the tests need
neither llama.cpp nor any C++.

## What is not here

The chat template, and sampling. The template is 18 KB of Jinja in the GGUF
metadata and gets its own reimplementation; this package's job ends at turning
a string into identifiers.
