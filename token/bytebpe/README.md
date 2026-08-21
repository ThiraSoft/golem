# bytebpe

The byte-level BPE tokenizer, the kind GPT-2 introduced and Qwen uses. Reads
its vocabulary out of a GGUF that declares `tokenizer.ggml.model = "gpt2"`.

`token/bpe` is the other one, for Gemma 4. The two share their middle — the
merge loop, in [`token/merge`](../merge/) — and differ at both ends.

|  | `bpe` (gemma4) | `bytebpe` (qwen2) |
|---|---|---|
| before merging | spaces become U+2581; cuts on newlines only | all 256 bytes get printable stand-ins; cuts into words |
| after merging | an unknown symbol falls back to `<0xNN>` pieces | every byte is a piece, so nothing can be unknown |

## The alphabet

A vocabulary is a list of strings and a byte-level tokenizer must see all 256
bytes, so the 68 that are not printable are moved to U+0100 and up. A space is
written `Ġ`, a newline `Ċ`. This is not a convention chosen here: it is the one
the vocabulary in the file was built with, and a different one would make every
piece miss. A test walks every normal piece in the file back through the
mapping, which is what says the table is right rather than plausible.

## The pre-tokenizer

`tokenizer.ggml.pre` names the rules; `qwen2` is the set implemented here. Its
regex is in llama.cpp's source and is not what llama.cpp runs:

```
(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
```

`\s+(?!\S)` needs a lookahead, which neither `std::regex` nor Go's RE2 has, so
llama.cpp hand-writes the scan in `unicode_regex_split_custom_qwen2`. So does
`split.go`, transcribed from that function rather than from the regex — where
the two could disagree, the function is what the vocabulary was built against.

Three rules are worth reading twice:

- **Rule 3 is one number**, with no quantifier: `1234` is four words. The
  llama3 rules say `\p{N}{1,3}` and belong to a different tokenizer.
- **Rule 2 takes its optional leading character whatever it is**, so a space or
  a comma before a letter joins the word: `" world"` and `",b"` are each one.
- **Rule 6 is the lookahead**: a run of whitespace gives its last character to
  what follows, unless nothing follows. `"a   b"` is `"a"`, `"  "`, `" b"`;
  `"a   "` is `"a"`, `"   "`.

`qwen35` — the rules the 27B declares — differs only by adding `\p{M}` to the
letter and punctuation classes, so this scanner is one predicate away from
serving it too. `knownPre` refuses a name it has not implemented rather than
splitting by the wrong rules, which would cost a few tokens a sentence and
raise nothing.

## What it is checked against

`testdata/qwen/tokenizer/cases.json`: forty-one segmentations recorded from
llama.cpp, compared identifier by identifier. Twenty of them are the same text
as `ref/gemma/corpus.tsv`, so a difference between the two fixtures is the
tokenizer and not the text; the rest are one case per pre-tokenizer rule, plus
Qwen's own markers with and without `parse_special`.

```bash
cmake -S ref -B build/ref -DLLAMA_DIR=/path/to/llama.cpp && cmake --build build/ref -j8
mkdir -p testdata/qwen/tokenizer
build/ref/dump_tokens "$GOLEM_MODEL_QWEN" testdata/qwen/tokenizer ref/qwen/corpus.tsv
```

The tests want `GOLEM_MODEL_QWEN` pointing at the same file, and skip without
it: the vocabulary lives in the weights, which this repository does not ship.
