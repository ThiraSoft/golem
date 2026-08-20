# An OpenAI-compatible server for Gemma, with tool calls

## What this is

A second command, `cmd/serve`, that maps a GGUF once and answers HTTP requests
on `/v1/chat/completions` and `/v1/models`, in the shape OpenAI's API defines.
It streams, it declares tools to the model, and it reports back the calls the
model makes. It does not run those calls: the client does, and sends the result
in the next request, which is what the OpenAI protocol asks of a server.

Two things stand between the current code and that server. The chat template in
`gemma/` renders text conversations only — tools, tool calls and tool responses
are refused rather than approximated. And `gemma`'s KV cache is stateful across
turns while `/v1/chat/completions` is stateless: every request carries the whole
conversation.

## The template, outward

`gemma/chat.go` grows the tool path of the Jinja template, spelled the way the
rest of that file is spelled: what the template writes, written out, and pinned
to a fixture that Jinja itself produced.

New types:

```go
// Tool is one function the model may call, in the shape the OpenAI API
// declares it: {"type": "function", "function": {…}}.
type Tool struct {
    Name        string
    Description string
    Parameters  *Schema // nil when the function takes none
    Response    *Schema // nil unless the caller declares one
}

// Schema is the subset of JSON Schema the template reads.
type Schema struct {
    Type        string   // "object", "string", "array", …
    Description string
    Properties  map[string]*Schema
    Required    []string
    Items       *Schema
    Enum        []string
    Nullable    bool
}

// ToolCall is one call the model made, or one being fed back as context.
type ToolCall struct {
    ID        string // the client's identifier; the template never renders it
    Name      string
    Arguments map[string]any
}
```

`Message` gains `ToolCalls []ToolCall` and accepts the role `tool`, which
carries a tool's result in `Content` and the function's name in `Name`.
`ChatOptions` gains `Tools []Tool`.

The rendering rules, read off the template:

- Tools open the system turn, after the system message and the thinking marker:
  `<|tool>declaration:NAME{description:<|"|>…<|"|>,parameters:{properties:{…},required:[<|"|>a<|"|>],type:<|"|>OBJECT<|"|>}}<tool|>`,
  one block per tool, all inside the single system turn.
- `<|"|>` is the template's quote. Property keys are bare. Maps are walked in
  sorted key order — Jinja's `dictsort` — so a Go map must be sorted before it
  is written.
- Types are upper-cased: `string` renders as `<|"|>STRING<|"|>`.
- A call is `<|tool_call>call:NAME{key:value,…}<tool_call|>`, arguments sorted,
  keys bare, string values quoted with `<|"|>`.
- A result is `<|tool_response>response:NAME{key:value}<tool_response|>`; a
  result that is not a map is wrapped as `{value:…}`.
- A tool result belongs to the model turn that made the call. The template
  writes no `<turn|>` after an assistant message whose calls are answered, and
  `add_generation_prompt` adds no `<|turn>model` header after one: the model
  keeps speaking in the turn it had started. An assistant message whose calls
  are *not* answered ends with a bare `<|tool_response>` instead.

Fixtures: `ref/gemma/dump_chats.py` gains cases covering a declaration with no
parameters, one with an enum, one with an array, a call with mixed argument
types, a result fed back, and a result left unanswered. `testdata/gemma/chat/`
holds what Jinja made of them. No new Jinja is interpreted at test time.

## The template, inward

`gemma/toolcall.go` reads what the model emits back into `[]ToolCall`. The
argument syntax is not JSON — `<|"|>` for quotes, bare keys, bare numbers and
booleans — so this is a small hand-written scanner over the text between
`<|tool_call>` and `<tool_call|>`, tested on its own, including on truncated and
malformed input, where it reports an error rather than a half-parsed call.

`ParseToolCalls(text string) (before string, calls []ToolCall, err error)`
returns the prose that preceded the first call and the calls that followed.

## The server

`cmd/serve` holds the HTTP layer and nothing about attention.

- `GET /v1/models` — one entry, named after the file, so a client that probes at
  startup finds what it expects.
- `POST /v1/chat/completions` — reads `messages`, `tools`, `tool_choice`
  (`auto` and `none` only; anything else is a 400), `stream`, `temperature`,
  `top_p`, `max_tokens`, `seed`, `stop`. Unsupported fields that change the
  answer's shape — `n` above 1, `logprobs`, image parts — are refused with a
  400 rather than ignored.
- `stream: false` returns one JSON body. `stream: true` writes SSE chunks and
  closes with `data: [DONE]`.
- Text reaches the client as it is drawn. As soon as `<|tool_call>` appears the
  stream holds its output until `<tool_call|>` closes it, then emits the parsed
  call as a single `tool_calls` delta: a partial call never leaves the server.
  Each call is given an identifier the client echoes back.
- `finish_reason` is `tool_calls` when the answer ended on a call, `length` on
  the token limit, `stop` otherwise.
- Errors use OpenAI's envelope: `{"error": {"message", "type", "code"}}`.

One model, one cache: a mutex serializes requests. A second request waits for
the first to finish. There is no queue and no cross-request batching, and the
README says so rather than implying otherwise.

## The cache across stateless requests

The server keeps the token identifiers currently held in the cache. Each request
renders its conversation, encodes it, and compares it to that record: the shared
prefix is already computed, and only the divergence is prefilled. A request that
appends one exchange to the last one costs one turn, as `cmd/chat` does today.

Two rules make that safe.

**Rewinding a sliding window.** Window blocks store their keys in a ring of
exactly the window, so writing positions past P and then rewinding to P
overwrites the slots of positions `[P-W+1, Q-W]`, which are still inside the
window seen from P. So when the shared prefix P is shorter than what was written
(Q), prefill restarts at `max(0, P-W+1)`, where W is the largest window among
the blocks — never at P itself. Rewriting those positions is idempotent for the
global blocks. The cost of a rewind is bounded by the window; appending to the
end rewinds nothing and costs nothing.

**A time to live.** `-cache-ttl` bounds how long a conversation's tokens stay in
memory. The memory itself is allocated once at startup and is never released, so
the flag frees nothing — what it does is stop the server from holding one
client's conversation indefinitely, and give a clean state after a pause.
Default `0`, meaning never. When set, a request arriving more than that long
after the last one served drops the record and resets the cache, and its
conversation is prefilled whole. The check happens under the same mutex, against
an injectable clock, so the test does not sleep.

## Testing

- `gemma/chat_test.go` gains the tool cases, checked against the Jinja fixture,
  as the existing cases are.
- `gemma/toolcall_test.go` covers the scanner, round trip and failures.
- `cmd/serve` is tested with `httptest` against a scripted engine and the
  word vocabulary already written for `cmd/chat/session_test.go` — no weights on
  the machine. What is checked: the JSON of a plain answer, the SSE frame
  sequence, a tool call surfacing as one delta and never in pieces, the prefix
  reuse (how many positions were fed on the second request), the rewind
  restarting a window early, and the TTL resetting on a moved clock.
- One test with real weights, behind `GOLEM_MODEL` like the others: a
  conversation declaring one tool, whose answer is a parseable call.

## Out of scope

`/v1/completions`, embeddings, `logprobs`, `n > 1`, images, audio, a request
queue, and running the tools. Each is a separate piece of work, and none is
stubbed in.
