# golem-server

An OpenAI-compatible API over a Gemma 4 GGUF, on the CPU.

```bash
go build ./cmd/golem-server
./golem-server -model gemma-4-E2B-it-QAT-Q4_0.gguf -addr 127.0.0.1:8080
```

Two endpoints:

| | |
|---|---|
| `POST /v1/chat/completions` | a conversation, streamed or not, tool declarations included |
| `GET /v1/models` | the one model, named after the file, for clients that probe at startup |

## Tools

Declare them the way the API declares them, and the calls come back the way it
returns them:

```bash
curl -s localhost:8080/v1/chat/completions -H 'content-type: application/json' -d '{
  "messages": [{"role": "user", "content": "What is the weather in Lyon?"}],
  "tools": [{"type": "function", "function": {
    "name": "get_weather", "description": "Current weather in a city.",
    "parameters": {"type": "object",
      "properties": {"city": {"type": "string", "description": "The city."}},
      "required": ["city"]}}}]}'
```

```json
{"choices":[{"index":0,"message":{"role":"assistant","content":"",
  "tool_calls":[{"id":"call_1","type":"function",
    "function":{"name":"get_weather","arguments":"{\"city\":\"Lyon\"}"}}]},
  "finish_reason":"tool_calls"}]}
```

Running the function is the client's part. Send the result back as a `tool`
message carrying the call's `tool_call_id`, as the protocol has it, and the
model speaks on inside the same turn — that is the template's own rule, not a
choice made here.

Streamed, a call never arrives in fragments. Prose leaves as it is drawn; from
the moment `<|tool_call>` appears the output is held back, and the whole call
leaves in a single `tool_calls` delta once it has closed. OpenAI's API allows
fragments and clients do reassemble them, but half a function's arguments is a
thing nothing can check.

## One request at a time

There is one model and one KV cache, so a mutex serializes requests: a second
one waits for the first to finish. No queue, no batching between requests. For
a machine running a 12B on eight cores that is the honest shape of it.

## The cache across stateless requests

`/v1/chat/completions` carries the whole conversation every time; the cache does
not want to be rebuilt every time. So the server remembers which tokens sit in
the cache, compares them with the prompt it just rendered, and prefills only the
divergence. A conversation growing by one exchange costs one exchange.

Rewinding is where it gets particular. A sliding-window block keeps its keys in
a ring of exactly the window, so writing up to position Q and then going back to
P overwrites the slots of positions `P-W+1 … Q-W`, which are still visible from
P. A rewind therefore restarts a window early rather than at P; its cost is
bounded by the window, and appending — the common case — rewinds nothing.

`-cache-ttl` bounds how long a conversation's tokens stay in memory. Say plainly
what it does not do: the cache is allocated once at startup and this frees none
of it. What it does is stop the server from holding one client's conversation
indefinitely, and give the next request a clean state. `0`, the default, never
forgets.

## A client that hangs up

Cancelling a request stops the drawing. With one model behind one lock, an
answer nobody is waiting for is not merely wasted: it is the next request's
wait. The generation loop watches the request's context and returns as soon as
the connection is gone.

## Where the time goes

Every request leaves a line on standard error, split between the two halves of
the wait — they have nothing in common and nothing to gain from being added up:

```
POST /v1/chat/completions 200 268 bytes in 51.31s — 3919 prompt in 51.226s (76.5/s), 2 drawn in 70ms (28.70/s), stop
```

A first call carrying a large system prompt and a page of tool declarations is
almost entirely prefill, and prefill is the slower half of this engine. The
cache is what pays it back: the next turn of that conversation shares the whole
prefix and reads only what was added. A client that rewrites the head of its
conversation between turns — an hour stamped into the system message, tools
reordered — diverges at position zero and repays the whole prompt every time.
The `prompt` figure in the log says which of the two is happening.

Sixteen, sixty-four, and up to five hundred and twelve positions per batch were
measured against thirty-two on this engine, and thirty-two won every time: at
3919 positions, 76.2/s against 71.7 at sixty-four and 46.4 at five hundred and
twelve. Past that the activations stop fitting in the caches.

## What it refuses

With a 400 and OpenAI's error envelope, rather than answering something else:
`n` above 1, `logprobs`, `tool_choice` beyond `auto` and `none`, a conversation
with no message, a tool result answering no call, and a prompt longer than
`-context`.

Not implemented at all: `/v1/completions`, embeddings, images, audio.

## Flags

| | |
|---|---|
| `-model` | the GGUF, or `GOLEM_MODEL` |
| `-addr` | what to listen on; `127.0.0.1:8080` by default |
| `-context` | positions to keep; 4096 by default |
| `-n` | most tokens for one answer, when the request names no `max_tokens` |
| `-cache-ttl` | forget a conversation's tokens after this long idle; `0` never |

Sampling follows the file's own values unless the request names `temperature`,
`top_p`, `top_k` or `seed`.
