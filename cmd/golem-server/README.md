# golem-server

An OpenAI-compatible API over a GGUF, on the CPU. Gemma 4 or Qwen3: the file
declares its own architecture and `engine/` opens whichever implements it, so
this command names neither, and tools work on both.

```bash
go build ./cmd/golem-server
./golem-server -model gemma-4-E2B-it-QAT-Q4_0.gguf -addr 127.0.0.1:8080
./golem-server -model Qwen3-4B-Q4_0.gguf -addr 127.0.0.1:8080
```

One model per server, and the request's `model` field routes nothing: there is
one set of weights, so there is nothing to route to. `-model`
names the file; `/v1/models` reports it.

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

## Several conversations, in one pass

`-parallel` is how many conversations the server holds at once, and they are
answered together rather than in turn. Each slot is a KV cache holding its own
conversation; the context is cut rather than multiplied, as llama.cpp's
`--parallel` cuts it, so `-context 4096 -parallel 2` is two conversations of
2048 positions and the memory is what it was.

What makes them cheap together is that they go through the model together. One
goroutine owns the weights, and whatever is waiting when it builds a pass goes
into that pass: four conversations each wanting one token are four tokens in
one read of a gigabyte of weights, rather than four reads. That is llama.cpp's
continuous batching, and on this engine it is worth this much — Gemma 4 E2B on
an eight-thread i7-9700K, clients asking for 64 tokens each at once:

| clients | tokens a second, all together |
|---|---|
| 1 | 20.7 |
| 2 | 34.7 |
| 4 | 54.2 |
| 8 | 73.8 |

Against llama.cpp's own server on the same machine and the same file
(`llama-server -dev none -ngl 0 -t 8 -c 4096 -np 4 -cb`), three rounds, a
different prompt every round so that neither server is answering out of a
cache it filled a moment ago:

| clients | golem | llama-server | golem ÷ llama-server |
|---|---|---:|---|
| 1 | 20.7 t/s | 20.3 | ×1.02 |
| 2 | 34.7 | 33.1 | ×1.05 |
| 4 | 54.2 | 60.5 | ×0.90 |
| 8 | 73.8 | 72.9 | ×1.01 |

Where the four-client deficit is, exactly: `GOLEM_DEBUG_BATCHES=1` over that
run says the batches are as wide as they can be — 63 of the 67 passes carried
all four conversations, 3.88 on average — and that a pass with its head takes
60ms. What is left is the 655ms spent reading the four prompts, which is not
scheduling but the rate this engine reads prompts at. Generation is at parity;
the prompt is the half that is behind, which `gemma/README.md` already says.

The engine's own figure for generation, without a server around it, is
`TestMixedBatchCost` in `gemma/`: 21.4 tokens a second for one conversation,
42.3 for two, 73.3 for four, 104.1 for eight.

Two things bound it. Generation is limited by reading the weights, so the first
conversations added are nearly free and the later ones are not — by eight, the
arithmetic is what is left. And the output head is the largest matrix in the
model: it is read once for every state being scored in a pass, which is why the
scoring travels with the pass rather than after it.

### On the card

`-vulkan -parallel N` batches the same way. The caches are cut the same way
too: one ring a conversation, laid end to end in the buffer a block already
had, with the slot travelling beside the position in `vk/attention.go`'s
position buffer — so the card holds N conversations for the memory one of N
times the context held. `TestMixedBatchCostVulkan` in `gemma/`, the 26B A4B on
an RX 9070 XT:

| conversations | pass | head | tokens a second |
|---|---|---|---|
| 1 | 11ms | 1ms | 82.1 |
| 2 | 14ms | 2ms | 125.9 |
| 4 | 14ms | 5ms | 211.4 |
| 8 | 22ms | 9ms | 255.0 |

Taken from four hundred and forty-eight positions of context, because a token
drawn at position 64 costs 7.6ms on this model and one drawn at 512 costs
14.1ms — the attention reads everything before it, and a table taken at the
shallow end would say a number no server ever sees.

That is 1.53, 2.57 and 3.11 of one conversation. What keeps it from being two,
four and eight is that generation is limited by reading the weights only while
there is nothing else to do: by eight columns the arithmetic and the head are
what is left.

A mixture had a threshold in the way of this. It answers a single column by
reading the eight matrices that column routed to, and a prompt by reading each
expert once for the columns that chose it — and it used to switch to the second
at two columns, so a token drawn for each of two conversations read a hundred
and twenty-eight experts to answer sixteen pairs. The two costs cross near
thirty-two on this checkpoint, which is where `vk/mixture.go`'s `byExpertFrom`
now sits; the by-column kernels take a column offset so they can answer a
handful of columns rather than one.

The scores kernel is what limits how a pass may be mixed. It answers
thirty-two columns to a workgroup off the keys of the union of their ranges,
and two conversations cannot share that scratch — a position means a different
entry of the cache in each. So a pass is cut into runs of one conversation
before the tiles are laid out. A prompt is one run and tiles as it always did;
a token drawn for each of four conversations is four runs of one column, which
is the shape generation has anyway.

How wide a pass may be is the other thing the card changes. A prompt is read in
chunks of thirty-two positions on the processor, because past sixty-four the
activations stop fitting in its caches; a card is idle at that width and reads
a prompt some five times faster at two hundred and fifty-six. Positions a
second by the width, the same 26B on the same card:

| width | 32 | 64 | 128 | 256 | 512 |
|---|---|---|---|---|---|
| positions a second | 950 | 2044 | 3797 | 4981 | 5486 |

`Runner.PassWidth` is which of the two this server uses, and `main.go` asks the
model where its blocks are and says so once. Five hundred and twelve is the
widest the stack carries and also the longest pass — 93ms, against 51ms at 256
— so a conversation waiting on its next token waits that much longer for one
that is reading a prompt. That wait is a fraction of a prompt, once; the rate
is every prompt.

End to end, through the API, with prompts of about five hundred and forty
tokens and nothing else on the card:

| | prompt | drawn |
|---|---|---|
| one client | 4055/s | 75.3/s |
| two clients, each | ~2130/s | ~57.8/s |
| two clients, together | ~4260/s | ~115.6/s |

The prompt figure was 804/s before the width followed the device, and two
clients drew 100/s together before the mixture's threshold moved.

Qwen3.5's GPU pipeline is the exception and still refuses `-parallel` above 1:
its delta-net blocks keep a state matrix a head rather than a ring, and there
is nothing there to cut into slots.

### Waiting for a pass

The runner waits a fraction of a pass — an eighth of what the last one took, at
most two milliseconds — for the other conversations in flight to arrive before
going without them. A lone client cannot feel that; a second client is worth
far more than it costs.

`GOLEM_DEBUG_BATCHES=1` says what went into each pass, which is how the table
above was checked rather than assumed.

Which slot answers is decided as llama.cpp's server decides it. The free slot
whose tokens share the longest prefix with this prompt wins, provided the
shared part is a tenth of the prompt or more; failing that, a slot holding no
conversation; failing that, the one nobody has come back to in longest. Three
requests measured on a Qwen3 0.6B — Alice's conversation, Bob's, then Alice's
next turn:

| | Alice's second turn costs |
|---|---|
| `-parallel 1` | 59 positions: Bob's conversation had taken the cache |
| `-parallel 2` | 21 positions: the added turn, and nothing else |

A request that waited for a slot — all of them held — says so in its log line,
next to the slot that answered it:

```
POST /v1/chat/completions 200 284 bytes in 139ms — slot 0 — 21 prompt in 37ms (566.4/s), 8 drawn in 102ms (78.65/s), length
```

A client that hangs up while queued takes its request with it, and never
reaches a slot.

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
| `-context` | positions to keep; 4096 by default, cut between the slots |
| `-parallel` | conversations at once; 1 by default |
| `-n` | most tokens for one answer, when the request names no `max_tokens` |
| `-cache-ttl` | forget a conversation's tokens after this long idle; `0` never |

Sampling follows the file's own values unless the request names `temperature`,
`top_p`, `top_k` or `seed`.
