# stt

Speech-to-text with the Kyutai STT model (`stt-1b-en_fr`).

Given audio in any common format (WAV, MP3, FLAC) at any sample rate and channel
count, this package transcribes what is said in English or French.

## Usage

```go
options, err := stt.Locate("/path/to/stt-checkpoint")
if err != nil {
    log.Fatal(err)
}
model, err := stt.Open(options)
if err != nil {
    log.Fatal(err)
}
defer model.Close()

// Whole-file transcription:
text, err := model.Transcribe(audioBytes)

// Streaming transcription:
err = model.TranscribeStream(ctx, audioBytes, func(seg stt.Segment) {
    fmt.Printf("[%5.2fs] %s\n", float64(seg.Frame)/12.5, seg.Text)
})
```

## Architecture

- **Codec**: Mimi SEANet encoder (24 kHz -> 12.5 Hz, latent dim 512) followed by a 32-level Split RVQ quantizer.
- **Trunk**: 16-layer autoregressive transformer (width 2048, 16 heads, SwiGLU FFN 5632, RoPE) decoding text tokens at 12.5 Hz.

## Command line and server

```bash
./golem-cli -stt ~/models/stt-1b-en_fr -transcribe recording.wav
./golem-cli -stt ~/models/stt-1b-en_fr -listen   # the microphone, until Ctrl-C
```

`-listen` finds its own recorder, trying `pw-record`, then `arecord`, then
`ffmpeg`, and prints words as they are decided.

The server carries it too, in OpenAI's shape, streamed or not, and may carry it
without a language model beside it:

```bash
./golem-server -stt ~/models/stt-1b-en_fr -addr 127.0.0.1:8080
curl -s localhost:8080/v1/audio/transcriptions -F file=@recording.wav -F model=stt
```

## Speed

On an i7-9700K with eight threads, one frame of the 80 ms budget costs 39.6 ms:
6.6 for the codec, 1.75 for the quantiser, 29.8 for the trunk in Q8_0. That is
**twice real time on the processor**, with the transcript bfloat16 gives, word
for word. Q4_0 is half again as fast and loses six per cent of the words, which
is why it is not the default.

## Several microphones at once

```bash
./golem-server -stt ~/models/stt-1b-en_fr -stt-parallel 2
```

One stream saturates that processor, and separate streams share nothing: their
aggregate throughput is flat from one client to eight. The trunk holds
fifty-four megabytes of weights a layer, nothing of that size stays in a cache,
and each stream reads all of it again for its one column.

So the streams of a group are stepped together and the weights are read once
for all of them, with the row as the outer loop and the batch as the inner one.
It is the same mechanism that lets a prompt be read faster than an answer is
written. Attention is not shared: each stream keeps its own cache and its own
window.

`-vulkan` then puts the trunk's four products on the card, at whatever width
the group runs:

```bash
./golem-server -stt ~/models/stt-1b-en_fr -stt-parallel 3 -vulkan
./golem-cli -stt ~/models/stt-1b-en_fr -vulkan -listen
```

Seventy seconds of audio per arm, at the steady state where attention walks the
whole window, in aggregate seconds of audio per second of wall clock:

| streams | separate | grouped | grouped + vulkan |
| ------- | -------- | ------- | ---------------- |
| 1       | 1.81     | 1.78    | 2.94             |
| 2       | 1.82     | 2.17    | 3.38             |
| 3       | 1.81     | 2.43    | 3.50             |
| 4       | 1.81     | 2.61    | 3.64             |

Per stream that is ×1.08 for two clients grouped, where separate streams left
them at ×0.91 and falling behind, and ×1.17 for three on the card. One
microphone alone goes from ×1.81 to ×2.94. Four still miss at ×0.91, because
the codec and the quantiser batch no better on a card than off one and they are
what the ceiling is now.

A full group answers 429 rather than queueing. `go test -run
TestConcurrentStreams` with `GOLEM_STT_CAPACITY` set is where these numbers
come from.

**The products move and nothing else does.** Attention stays on the processor,
since each stream owns a cache of seven hundred and fifty positions and moving
those would mean moving the streams, so a block on the card is four dispatches
with the processor in between. That middle is not free: a product staged,
dispatched and read back costs about twice its kernel. It is paid anyway,
because the same products cost six times more on the processor than the round
trip costs on top. The logit head stays too. It is bfloat16, which these
kernels do not read, and it is 2.68 ms against the trunk's thirty.
