# yue2

YuE2 from style tags and lyrics to a stereo song at 48 kHz, the way
yue2.cpp renders it: the same request JSON, the same stages, and for the same
seeds the same song, to within the arithmetic of the card.

| file, under `GOLEM_YUE2_DIR` (default `/mnt/data/LLMs_models/Serveurperso/YuE2-GGUF`) | what it is |
|---|---|
| `YuE2-3B-Q8_0.gguf` | the 28-layer backbone with its two weight sets (AR and NAR), the head, the tokenizer |
| `YuE2-Vae-F32.gguf` | the Oobleck decoder, latents to audio at 1920 samples a frame |

They are not in this repository. `ref/yue2/` records the fixtures the tests
compare against, from yue2.cpp's own code and CLI.

## The stages

One transformer of 28 blocks with two weight sets a block, and a decoder:

1. The score (`ar.go`, `generate.go`, `sampler.go`, `philox.go`): the AR
   half writes an ABC score token by token, unless the request brings one
   or turns it off (`cot: off`).
2. The codes: the same half writes the semantic stream, 25 codes a second,
   into a key and value cache, with classifier-free guidance in a second
   slot when the guidance is not one. A request with `semantic_tokens`
   replays them instead: the prefix and the codes are prefilled.
3. The flow (`nar.go`, `rng.go`): the NAR half reads the AR cache, copied on
   the card, and integrates 32 midpoint steps from torch's normal noise to
   64 latent channels a frame. A song longer than the context allows is cut
   into chunks of `(context - prefix - 3) / 2` frames, each re-prefilling its
   codes behind the prefix.
4. The decoder (`vae.go`): tiles of 512 frames with a halo of 16.
5. The output (`wav.go`): yue2.cpp's percentile normalization and its
   wav16, wav24 and wav32 writers, byte for byte. MP3 is not written.

`request.go` reads and writes yue2.cpp's request JSON field for field;
`pipeline.go` chains the stages. The three stay on the card for the
pipeline's life: no reload between stages or songs.

```go
p, err := yue2.Open(yue2.Options{}) // Q8_0, the model's whole context
r, err := yue2.ReadRequest("song.json")
song, err := p.Generate(r, nil)
wav, err := song.Encode(yue2.FormatWAV16, r.PeakClip)
replay := song.Replay(r).JSON() // renders the same song again, without the AR
```

## Running it

```sh
go build ./cmd/yue2
yue2 -request song.json -out song.wav            # writes song.wav and song.replay.json
yue2 -request replay.json -out song.wav -forced   # the codes decoded token by token
yue2 -request song.json -out song.wav -context 8192
```

`-forced` takes the request's score and codes as forced tokens through the
token-by-token decode rather than a prefill: every step computes its
distribution and then takes the given token, so it costs what drawing them
would. `-context` is yue2.cpp's `--max-seq`. It decides the chunking, so
yue2.cpp's output is matched only at the same value; the default is the
model's 24576, as yue2.cpp's. The card takes 13.45 GB at that context
(`cmd/yue2` reads the driver's count before and after loading); the AR's two
cache slots are 5.6 GB of it, 28 layers of 24576 positions of 1024 fp16 keys
and as many values each.

## How it is checked

Every stage is held to a recording of yue2.cpp (`ref/yue2/README.md`) on the
reference device, Vulkan0 on the RX 9070 XT. Three recordings of yue2.cpp
are taken with its arithmetic made exact, because its defaults on this card
are not, and golem's products sum fp16 operands in float32:

- the decoder against `vae_f32acc/`: ggml-vulkan accumulates fp16 products
  in fp16 by default (task 3);
- the flow matching half against `nar_exact/`: ggml-vulkan rounds every
  activation entering a Q8_0 product to Q8_1 by default (task 5);
- the whole pipeline's audio against `pipeline/short_exact`: the same
  rounding, end to end. yue2.cpp's default and exact renders of the short
  replay are themselves 28.5 dB apart on the audio; golem is 39.2 dB from the
  exact one and 29.1 dB from the default one (`TestPipelineShortReplay`).

`TestPipelineShortReplay` asserts 30 dB against `pipeline/short_exact`
(audio and latents) and against `nar/latent`; against the default render it
asserts what yue2.cpp's exact render reaches there, 28.5 dB. `TestPipelineChunks`
(`--max-seq 3150`, two chunks) asserts 30 dB against yue2.cpp's default
two-chunk render, and reaches 39.95 dB on the latents and 40.20 dB on the
audio. `TestPipelineSeal` holds a generated song's sealed first chunk to the
replay's prefilled one. `TestNormalize` holds the output stage to yue2.cpp's
bytes. `TestGuerrerosReplay` (cmd/yue2, heavy) renders the whole recorded
replay and holds its normalized wav16 to yue-synth's `bench/q8-replay.wav`
(30 dB; it reaches 34.75 dB) and its replay request to `bench/q8-replay.json`,
field for field.

## Numbers

Measured on 2026-10-20 on the RX 9070 XT (i7-9700K, 31 GB): golem's from
the tests' logs (`TestGuerrerosReplay`, `TestPipelineShortReplay`), yue2.cpp's
from yue-synth's log of the same request on the same card
(`bench/q8-replay.log`).

| | golem | yue2.cpp |
|---|---|---|
| Guerreros replay, 4870 frames (194.8 s of audio), one chunk | 29.2 s | 70.1 s, loads included |
| of which the prefill of 7877 positions | 9.59 s | |
| of which the flow, 32 steps | 18.99 s | 65.9 s |
| of which the decoder | 0.64 s | |
| short replay, 100 frames | 3.88 s | |

The prefill reads eight positions a pass (no tiled Q8_0 product yet). A
planned score is slow: the sampler sorts every candidate of the score
window at every step, where yue2.cpp partially sorts the top k.
