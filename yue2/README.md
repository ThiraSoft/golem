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

The tests assert 30 dB against the exact recordings and log the default
ones. The two-chunk render (`TestPipelineChunks`, `--max-seq 3150`) is
within 40.2 dB of yue2.cpp's default render on the audio.

## Numbers

Measured on 2026-10-20 on the RX 9070 XT (i7-9700K, 31 GB), `cmd/yue2`
against yue-synth on the same card:

| | golem | yue2.cpp |
|---|---|---|
| load (both halves and the decoder, once) | 4.6 s | 0.8 + 0.5 + 0.5 s, a stage at a time |
| Guerreros replay, 4870 frames (194.8 s of audio), one chunk | 29.5 s | 70.1 s |
| of which the prefill of 7877 positions | 9.6 s | |
| of which the flow, 32 steps | 19.0 s | 65.9 s |
| of which the decoder | 0.64 s | |
| short replay, 100 frames | 3.95 s | 4.0 s, loads included |

The full replay's normalized wav16 is 34.75 dB from yue2.cpp's
`bench/q8-replay.wav`, and its replay JSON carries the same fields and
values as yue2.cpp's. The prefill reads eight positions a pass (no tiled
Q8_0 product yet). A score is slow: `-forced` on the short replay's request
decodes its 2433 score tokens in 65.3 s, 26.8 ms a token where the AR's pass
is about 5 ms, because `Distribution` sorts all 151849 candidates of the
score window every step where yue2.cpp partially sorts the top k.
