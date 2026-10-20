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
yue2 -request song.json -out song.wav -tokens song.csv -score song.abc -compare bench/q8-full
```

`-forced` takes the request's score and codes as forced tokens through the
token-by-token decode rather than a prefill: every step computes its
distribution and then takes the given token, so it costs what drawing them
would. `-context` is yue2.cpp's `--max-seq`. It decides the chunking, so
yue2.cpp's output is matched only at the same value; the default is the
model's 24576, as yue2.cpp's. `-tokens` and `-score` write the codes and
the score as yue-synth's `--tokens` and `--score` do, and `-compare` takes the
prefix of a yue-synth run and prints where the score (in tokens) and the
codes first leave it, and the SNR of the wav16 against its wav when the two
songs have the same length. The card takes 13.45 GB at that context
(`cmd/yue2` reads the driver's count before and after loading); the AR's two
cache slots are 5.6 GB of it, 28 layers of 24576 positions of 1024 fp16 keys
and as many values each.

## Accuracy

Every stage is held to a recording of yue2.cpp (`ref/yue2/README.md`) on the
reference device, Vulkan0 on the RX 9070 XT. The numbers below are what the
tests log on 2026-10-20 (`go test ./yue2/ -short -p 1 -v`, and
`GOLEM_FULL_TEST=1 go test ./cmd/yue2/ -run TestGuerrerosReplay`); rms
errors are relative to the recording.

**Three recordings are taken with yue2.cpp's arithmetic made exact**,
because its defaults on this card are not, and golem's products sum fp16
operands in float32. A bar against the default recording would measure
yue2.cpp's rounding as much as golem's:

- the decoder against `vae_f32acc/`: ggml-vulkan accumulates fp16 products
  in fp16 by default, which leaves its own first convolution 0.15% from the
  exact sum of its operands;
- the flow matching half against `nar_exact/`: ggml-vulkan rounds every
  activation entering a Q8_0 product to Q8_1 by default (0.34% on the
  input block alone);
- the whole pipeline's audio against `pipeline/short_exact`: the same
  rounding, end to end. yue2.cpp's default and exact renders of the short
  replay are themselves 28.53 dB apart on the audio.

The distances to the default recordings are logged beside the asserted ones.

| stage | test | against the exact recording (bar) | against yue2.cpp's default |
|---|---|---|---|
| decoder, 11 waypoints on 64 frames | `TestVAEWaypoints` | 0.0000% to 0.0511%, audio 0.0482% (0.2%) | 0.15% to 0.84%, audio 0.49% |
| decoder, 600 frames in tiles | `TestVAETiled` | | 40.49 dB (40 dB) |
| flow, input block and timestep | `TestNARInput` | 0.0016%, 0.0141% (0.1%) | 0.34%, 0.0141% |
| flow, layers 0 to 27 | `TestNARLayers` | 0.0147% to 0.0537% (1%) | 0.23% to 1.22% |
| flow, first velocity | `TestNARVelocity` | 0.0380% (2%) | 0.94% (2%) |
| flow, 32 midpoint steps on 100 frames | `TestNARSolve` | | latents 42.41 dB (30 dB) |
| AR prefill, hidden after layers 0, 13, 27 | `TestARPrefill` | 0.12%, 0.34%, 0.80% (logged) | 0.055%, 0.26%, 0.78% (1%) |
| AR prefill, hidden after the final norm | `TestARPrefill` | 0.95% (logged) | 0.93% (1%) |
| AR prefill, logits on the score and code rows | `TestARPrefill` | 0.078%, 1.28% (logged) | 0.092%, 1.05% a column at a time, codes 1.19% from the wide prefill (2%) |
| AR, eight decoded codes | `TestARDecode` | | 0.70% to 1.19% (2%), 8 of 8 tokens drawn as recorded |
| AR, guidance, four steps of two branches | `TestARGuided` | | 0.75% to 1.39% (2%), 4 of 4 tokens drawn as recorded |
| short replay, 100 frames | `TestPipelineShortReplay` | latents 41.22 dB, audio 32.57 dB (30 dB) | latents 41.72 dB, audio 28.97 dB (28.5 dB) |
| two chunks, `--max-seq 3150` | `TestPipelineChunks` | | latents 38.54 dB, audio 38.60 dB (30 dB) |
| Guerreros replay, 4870 frames, normalized wav16 | `TestGuerrerosReplay` | | 34.55 dB (30 dB) against `bench/q8-replay.wav` |

The AR is the one half held to yue2.cpp's default recording: its
logits feed a sampler, and the tokens it draws are the check that matters.
yue2.cpp's own default and exact prefills are 1.45% apart on the code
logits, which is why their bar is 2% (the exact recording is logged).

**The floors the whole song is read against**, both from yue2.cpp alone on
the Guerreros request (`bench/`): its full run and its replay of the same
codes differ by 34.8 dB on the wav16 (cache written by the decode against
cache written by a prefill), and its Q8_0 and BF16 replays differ by
25.8 dB. golem's replay lands at 34.55 dB from yue2.cpp's replay, at the
first floor; golem's teacher-forced run, whose cache the decode writes as
in yue2.cpp's full run, lands at 35.53 dB from `bench/q8-full.wav`
(`cmd/yue2 -forced -compare bench/q8-full`, both forced runs of the
sitting below). `TestGuerrerosReplay` asserts 30 dB, above
the 25.8 dB floor.

`TestPipelineSeal` holds a generated song's sealed first chunk to the
replay's prefilled one, `TestNormalize` the output stage to yue2.cpp's
bytes, and `TestGuerrerosReplay` the replay request to
`bench/q8-replay.json`, field for field.

## Speed

Measured on 2026-10-20 on the RX 9070 XT (i7-9700K, 31 GB), in one sitting:
card idle (0.79 GB on it, the desktop's and Steam's, when checked before
the first run and once the last had released its memory; no other GPU
process), DPM `auto`, power
profile `performance`, golem at `962be51`, yue2.cpp at `8af0174`. Each run
was made twice; the table is the second, the page cache warm for both
engines, and the first is within 0.2 s of it everywhere but yue2.cpp's
first full run, whose LM load took 1.53 s against 0.80 (109.56 s on the
wall clock). The commands are in
`ref/yue2/README.md` (`bench_yue2cpp.sh`, `bench_golem.sh`), the outputs in
`bench/t11b/`.

The Guerreros song, 2433 score tokens, 4870 codes, 194.8 s of audio. The
acceptance run is the first column: golem teacher-forced, the score and
codes fed through the token-by-token decode with every distribution
computed, which is the work of drawing them; yue2.cpp draws the same
tokens from the same seeds.

| stage | golem `-forced` | yue2.cpp full | golem replay | yue2.cpp replay |
|---|---:|---:|---:|---:|
| model open | **1.5 s** | 1.8 s (0.80 + 0.51 + 0.47, a stage at a time) | **1.6 s** | 1.8 s |
| score, 2433 tokens, prompt prefill included | **12.92 s** | 13.6 s | | |
| codes, 4870 tokens, prefill of 3006 included | **25.07 s** | 26.1 s | | |
| prefill of the 7877 positions | | | 1.21 s | not logged |
| flow, 32 steps on 4870 frames | **19.08 s** | 65.68 s | **19.15 s** | 65.72 s |
| decoder | **0.64 s** | 1.46 s | **0.64 s** | 1.45 s |
| wav16 | 0.17 s | not logged | 0.17 s | not logged |
| one request, golem's open apart, yue2.cpp's loads in | 57.89 s | 108.3 s | 21.18 s | 70.0 s |
| the process, wall clock | **59.68 s** | 108.64 s | **23.07 s** | 70.26 s |

The acceptance holds on the wall clock of the whole process, both sides
loading their models: 59.68 s against 108.64 s. The request row is not a
like-for-like pair: golem's leaves its 1.5 s open out, yue2.cpp's
`[Pipeline] Done` keeps its 1.8 s of loads in, since it loads each stage
when it reaches it. yue2.cpp's stage lines leave its prefills apart
(74 ms and 217 ms) and round to 0.1 s; the prefills are added in here.
Every stage of the request is golem's now: the codes take 5.15 ms a token
against 5.36 (prefills included), the score 5.31 against 5.58, and the
open 1.5 s against 1.8. The flow is 3.4 times faster, and it is most of
yue2.cpp's time.

One loss is left, the prompt's prefill: `BenchmarkARPrefill` takes
234.1 ms for the 3006-position Guerreros prompt, measured the same
evening, against the 217 ms yue2.cpp logs for it in this sitting. Of its
last pass, positions 2560 to 3006, 54.9% is the attention
(`vk/shaders/yue2_attn_wide.comp`, 31.4 ms of 57.3), and most of the rest
is the products, as wide passes of 512 on the matrix cores.

The card holds 13.45 GB for golem's pipeline at the model's whole context
(the driver's count before and after the open; a run that starts while the
previous process's memory is still being released reads less, and is not
the figure).

## Same seeds

The free run, from the request's seeds alone (`bench/guerreros-full.json`,
`cmd/yue2 -compare bench/q8-full`), twice, token for token the same both
times:

| | golem | yue2.cpp |
|---|---:|---:|
| score | 2883 tokens, 15.42 s | 2433 tokens, 13.6 s |
| codes | 5405 tokens, 28.70 s | 4870 tokens, 26.1 s |
| flow, 32 steps | 22.38 s (5405 frames) | 65.68 s (4870 frames) |
| decoder | 0.71 s | 1.46 s |
| one request, golem's open apart, yue2.cpp's loads in | 67.40 s | 108.3 s |
| the process, wall clock | 69.16 s | 108.64 s |

golem's score leaves yue2.cpp's Q8_0 one at its token 873, and is byte for
byte the score yue2.cpp's BF16 run writes from the same seeds
(`bench/bf16-full.abc`, 2883 tokens); its codes follow that run's for 30
codes and then leave them. yue2.cpp's own Q8_0 and BF16 runs with the same
seeds also write different scores (2433 and 2883 tokens) and
different codes.
The two songs differ in length, so the free run is not the acceptance
figure; the teacher-forced run above does the same work as yue2.cpp's.
