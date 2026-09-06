# Contributing

```bash
go build ./...
go test ./... -short -p 1
```

Weights are not in this repository, and every test that needs one skips
cleanly when it cannot find it. That is why the command above is safe on a
machine with no models and no card, and why CI runs it.

## What we need most

1. **ARM benchmarks.** The arm64 kernels are correct and tuned by nobody:
   written and verified under emulation, never once timed on real hardware. Run
   [`./benchmark-arm.sh`](benchmark-arm.sh) on Apple Silicon or Graviton and
   share what it prints.
2. **Bug reports.** If a test comparing against PyTorch or llama.cpp fails on
   your setup, please open an issue.
3. **Kernel optimization.** The NEON side of `nn/` is where the room is.

## Running the tests

`go test ./...` is the correctness suite. It runs the real models on both paths
(the parity fixtures, generation, vision, speech, the Vulkan kernels, and the
12B, 26B and 27B on the card) and takes about seventeen minutes on the machine
below, package by package.

```bash
go test ./...                           # correctness, every model, the card
GOLEM_FULL_TEST=1 go test ./qwen35/     # the rest, one package at a time
```

**`-p 1` matters if you have a GPU, and it is not a preference.** `go test
./...` runs one test binary per package concurrently, and three of these
packages put whole models on the card. Two of them at once ask for more than a
sixteen-gigabyte card has, and what comes back is `vkQueueSubmit failed
(VkResult -4)`, a lost device, in whichever binary happened to be second. It
reads like a driver fault, it is reproducible only by accident, and it cost an
afternoon of looking in the wrong place here.

**`-short`** skips what streams a fifty-two-gigabyte checkpoint or decodes
several hundred tokens on the processor. Without it the suite is hours, and
`qwen35` alone wants more than `go test`'s ten-minute default: give it
`-timeout 40m`.

### The thirty-second line

A check that takes longer than thirty seconds waits behind `GOLEM_FULL_TEST`,
whichever device it runs on. So does anything that is a measurement rather than
a check (a profile, a cost, a bench) and anything that streams a checkpoint
larger than the card. Each says which it is when it skips.

That leaves fifteen tests behind the variable, and they are the ones worth
knowing about. `TestVulkanPassProfile` runs longer than the test timeout
allows. `TestStreamedBF16MatchesWidened` streams fifty-two gigabytes twice.
`TestStreamedCalibrationIsWindowIndependent` calibrates the 27B on the
processor for three minutes.

Run them a package at a time and watch the machine. They are as much a load
test as a test, and running everything at once has taken this machine down.

`internal/heavy` is the whole mechanism: one guard, one reason per test, and
that reason reaches the skip line.

### How long each package takes

On an i7-9700K and an RX 9070 XT with `-p 1` and every checkpoint on disk:
`gemma` 378 s, `qwen35` 214 s, `vk` 117 s, `stt` 105 s, `pockettts` 96 s,
`compress` 92 s, `cmd/golem-server` 35 s, `qwen` 22 s, everything else under
six.

`go test` runs eight packages at once by default, which is faster and hungrier.
Several of them map a checkpoint of tens of gigabytes at the same time, so on a
machine with less memory `-p 2` is the flag that keeps it comfortable.

## Who wrote this

The code here was written by an AI agent, Claude, directed and reviewed by a
human. The commits carry it in their trailers.

It changes one thing for a contributor, and it is the thing the house rules
below are about: nothing in this repository is trusted because it looks right.
A layer is trusted because its intermediate activations were recorded from
llama.cpp or PyTorch and compared. A speed claim is trusted because the
benchmark that made it is in the tree. Hold a patch of yours to the same bar,
and hold ours to it too if something reads wrong.

## House rules

- **No layer is deemed correct until its intermediate activations match the
  reference implementation.** `ref/` says what recorded each fixture and how to
  record it again.
- **Every number in a README is a benchmark in this repository**, run on the
  machine named beside it. Nothing is estimated, and where golem loses it says
  so.
- Nothing is promoted into the shared layer on the strength of a guess. Code
  moves into `nn/`, `vk/` or `chat/` once two engines are shown to want it, in
  the same commit that makes them both use it.
