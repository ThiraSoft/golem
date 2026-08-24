#!/bin/bash
#
# The arm64 kernels in this repository have never been timed on an ARM machine.
#
# They were written and verified under QEMU on an x86 desktop, where the cost of
# an instruction is the cost of translating it, so every ratio quoted in the
# README is x86-64 and every arm64 figure is an emulator's opinion. This script
# is the measurement nobody here could take. If you have an Apple Silicon Mac, a
# Graviton, an Ampere or a Pi, running it and posting golem-arm.txt in an issue
# is the single most useful thing this project can be sent.
#
#     ./benchmark-arm.sh
#
# Parts 1 to 5 need no model files. Parts 6 and 7 use them if they are there.
# Nothing here writes to the repository or the network.

set -u

case "$(uname -m)" in
  arm64|aarch64) ;;
  *) echo "This is for arm64. On $(uname -m) the kernels measured here do not exist."; exit 1 ;;
esac

OUT="golem-arm.txt"
exec > >(tee "$OUT") 2>&1
say() { echo; echo "######## $* ########"; echo; }

say 1. THE MACHINE
uname -srm
go version
if [ "$(uname -s)" = "Darwin" ]; then
  sw_vers | sed 's/^/  /'
  sysctl -n machdep.cpu.brand_string
  echo "cores      : $(sysctl -n hw.logicalcpu) logical"
  # The split matters: part 5 is about cores of unequal speed, and this is
  # where a reader finds out whether this machine has any.
  echo "performance: $(sysctl -n hw.perflevel0.logicalcpu 2>/dev/null || echo '?')"
  echo "efficiency : $(sysctl -n hw.perflevel1.logicalcpu 2>/dev/null || echo '?')"
  echo "memory     : $(( $(sysctl -n hw.memsize) / 1024 / 1024 / 1024 )) GB"
else
  grep -m1 "model name" /proc/cpuinfo 2>/dev/null || echo "model: $(uname -p)"
  echo "cores      : $(nproc)"
  grep -m1 MemTotal /proc/meminfo 2>/dev/null
fi
git log --oneline -1 --decorate 2>/dev/null

say 2. IT BUILDS AND IT PASSES
go build ./... && echo "build ok"
go vet ./... && echo "vet ok"
go test ./... 2>&1 | grep -vE "no test files"

say 3. THE ASSUMPTION THIS PORT MAKES
# On darwin the engine ASSUMES FEAT_DotProd rather than asking: asking would
# mean golang.org/x/sys/cpu, whose darwin support wants Go 1.25, against a
# README that promises 1.23 and nothing else. Every Apple part from the M1 on
# has it, so the assumption should hold — but it has never been tested on one.
# On linux the same question is a real probe of /proc/self/auxv.
#
# If these skip or fail, every Q4_0 and Q6_K figure below is meaningless.
go test ./nn -run 'Encoding|DotProd|SDOT|UDOT' -v 2>&1 | grep -E "^(=== RUN|--- |ok|FAIL)"

say 4. THE KERNELS, AGAINST THE GO THEY REPLACE
echo "# Same process, same data, NEON against the portable loop it replaces."
echo "# Emulated, these came out at x1.38 for Q4_0 and x1.34 for bfloat16."
echo "# Expect less here: a real core hides arithmetic behind memory, and the"
echo "# instruction count stops being what decides."
go test ./nn -run xxx -bench 'DotQ4_0Go|DotQ4_0NEON|DotBF16' -benchtime 2s -count=6

say 5. THE POOL, ON CORES OF UNEQUAL SPEED
# The worker pool used to hand out ranges of equal size, so the barrier waited a
# whole range past the moment the fastest core ran dry. On a chip whose cores
# differ two or three times that range belongs to the slowest one, so the ranges
# now shrink as the work runs out. It was written for this machine and never run
# on one.
#
# Batch* are Q4_0 kernels that the bfloat16 work does not touch, so main against
# 48a366b isolates that one change.
BEFORE=48a366b
HERE=$(git rev-parse --abbrev-ref HEAD 2>/dev/null)
echo "--- HEAD ($HERE): ranges that shrink ---"
go test ./nn -run xxx -bench 'BatchFFN|BatchDown|BatchQ|MatVecFF|ParallelRead' -benchtime 2s -count=6
echo
if [ -n "$(git status --porcelain 2>/dev/null)" ]; then
  echo "--- $BEFORE skipped: the working tree has changes and this part moves HEAD ---"
elif ! git cat-file -e "$BEFORE^{commit}" 2>/dev/null; then
  echo "--- $BEFORE skipped: not in this clone ---"
else
  echo "--- $BEFORE: ranges of equal size ---"
  git checkout -q "$BEFORE" && \
    go test ./nn -run xxx -bench 'BatchFFN|BatchDown|BatchQ|MatVecFF|ParallelRead' -benchtime 2s -count=6
  git checkout -q "$HERE"
  echo "back on: $(git log --oneline -1)"
fi

say 6. THE SPEECH ENGINE
if go test ./pockettts -run TestFullSynthesisAgainstReference 2>&1 | grep -qE "no test files"; then
  echo "skipped"
else
  echo "# Against PyTorch, frame by frame through the feedback loop, where a"
  echo "# rounding gap does not repeat but accumulates. Emulated arm64 gave"
  echo "# 0.048% of the scale, x86-64 gives 0.197%, the bar is 0.5%. A figure"
  echo "# far from either means the arithmetic differs, not the instruction set."
  go test ./pockettts -run TestFullSynthesisAgainstReference -v 2>&1 | grep -E "max gap|SKIP|--- |^ok|FAIL"
  go test ./pockettts -run xxx -bench 'Frame|Synthesis|AdvanceLatent' -benchtime 3s -count=4
fi

say 7. A WHOLE MODEL, IF YOU HAVE ONE
if [ -n "${GOLEM_MODEL:-}" ] && [ -f "${GOLEM_MODEL:-}" ]; then
  go build ./cmd/golem-cli
  echo "# Three runs: the page cache makes the first one a lie."
  for i in 1 2 3; do
    ./golem-cli -model "$GOLEM_MODEL" -seed 1 -temp 0 -n 128 \
      -p "Explain a mutex in one sentence." -stats 2>&1 | tail -4
  done
  go test ./gemma -run xxx -bench . -benchtime 20x 2>&1 | grep -E "^(Benchmark|ok|FAIL)"
else
  echo "skipped: point GOLEM_MODEL at a Gemma or Qwen GGUF to include this part"
fi

say DONE
echo "All of it is in $OUT."
echo
echo "If you are comparing against llama.cpp, give llama-bench -ngl 0 or it"
echo "answers for the GPU, and run each side twice — the page cache makes a"
echo "first read of the weights look like arithmetic."
