#!/usr/bin/env bash
# Times yue2.cpp (yue-synth, Vulkan0) on one request and keeps its log.
# usage: bench_yue2cpp.sh YUE2_ROOT QUANT REQUEST.json OUT_PREFIX
# The card must be free: unload the musicbench worker first
# (curl -X POST localhost:8005/api/unload) and check rocm-smi.
set -euo pipefail
root=$1 quant=$2 req=$3 out=$4
models=$root/models
start=$(date +%s.%N)
"$root/build/yue-synth" --device Vulkan0 \
	--model "$models/YuE2-3B-$quant.gguf" --vae "$models/YuE2-Vae-F32.gguf" \
	--request "$req" --out "$out.wav" --tokens "$out.tokens.csv" --score "$out.abc" \
	--latent "$out.vae" 2>"$out.log"
end=$(date +%s.%N)
echo "wall $(awk "BEGIN{print $end - $start}") s" | tee -a "$out.log"
grep -E "\[AR\] (Score|Semantic):|\[NAR\] Solved|\[VAE\] (Tiled decode done|Decoded)|\[Pipeline\] Done|Load" "$out.log" || true
