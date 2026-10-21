#!/usr/bin/env bash
# Times golem's cmd/yue2 on one request, as bench_yue2cpp.sh times yue-synth.
# usage: bench_golem.sh YUE2_BINARY REQUEST.json OUT_PREFIX [FLAGS...]
# FLAGS go to the command: -forced for the teacher-forced run,
# -compare PREFIX to say where a free run leaves a yue-synth one.
# Build the binary first (go build -o <path> ./cmd/yue2), so the wall clock
# is the process alone, the model open included.
set -euo pipefail
export LC_ALL=C
bin=$1 req=$2 out=$3
shift 3
mkdir -p "$(dirname "$out")"
start=$(date +%s.%N)
"$bin" -request "$req" -out "$out.wav" \
	-tokens "$out.tokens.csv" -score "$out.abc" "$@" 2>"$out.log"
end=$(date +%s.%N)
echo "wall $(awk "BEGIN{print $end - $start}") s" | tee -a "$out.log"
grep -E "^\[(Load|Time|Done|Compare)\]" "$out.log" || true
# The scope systemd-run puts the script in (ref/yue2/README.md): its peak,
# the page cache of the mmapped weights included, so not a resident size.
# Meaningful only inside that scope: run bare, this is the login session's
# cgroup and its peak is the whole desktop's.
peak=/sys/fs/cgroup$(cut -d: -f3 /proc/self/cgroup)/memory.peak
if [ -r "$peak" ]; then echo "cgroup memory.peak $(($(cat "$peak") / 1048576)) MiB" | tee -a "$out.log"; fi
