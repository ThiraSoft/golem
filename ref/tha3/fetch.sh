#!/bin/sh
# Downloads the separable_float checkpoints of Talking Head(?) Anime 3 into
# the directory given, or ./tha3-pt. The URLs are the ones the upstream demo's
# colab.ipynb uses. The weights are CC-BY 4.0, by Pramook Khungurn.
set -eu
dir=${1:-tha3-pt}
mkdir -p "$dir"
get() {
	curl -fL -o "$dir/$1.pt" "https://www.dropbox.com/s/$2/$1.pt?dl=1"
}
get editor nwdxhrpa9fy19r4
get eyebrow_decomposer hfzjcu9cqr9wm3i
get eyebrow_morphing_combiner g04dyyyavh5o1e2
get face_morpher vgi9dsj95y0rrwv
get two_algo_face_body_rotator 8u0qond8po34l24
ls -l "$dir"
