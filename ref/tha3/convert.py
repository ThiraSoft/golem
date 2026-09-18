"""Converts the THA3 separable_float checkpoints to safetensors.

Usage (from a venv holding torch and safetensors):
    python ref/tha3/convert.py <dir with the .pt files> <out dir>

Each .pt is a plain state dict. It is written back as F32 safetensors under
the same tensor names, so the Go side reads the names the Python modules use.
A state dict carrying BatchNorm statistics or spectral norm is refused: none
of the five separable_float networks has either, and a variant that did would
need folding this script does not do.
"""

import sys
from pathlib import Path

import torch
from safetensors.torch import save_file

NETWORKS = [
    "eyebrow_decomposer",
    "eyebrow_morphing_combiner",
    "face_morpher",
    "two_algo_face_body_rotator",
    "editor",
]
REFUSED = ("running_mean", "running_var", "weight_orig", "weight_u", "weight_v")


def main() -> None:
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    listing = []
    for name in NETWORKS:
        state = torch.load(src / f"{name}.pt", map_location="cpu", weights_only=True)
        tensors = {}
        for key, value in state.items():
            if key.endswith(REFUSED):
                sys.exit(f"{name}: {key} needs folding, which this script does not do")
            tensors[key] = value.detach().to(torch.float32).contiguous()
            listing.append(f"{name}\t{key}\t{list(value.shape)}")
        save_file(tensors, str(out / f"{name}.safetensors"))
        print(f"{name}: {len(tensors)} tensors")
    (out / "keys.txt").write_text("\n".join(listing) + "\n")


if __name__ == "__main__":
    main()
