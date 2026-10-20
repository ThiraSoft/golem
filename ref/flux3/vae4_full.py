"""Decodes t2i4/latent whole, on the CPU in float32, as t2i4/rgb_full.

ComfyUI on a 16 GB card runs out of memory decoding 2048 x 2048 and falls
back to decode_tiled_ (t2i4/timings.json says "tiled"), the mean of three
tilings, which is not what a full decode gives. This records the full one,
the reference golem's banded decode is held to.

    systemd-run --user --scope -q -p MemoryMax=24G -p MemorySwapMax=16G \\
        /mnt/data/dev/ComfyUI/.venv/bin/python ref/flux3/vae4_full.py \\
        /mnt/data/dev/ComfyUI-flux3 testdata/flux3
"""

import json
import os
import sys
import time

import numpy as np

COMFY = sys.argv[1]
OUT = os.path.abspath(sys.argv[2])

sys.path.insert(0, COMFY)
os.chdir(COMFY)
sys.argv = [sys.argv[0], "--cpu", "--disable-mmap"]

import torch  # noqa: E402
import comfy.options  # noqa: E402
comfy.options.enable_args_parsing()
import nodes  # noqa: E402


def main():
    torch.set_grad_enabled(False)
    meta_path = os.path.join(OUT, "meta.json")
    with open(meta_path) as f:
        meta = json.load(f)
    shape = meta["t2i4/latent"]
    z = torch.from_numpy(np.fromfile(os.path.join(OUT, "t2i4/latent"), dtype=np.float32).reshape(shape))
    vae = nodes.VAELoader().load_vae("flux2-vae.safetensors")[0]
    fm = vae.first_stage_model.to("cpu", torch.float32)
    t0 = time.perf_counter()
    out = fm.decode(z)
    print(f"full decode on the CPU: {time.perf_counter() - t0:.1f} s")
    rgb = torch.clamp((out + 1.0) / 2.0, 0.0, 1.0)[0].movedim(0, -1).contiguous().numpy()
    rgb.astype(np.float32).tofile(os.path.join(OUT, "t2i4/rgb_full"))
    meta["t2i4/rgb_full"] = list(rgb.shape)
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=1, sort_keys=True)


if __name__ == "__main__":
    main()
