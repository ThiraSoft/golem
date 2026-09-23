"""Converts the eye upscaler and records what it computes, for the Go side.

Usage (from the venv of ref/tha3/README.md):
    python ref/tha3/upscaler.py <realesr-animevideov3.pth> <weights dir> [<fixture dir>]

The upscaler is Real-ESRGAN's realesr-animevideov3, an SRVGGNetCompact: a 3x3
convolution to 64 channels and sixteen more among them, each followed by a
PReLU, and a last one to 48 channels that a pixel shuffle turns into the
picture four times larger, added to the input enlarged by nearest neighbour.
It is by Xintao Wang et al., BSD-3-Clause, from
https://github.com/xinntao/Real-ESRGAN.

The checkpoint is written as F32 safetensors, eye_upscaler.safetensors, under
its own tensor names. With a fixture dir, the network is also run here, in
plain torch rather than through any package, on a 40x48 crop of noise and
gradients: the input and output go to <fixture dir>/{in,out}.bin as float32,
with their shapes in fixtures.json.
"""

import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import save_file

CONVS = 18


def forward(state: dict, x: torch.Tensor) -> torch.Tensor:
    y = x
    for i in range(CONVS):
        y = F.conv2d(y, state[f"body.{2 * i}.weight"], state[f"body.{2 * i}.bias"], padding=1)
        if i < CONVS - 1:
            y = F.prelu(y, state[f"body.{2 * i + 1}.weight"])
    y = F.pixel_shuffle(y, 4)
    return y + F.interpolate(x, scale_factor=4, mode="nearest")


def main() -> None:
    checkpoint, out = Path(sys.argv[1]), Path(sys.argv[2])
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = state.get("params", state)
    tensors = {k: v.detach().to(torch.float32).contiguous() for k, v in state.items()}
    if len(tensors) != 3 * CONVS - 1:
        sys.exit(f"{checkpoint}: {len(tensors)} tensors, want {3 * CONVS - 1}")
    out.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(out / "eye_upscaler.safetensors"))
    print(f"eye_upscaler: {len(tensors)} tensors")
    if len(sys.argv) < 4:
        return
    fixture = Path(sys.argv[3])
    fixture.mkdir(parents=True, exist_ok=True)
    g = torch.Generator().manual_seed(7)
    h, w = 40, 48
    ramp = torch.linspace(0, 1, w).expand(h, w)
    x = torch.stack([ramp, ramp.flip(1), torch.rand(h, w, generator=g)])
    x = x.unsqueeze(0).contiguous()
    with torch.no_grad():
        y = forward(tensors, x)
    shapes = {}
    for name, t in (("in", x), ("out", y)):
        (fixture / f"{name}.bin").write_bytes(t.contiguous().numpy().tobytes())
        shapes[name] = list(t.shape)
    (fixture / "fixtures.json").write_text(json.dumps({"tensors": shapes}) + "\n")
    print(f"fixture: {shapes}")


if __name__ == "__main__":
    main()
