"""Produces the PyTorch reference fixtures for layer 0 of the flow_lm.

Usage (from the venv that holds pocket_tts):
    python ref/dump_layer.py <model.safetensors> <out_dir>

Writes one raw float32 file per intermediate activation, plus a fixtures.json
describing the shapes. The Go tests read these files back; they no longer need
Python.
"""

import json
import math
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file

from pocket_tts.modules.mimi_transformer import StreamingTransformerLayer
from pocket_tts.modules.rope import RotaryEmbedding

D_MODEL = 1024
NUM_HEADS = 16
DIM_FF = 4096
T = 3  # three timesteps: enough to exercise the causal mask and the KV cache

PREFIX = "flow_lm.transformer.layers.0."


def main() -> None:
    weights_path, out_dir = sys.argv[1], Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)

    tensors = load_file(weights_path)
    layer = StreamingTransformerLayer(
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        dim_feedforward=DIM_FF,
        context=None,
        rope=RotaryEmbedding(max_period=10000.0),
        layer_scale=None,
    )
    state = {
        k[len(PREFIX):]: v.float() for k, v in tensors.items() if k.startswith(PREFIX)
    }
    missing, unexpected = layer.load_state_dict(state, strict=False)
    assert not unexpected, unexpected
    assert not missing, missing
    layer.eval()

    # Deterministic input, independent of any seed: x[t, i] = sin((t*D + i) * 0.01)
    idx = torch.arange(T * D_MODEL, dtype=torch.float32)
    x = torch.sin(idx * 0.01).view(1, T, D_MODEL)

    dumped: dict[str, list[int]] = {}

    def dump(name: str, tensor: torch.Tensor) -> None:
        arr = tensor.detach().contiguous().float()
        (out_dir / f"{name}.bin").write_bytes(arr.numpy().tobytes())
        dumped[name] = list(arr.shape)

    attn = layer.self_attn
    with torch.no_grad():
        dump("input", x)

        # --- self-attention block, decomposed ---
        n1 = layer.norm1(x)
        dump("norm1", n1)

        projected = attn.in_proj(n1)
        dump("in_proj", projected)

        d = attn.dim_per_head
        q, k, v = torch.unbind(projected.view(1, T, 3, NUM_HEADS, d), dim=2)
        q_rope, k_rope = attn.rope(q, k, offset=0)
        dump("q_rope", q_rope)
        dump("k_rope", k_rope)
        dump("v", v)

        attn_out = attn(n1, None)
        dump("attn_out", attn_out)

        x1 = x + attn_out
        dump("post_attn", x1)

        # --- feed-forward block, decomposed ---
        n2 = layer.norm2(x1)
        dump("norm2", n2)
        h = layer.linear1(n2)
        dump("ff_hidden", h)
        g = torch.nn.functional.gelu(h)
        dump("ff_gelu", g)
        out = layer.forward(x, None)
        dump("output", out)

    (out_dir / "fixtures.json").write_text(
        json.dumps(
            {
                "d_model": D_MODEL,
                "num_heads": NUM_HEADS,
                "dim_feedforward": DIM_FF,
                "seq_len": T,
                "max_period": 10000.0,
                "torch": torch.__version__,
                "tensors": dumped,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"{len(dumped)} activations written to {out_dir}")


if __name__ == "__main__":
    main()
