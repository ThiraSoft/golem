"""Produces the PyTorch reference fixtures for Kyutai STT.

Usage (from a venv that holds moshi):
    pip install moshi
    python ref/dump_stt.py kyutai/stt-1b-en_fr testdata/stt

Writes one raw float32 file per activation — int32 for the codes — plus a
fixtures.json describing the shapes. The Go tests read these back; they need
no Python.

Nothing this writes is versioned: the activations belong to whoever published
the weights. See the .gitignore.
"""

import json
import sys
from pathlib import Path

import torch
from moshi.models import loaders

# Eight frames. The encoder's receptive field is longer than one frame, so a
# single one would leave the causal padding untested, and eight is enough for
# the trunk's attention to have a past without making the fixtures heavy.
FRAMES = 8
FRAME_SIZE = 1920


def main() -> None:
    repo, out_dir = sys.argv[1], Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)

    repo_path = Path(repo)
    if repo_path.is_dir() and (repo_path / "model.safetensors").exists():
        info = loaders.CheckpointInfo.from_hf_repo(
            "kyutai/stt-1b-en_fr",
            moshi_weights=repo_path / "model.safetensors",
            mimi_weights=repo_path / "mimi-pytorch-e351c8d8@125.safetensors",
            tokenizer=repo_path / "tokenizer_en_fr_audio_8000.model",
            config_path=repo_path / "config.json",
        )
    else:
        info = loaders.CheckpointInfo.from_hf_repo(repo)
    mimi = info.get_mimi(device="cpu")
    lm = info.get_moshi(device="cpu")
    mimi.eval()
    lm.eval()

    dumped: dict[str, list[int]] = {}

    def dump(name: str, tensor: torch.Tensor) -> None:
        arr = tensor.detach().contiguous()
        arr = arr.int() if arr.dtype in (torch.int32, torch.int64) else arr.float()
        (out_dir / f"{name}.bin").write_bytes(arr.numpy().tobytes())
        dumped[name] = list(arr.shape)

    # The same deterministic waveform as ref/dump_encoder.py, so the two
    # encoders are compared on identical input: x[t] = sin(t * 0.01) * 0.5.
    n = FRAMES * FRAME_SIZE
    idx = torch.arange(n, dtype=torch.float32)
    audio = (torch.sin(idx * 0.01) * 0.5).view(1, 1, n)
    dump("audio", audio)

    with torch.no_grad():
        emb = mimi.encoder(audio)
        dump("encoder", emb)
        (after,) = mimi.encoder_transformer(emb)
        dump("encoder_transformer", after)
        latents = mimi.downsample(after)
        dump("latents", latents)

        codes = mimi.quantizer.encode(latents)  # [B, n_q, T]
        dump("codes", codes[0].to(torch.int32))

        # The trunk, stepped by hand so that every waypoint is reachable.
        text = torch.full((1, 1), lm.text_emb.num_embeddings - 1, dtype=torch.long)
        embs, block0s, trunks, logits = [], [], [], []
        with lm.streaming(batch_size=1):
            for t in range(codes.shape[-1]):
                x = lm.text_emb(text)
                for q in range(codes.shape[1]):
                    x = x + lm.emb[q](codes[:, q, t : t + 1])
                embs.append(x)
                h = lm.transformer.layers[0](x)
                h = h if not isinstance(h, tuple) else h[0]
                block0s.append(h)
                for layer in lm.transformer.layers[1:]:
                    h = layer(h)
                    h = h if not isinstance(h, tuple) else h[0]
                y = lm.out_norm(h)
                trunks.append(y)
                l = lm.text_linear(y)
                logits.append(l)
                text = l.argmax(dim=-1)
        dump("emb", torch.cat(embs, dim=1))
        dump("block0", torch.cat(block0s, dim=1))
        dump("trunk", torch.cat(trunks, dim=1))
        dump("logits", torch.cat(logits, dim=1))

    (out_dir / "fixtures.json").write_text(json.dumps(dumped, indent=2))


if __name__ == "__main__":
    main()
