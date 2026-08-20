"""Produces the PyTorch reference fixtures for the Mimi encoder and the voice
prompt it feeds.

Usage (from a venv that holds pocket_tts):
    python ref/dump_encoder.py <language> <out_dir>

Writes one raw float32 file per intermediate activation, plus a fixtures.json
describing the shapes. The Go tests read these files back; they need no Python.

Nothing this writes is versioned: the activations belong to whoever published
the weights. See the .gitignore.
"""

import json
import sys
from pathlib import Path

import torch

from pocket_tts.models.tts_model import TTSModel

# Two frames of audio. The encoder's receptive field is longer than one frame,
# so a single one would leave the streaming padding untested, and a long input
# would only make the fixtures heavier.
FRAMES = 2
FRAME_SIZE = 1920


def main() -> None:
    language, out_dir = sys.argv[1], Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)

    model = TTSModel.load_model(language=language)
    mimi = model.mimi
    mimi.eval()

    dumped: dict[str, list[int]] = {}

    def dump(name: str, tensor: torch.Tensor) -> None:
        arr = tensor.detach().contiguous().float()
        (out_dir / f"{name}.bin").write_bytes(arr.numpy().tobytes())
        dumped[name] = list(arr.shape)

    # A deterministic waveform, independent of any seed and of any recording:
    # x[t] = sin(t * 0.01) * 0.5, which stays inside [-1, 1] and exercises every
    # channel of the first convolution.
    n = FRAMES * FRAME_SIZE
    idx = torch.arange(n, dtype=torch.float32)
    audio = (torch.sin(idx * 0.01) * 0.5).view(1, 1, n)
    dump("audio", audio)

    with torch.no_grad():
        emb = mimi.encoder(audio, model_state=None)
        dump("encoder", emb)

        (after,) = mimi.encoder_transformer(emb, model_state=None)
        dump("encoder_transformer", after)

        latents = mimi._to_framerate(after)
        dump("latents", latents)

        conditioning = torch.nn.functional.linear(
            latents.transpose(-1, -2).to(torch.float32), model.flow_lm.speaker_proj_weight
        )
        dump("speaker_proj", conditioning)

    (out_dir / "fixtures.json").write_text(json.dumps(dumped, indent=2, sort_keys=True))
    print(f"{len(dumped)} fixtures in {out_dir}")


if __name__ == "__main__":
    main()
