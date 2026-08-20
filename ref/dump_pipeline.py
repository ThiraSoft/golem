"""Produces the reference fixtures of the whole pipeline, frame by frame.

Replays what `TTSModel.generate_audio` does, but keeping control of the only
non-deterministic element — the starting noise of the flow — and writing every
intermediate quantity to disk. The Go engine reads the noise back and must
recover the same latents, then the same audio.

Usage (from the venv that holds pocket_tts):
    python ref/dump_pipeline.py <voice_state.safetensors> <out_dir> [language] [text]

The language defaults to french_24l. Its text rules — semicolon folding, padding
of short inputs — are read from the model itself, so the fixture always records
what that language actually does rather than what French does.
"""

import json
import sys
from pathlib import Path

import torch

from pocket_tts.models.tts_model import TTSModel, _import_model_state, prepare_text_prompt
from pocket_tts.modules.stateful_module import increment_steps, init_states

TEXT = "Bonjour le monde."
LANGUAGE = "french_24l"
FRAMES = 8
SEED = 1234


def main() -> None:
    voice_state, out_dir = sys.argv[1], Path(sys.argv[2])
    language = sys.argv[3] if len(sys.argv) > 3 else LANGUAGE
    raw_text = sys.argv[4] if len(sys.argv) > 4 else TEXT
    frames = FRAMES
    out_dir.mkdir(parents=True, exist_ok=True)

    shapes: dict[str, list[int]] = {}

    def dump(name: str, t: torch.Tensor) -> None:
        arr = t.detach().contiguous().float()
        (out_dir / f"{name}.bin").write_bytes(arr.numpy().tobytes())
        shapes[name] = list(arr.shape)

    model = TTSModel.load_model(language=language)
    flow_lm = model.flow_lm
    state = _import_model_state(voice_state, torch.device("cpu"))

    text, _ = prepare_text_prompt(
        raw_text, model.pad_with_spaces_for_short_inputs, model.remove_semicolons
    )
    prepared = flow_lm.conditioner.prepare(text)
    tokens = prepared.tokens
    (out_dir / "tokens.bin").write_bytes(
        tokens.to(torch.int32).numpy().tobytes()
    )
    shapes["tokens"] = list(tokens.shape)

    start = int(next(iter(state.values()))["offset"].view(-1)[0].item())
    model._expand_kv_cache(state, sequence_length=start + tokens.shape[1] + frames + 4)

    with torch.no_grad():
        # --- text conditioning ---
        text_emb = flow_lm.conditioner(prepared)
        dump("text_emb", text_emb)

        # --- prompt fill: the transformer swallows the text tokens ---
        entry = flow_lm.input_linear(
            torch.empty((1, 0, flow_lm.ldim), dtype=torch.float32)
        )
        prompt_out = flow_lm.transformer(text_emb, state)
        increment_steps(flow_lm, state, increment=tokens.shape[1])
        dump("prompt_out", prompt_out)

        # --- autoregressive generation ---
        generator = torch.Generator().manual_seed(SEED)
        current = torch.full((1, 1, flow_lm.ldim), float("nan"))
        noises, latents, conds, eos = [], [], [], []

        for _ in range(frames):
            sequence = torch.where(torch.isnan(current), flow_lm.bos_emb, current)
            entry = flow_lm.input_linear(sequence)
            out = flow_lm.transformer(entry, state)
            increment_steps(flow_lm, state, increment=1)
            out = flow_lm.out_norm(out)[:, -1]
            conds.append(out)
            eos.append(flow_lm.out_eos(out))

            noise = torch.empty((1, flow_lm.ldim))
            torch.nn.init.normal_(noise, mean=0.0, std=model.temp**0.5, generator=generator)
            noises.append(noise)

            # single-step lsd_decode: x1 = x0 + v(0, 1, x0)
            zero = torch.zeros_like(noise[..., :1])
            one = torch.ones_like(zero)
            latent = noise + flow_lm.flow_net(out, zero, one, noise)
            latents.append(latent)
            current = latent[:, None, :]

        dump("cond", torch.cat(conds, dim=0))
        dump("eos_logit", torch.cat(eos, dim=0))
        dump("noise", torch.cat(noises, dim=0))
        dump("latent", torch.cat(latents, dim=0))

        # --- Mimi decoding, frame by frame, streaming ---
        mimi = model.mimi
        mimi_state = init_states(mimi, batch_size=1, sequence_length=frames * 16 + 64)
        chunks = []
        for i, latent in enumerate(latents):
            denorm = latent * flow_lm.emb_std + flow_lm.emb_mean
            quant = mimi.quantizer(denorm[:, :, None])
            if i == 0:
                dump("mimi_quant0", quant)
            audio = mimi.decode_from_latent(quant, mimi_state)
            increment_steps(mimi, mimi_state, increment=16)
            chunks.append(audio)
        dump("audio", torch.cat(chunks, dim=-1))

    (out_dir / "fixtures.json").write_text(
        json.dumps(
            {
                "language": language,
                "num_layers": len(flow_lm.transformer.layers),
                "text": text,
                "frames": frames,
                "seed": SEED,
                "temp": model.temp,
                "eos_threshold": model.eos_threshold,
                "voice_offset": start,
                "tensors": shapes,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"{len(shapes)} quantities written to {out_dir}")


if __name__ == "__main__":
    main()
