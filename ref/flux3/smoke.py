"""Headless smoke test for FLUX 3 Image Klein in ComfyUI.

Runs the text-to-image workflow through ComfyUI's own nodes in-process.

Upstream workflow defaults size to 2048 x 2048 (node 19 ResolutionSelector with
4.0 MP square); golem defaults to 1024 x 1024 for the 16 GB card budget. This
smoke test runs at 1024 x 1024 with the optional LLM prompt upsampler (node 127)
bypassed, which is a non-goal for golem.
"""

import math
import os
import sys

import numpy as np
from PIL import Image

COMFY = "/mnt/data/dev/ComfyUI-flux3"
sys.path.insert(0, COMFY)
os.chdir(COMFY)
sys.argv = [
    sys.argv[0],
    "--use-split-cross-attention",
    "--fp16-text-enc",
    "--fp16-vae",
    "--disable-mmap",
    "--reserve-vram", "1.5",
    "--deterministic",
]

import comfy.options
comfy.options.enable_args_parsing()
import nodes
from comfy_extras.nodes_flux import EmptyFlux2LatentImage
from comfy_extras.nodes_model_advanced import ModelSamplingFlux

UNET = "flux3-image-klein-bf16.safetensors"
CLIP = "qwen3vl_4b_fp8_scaled.safetensors"
VAE = "flux2-vae.safetensors"

PROMPT = (
    "close up profile of a young male athlete with closed eyes pressing his "
    "sweaty forehead against a soccer ball, right hand gripping the ball, "
    "stark white background, athletic jacket, white ball with jagged red graphic "
    "patterns, high contrast studio lighting, highly detailed skin texture, "
    "sharp focus Absolutely no text, no typography, no letters, no words, "
    "no numbers, no logos, no watermark, no captions, pure imagery only."
)

WIDTH = 1024
HEIGHT = 1024
STEPS = 16
CFG = 1.0
SAMPLER = "euler"
SCHEDULER = "simple"
SEED = 758781618552631
ALPHA = 16.0
SHIFT = math.log(ALPHA)


def main():
    print(f"Loading UNET: {UNET}")
    unet = nodes.UNETLoader().load_unet(UNET, "default")[0]

    print(f"Loading CLIP: {CLIP} (flux3)")
    clip = nodes.CLIPLoader().load_clip(CLIP, "flux3", "default")[0]

    print(f"Loading VAE: {VAE}")
    vae = nodes.VAELoader().load_vae(VAE)[0]

    print(f"Patching ModelSamplingFlux (shift={SHIFT:.5f}, size={WIDTH}x{HEIGHT})")
    patched_unet = ModelSamplingFlux().patch(unet, SHIFT, SHIFT, WIDTH, HEIGHT)[0]

    print("Encoding prompt...")
    cond = nodes.CLIPTextEncode().encode(clip, PROMPT)[0]
    uncond = nodes.CLIPTextEncode().encode(clip, "")[0]

    print(f"Creating empty latent ({WIDTH}x{HEIGHT})...")
    latent = EmptyFlux2LatentImage.execute(WIDTH, HEIGHT, 1)[0]

    print(f"Sampling: {STEPS} steps, {SAMPLER}/{SCHEDULER}, cfg={CFG}, seed={SEED}...")
    samples = nodes.KSampler().sample(
        patched_unet, SEED, STEPS, CFG, SAMPLER, SCHEDULER, cond, uncond, latent, denoise=1.0
    )[0]

    print("Decoding latent with VAE...")
    decoded = nodes.VAEDecode().decode(vae, samples)[0]

    arr = 255.0 * decoded[0].cpu().numpy()
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    out_path = os.path.join(root, "testdata", "flux3", "smoke.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    img.save(out_path)
    print(f"Wrote smoke picture to {out_path} ({img.size})")


if __name__ == "__main__":
    main()
