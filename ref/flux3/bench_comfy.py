"""Benchmark ComfyUI on FLUX 3 Image Klein on the RX 9070 XT.

Times text encoder, DiT load, DiT per-step, VAE decode, and total picture
generation at 1024x1024 and 768x1024 (16 steps and 8 steps).

Run via:
    systemd-run --user --scope -q -p MemoryMax=24G -p MemorySwapMax=16G \
        /mnt/data/dev/ComfyUI/.venv/bin/python ref/flux3/bench_comfy.py
"""

import math
import os
import sys
import time

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

import torch  # noqa: E402
import comfy.options  # noqa: E402
comfy.options.enable_args_parsing()
import nodes  # noqa: E402
from comfy_extras.nodes_flux import EmptyFlux2LatentImage  # noqa: E402
from comfy_extras.nodes_model_advanced import ModelSamplingFlux  # noqa: E402

UNET = "flux3-image-klein-bf16.safetensors"
CLIP = "qwen3vl_4b_fp8_scaled.safetensors"
VAE = "flux2-vae.safetensors"

STEPS = 16
CFG = 1.0
SAMPLER = "euler"
SCHEDULER = "simple"
ALPHA = 16.0
SHIFT = math.log(ALPHA)  # 2.77259

PROMPT = (
    "close up profile of a young male athlete with closed eyes pressing his "
    "sweaty forehead against a soccer ball, right hand gripping the ball, "
    "stark white background, athletic jacket, white ball with jagged red graphic "
    "patterns, high contrast studio lighting, highly detailed skin texture, "
    "sharp focus Absolutely no text, no typography, no letters, no words, "
    "no numbers, no logos, no watermark, no captions, pure imagery only."
)


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def main():
    torch.set_grad_enabled(False)
    print("=== ComfyUI FLUX 3 Image Klein Benchmark ===")
    print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    # 1. Model Loading
    print("\n--- 1. Model Loading ---")
    t0 = time.perf_counter()
    unet = nodes.UNETLoader().load_unet(UNET, "default")[0]
    sync()
    t_unet_load = time.perf_counter() - t0
    print(f"DiT (bf16) load: {t_unet_load:.3f} s")

    t0 = time.perf_counter()
    clip = nodes.CLIPLoader().load_clip(CLIP, "flux3", "default")[0]
    sync()
    t_clip_load = time.perf_counter() - t0
    print(f"CLIP load: {t_clip_load:.3f} s")

    t0 = time.perf_counter()
    vae = nodes.VAELoader().load_vae(VAE)[0]
    sync()
    t_vae_load = time.perf_counter() - t0
    print(f"VAE load: {t_vae_load:.3f} s")

    # 2. Text Encoder
    print("\n--- 2. Text Encoder ---")
    t0 = time.perf_counter()
    pos = nodes.CLIPTextEncode().encode(clip, PROMPT)[0]
    neg = nodes.CLIPTextEncode().encode(clip, "")[0]
    sync()
    t_enc_first = time.perf_counter() - t0
    print(f"Text encode (first/cold): {t_enc_first:.3f} s")

    t0 = time.perf_counter()
    pos = nodes.CLIPTextEncode().encode(clip, PROMPT)[0]
    neg = nodes.CLIPTextEncode().encode(clip, "")[0]
    sync()
    t_enc_warm = time.perf_counter() - t0
    print(f"Text encode (warm): {t_enc_warm:.3f} s")

    # Hook diffusion model forward to record per-step times
    step_times = []

    def make_dm_hook():
        last_t = [None]

        def pre_hook(m, args, kwargs):
            sync()
            last_t[0] = time.perf_counter()

        def post_hook(m, args, kwargs, output):
            sync()
            if last_t[0] is not None:
                step_times.append(time.perf_counter() - last_t[0])

        return pre_hook, post_hook

    dm = unet.model.diffusion_model
    pre_h, post_h = make_dm_hook()
    h1 = dm.register_forward_pre_hook(pre_h, with_kwargs=True)
    h2 = dm.register_forward_hook(post_h, with_kwargs=True)

    # 3. Benchmark runs
    # 1024x1024, 16 steps (First run: DiT uploaded into VRAM)
    print("\n--- 3. Generation 1024x1024 (16 steps) ---")
    step_times.clear()
    t0 = time.perf_counter()
    p_unet = ModelSamplingFlux().patch(unet, SHIFT, SHIFT, 1024, 1024)[0]
    lat = EmptyFlux2LatentImage.execute(1024, 1024, 1)[0]
    t_samp0 = time.perf_counter()
    samples = nodes.common_ksampler(p_unet, 42, 16, CFG, SAMPLER, SCHEDULER, pos, neg, lat, denoise=1.0)[0]
    sync()
    t_samp_first = time.perf_counter() - t_samp0

    t_dec0 = time.perf_counter()
    _ = nodes.VAEDecode().decode(vae, samples)[0]
    sync()
    t_dec_1024 = time.perf_counter() - t_dec0
    t_first_1024 = time.perf_counter() - t0

    step_1024 = min(step_times) if step_times else (t_samp_first / 16)
    print(f"1024x1024 (16 steps, cold VRAM): total {t_first_1024:.3f} s (sample {t_samp_first:.3f} s, decode {t_dec_1024:.3f} s, best step {step_1024:.3f} s)")

    # 1024x1024, 16 steps (Warm run: DiT already on card)
    step_times.clear()
    t0 = time.perf_counter()
    p_unet = ModelSamplingFlux().patch(unet, SHIFT, SHIFT, 1024, 1024)[0]
    lat = EmptyFlux2LatentImage.execute(1024, 1024, 1)[0]
    t_samp0 = time.perf_counter()
    samples = nodes.common_ksampler(p_unet, 42, 16, CFG, SAMPLER, SCHEDULER, pos, neg, lat, denoise=1.0)[0]
    sync()
    t_samp_warm_1024 = time.perf_counter() - t_samp0

    t_dec0 = time.perf_counter()
    _ = nodes.VAEDecode().decode(vae, samples)[0]
    sync()
    t_dec_1024_warm = time.perf_counter() - t_dec0
    t_warm_1024 = time.perf_counter() - t0
    step_1024_warm = min(step_times) if step_times else (t_samp_warm_1024 / 16)
    print(f"1024x1024 (16 steps, warm): total {t_warm_1024:.3f} s (sample {t_samp_warm_1024:.3f} s, decode {t_dec_1024_warm:.3f} s, best step {step_1024_warm:.3f} s)")

    # 1024x1024, 8 steps (Warm run)
    step_times.clear()
    t0 = time.perf_counter()
    p_unet = ModelSamplingFlux().patch(unet, SHIFT, SHIFT, 1024, 1024)[0]
    lat = EmptyFlux2LatentImage.execute(1024, 1024, 1)[0]
    t_samp0 = time.perf_counter()
    samples = nodes.common_ksampler(p_unet, 42, 8, CFG, SAMPLER, SCHEDULER, pos, neg, lat, denoise=1.0)[0]
    sync()
    t_samp_8_1024 = time.perf_counter() - t_samp0

    t_dec0 = time.perf_counter()
    _ = nodes.VAEDecode().decode(vae, samples)[0]
    sync()
    t_dec_8_1024 = time.perf_counter() - t_dec0
    t_8_1024 = time.perf_counter() - t0
    print(f"1024x1024 (8 steps, warm): total {t_8_1024:.3f} s (sample {t_samp_8_1024:.3f} s, decode {t_dec_8_1024:.3f} s)")

    # 768x1024, 16 steps (Warm run)
    print("\n--- 4. Generation 768x1024 ---")
    step_times.clear()
    t0 = time.perf_counter()
    p_unet = ModelSamplingFlux().patch(unet, SHIFT, SHIFT, 768, 1024)[0]
    lat = EmptyFlux2LatentImage.execute(768, 1024, 1)[0]
    t_samp0 = time.perf_counter()
    samples = nodes.common_ksampler(p_unet, 42, 16, CFG, SAMPLER, SCHEDULER, pos, neg, lat, denoise=1.0)[0]
    sync()
    t_samp_768 = time.perf_counter() - t_samp0

    t_dec0 = time.perf_counter()
    _ = nodes.VAEDecode().decode(vae, samples)[0]
    sync()
    t_dec_768 = time.perf_counter() - t_dec0
    t_warm_768 = time.perf_counter() - t0
    step_768 = min(step_times) if step_times else (t_samp_768 / 16)
    print(f"768x1024 (16 steps, warm): total {t_warm_768:.3f} s (sample {t_samp_768:.3f} s, decode {t_dec_768:.3f} s, best step {step_768:.3f} s)")

    # 768x1024, 8 steps (Warm run)
    step_times.clear()
    t0 = time.perf_counter()
    p_unet = ModelSamplingFlux().patch(unet, SHIFT, SHIFT, 768, 1024)[0]
    lat = EmptyFlux2LatentImage.execute(768, 1024, 1)[0]
    t_samp0 = time.perf_counter()
    samples = nodes.common_ksampler(p_unet, 42, 8, CFG, SAMPLER, SCHEDULER, pos, neg, lat, denoise=1.0)[0]
    sync()
    t_samp_8_768 = time.perf_counter() - t_samp0

    t_dec0 = time.perf_counter()
    _ = nodes.VAEDecode().decode(vae, samples)[0]
    sync()
    t_dec_8_768 = time.perf_counter() - t_dec0
    t_8_768 = time.perf_counter() - t0
    print(f"768x1024 (8 steps, warm): total {t_8_768:.3f} s (sample {t_samp_8_768:.3f} s, decode {t_dec_8_768:.3f} s)")

    h1.remove()
    h2.remove()

    print("\n=== SUMMARY TABLE ===")
    print(f"Text Encoder: {t_enc_warm*1000:.1f} ms (cold {t_enc_first*1000:.1f} ms)")
    print(f"DiT Load: {t_unet_load:.2f} s")
    print(f"VAE Decode: 1024x1024: {t_dec_1024*1000:.1f} ms, 768x1024: {t_dec_768*1000:.1f} ms")
    print(f"DiT Step (1024x1024): {step_1024_warm:.3f} s")
    print(f"DiT Step (768x1024): {step_768:.3f} s")
    print(f"1024x1024 (16 steps, Keep): {t_warm_1024:.2f} s")
    print(f"1024x1024 (8 steps, Keep): {t_8_1024:.2f} s")
    print(f"768x1024 (16 steps, Keep): {t_warm_768:.2f} s")
    print(f"768x1024 (8 steps, Keep): {t_8_768:.2f} s")


if __name__ == "__main__":
    main()
