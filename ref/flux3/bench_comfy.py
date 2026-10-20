"""Benchmark ComfyUI on FLUX 3 Image Klein on the RX 9070 XT.

Times text encoder, DiT load, DiT per-step, VAE decode, and total picture
generation at 1024x1024 and 768x1024 (16 steps and 8 steps).

Run via:
    systemd-run --user --scope -q -p MemoryMax=24G -p MemorySwapMax=16G \
        /mnt/data/dev/ComfyUI/.venv/bin/python ref/flux3/bench_comfy.py
"""

import logging
import math
import os
import sys
import time

logging.basicConfig(level=logging.INFO, format="%(message)s")

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
    import comfy.model_management
    t0 = time.perf_counter()
    unet = nodes.UNETLoader().load_unet(UNET, "default")[0]
    sync()
    t_unet_host = time.perf_counter() - t0

    t0_vram = time.perf_counter()
    comfy.model_management.load_models_gpu([unet])
    sync()
    t_unet_vram_upload = time.perf_counter() - t0_vram
    t_unet_load_total = t_unet_host + t_unet_vram_upload
    print(f"DiT (bf16) load: host {t_unet_host:.3f} s, vram upload {t_unet_vram_upload:.3f} s, total {t_unet_load_total:.3f} s")

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

    # 2. Text Encoder (positive prompt only, matching golem CFG 1)
    print("\n--- 2. Text Encoder ---")
    t0 = time.perf_counter()
    pos = nodes.CLIPTextEncode().encode(clip, PROMPT)[0]
    sync()
    t_enc_cold = time.perf_counter() - t0
    print(f"Text encode (cold, includes CLIP upload to VRAM): {t_enc_cold:.3f} s")

    enc_times = []
    for _ in range(4):
        t0 = time.perf_counter()
        pos = nodes.CLIPTextEncode().encode(clip, PROMPT)[0]
        sync()
        enc_times.append(time.perf_counter() - t0)
    t_enc_warm = min(enc_times)
    print(f"Text encode (warm, min): {t_enc_warm*1000:.1f} ms")

    # Empty negative prompt (cached once for CFG=1 sampler calls)
    neg = nodes.CLIPTextEncode().encode(clip, "")[0]

    # 3. VAE Decode (warm min over repeated runs)
    print("\n--- 3. VAE Decode ---")
    lat_1024 = EmptyFlux2LatentImage.execute(1024, 1024, 1)[0]
    _ = nodes.VAEDecode().decode(vae, lat_1024)[0]
    sync()
    dec_1024_times = []
    for _ in range(4):
        t0 = time.perf_counter()
        _ = nodes.VAEDecode().decode(vae, lat_1024)[0]
        sync()
        dec_1024_times.append(time.perf_counter() - t0)
    t_dec_1024_warm = min(dec_1024_times)
    print(f"VAE decode 1024x1024 (warm min): {t_dec_1024_warm*1000:.1f} ms")

    lat_768 = EmptyFlux2LatentImage.execute(768, 1024, 1)[0]
    _ = nodes.VAEDecode().decode(vae, lat_768)[0]
    sync()
    dec_768_times = []
    for _ in range(4):
        t0 = time.perf_counter()
        _ = nodes.VAEDecode().decode(vae, lat_768)[0]
        sync()
        dec_768_times.append(time.perf_counter() - t0)
    t_dec_768_warm = min(dec_768_times)
    print(f"VAE decode 768x1024 (warm min): {t_dec_768_warm*1000:.1f} ms")

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

    # Case (a): prompt changed each picture (text encode inside window; CLIP & DiT swap on 16 GB card)
    def run_e2e_changed(w, h, steps, prompt_text):
        sync()
        t0 = time.perf_counter()
        p = nodes.CLIPTextEncode().encode(clip, prompt_text)[0]
        p_unet = ModelSamplingFlux().patch(unet, SHIFT, SHIFT, w, h)[0]
        lat = EmptyFlux2LatentImage.execute(w, h, 1)[0]
        t_samp0 = time.perf_counter()
        samples = nodes.common_ksampler(p_unet, 42, steps, CFG, SAMPLER, SCHEDULER, p, neg, lat, denoise=1.0)[0]
        sync()
        t_sample = time.perf_counter() - t_samp0

        t_dec0 = time.perf_counter()
        _ = nodes.VAEDecode().decode(vae, samples)[0]
        sync()
        t_dec = time.perf_counter() - t_dec0
        t_total = time.perf_counter() - t0
        return t_total, t_sample, t_dec

    # Case (b): prompt unchanged (encode precomputed/cached, DiT stays on GPU without swapping)
    def run_e2e_cached(w, h, steps, cond):
        sync()
        t0 = time.perf_counter()
        p_unet = ModelSamplingFlux().patch(unet, SHIFT, SHIFT, w, h)[0]
        lat = EmptyFlux2LatentImage.execute(w, h, 1)[0]
        t_samp0 = time.perf_counter()
        samples = nodes.common_ksampler(p_unet, 42, steps, CFG, SAMPLER, SCHEDULER, cond, neg, lat, denoise=1.0)[0]
        sync()
        t_sample = time.perf_counter() - t_samp0

        t_dec0 = time.perf_counter()
        _ = nodes.VAEDecode().decode(vae, samples)[0]
        sync()
        t_dec = time.perf_counter() - t_dec0
        t_total = time.perf_counter() - t0
        return t_total, t_sample, t_dec

    # 4. Benchmark runs at 1024x1024
    print("\n--- 4. Generation 1024x1024 ---")
    # Untimed warmup run per size (matching golem)
    print("Warmup 1024x1024 (untimed)...")
    _ = run_e2e_changed(1024, 1024, 16, PROMPT)

    # Case (a): prompt changed each picture (CLIP and DiT swap on 16 GB)
    print("\nMeasuring Case (a): prompt changed each picture (swapping)...")
    step_times.clear()
    t_a_1024_16, t_samp_a_1024_16, _ = run_e2e_changed(1024, 1024, 16, PROMPT + " variation 1")
    print(f"Case (a) 1024x1024 (16 steps, prompt changed): total {t_a_1024_16:.3f} s (sample {t_samp_a_1024_16:.3f} s)")

    step_times.clear()
    t_a_1024_8, t_samp_a_1024_8, _ = run_e2e_changed(1024, 1024, 8, PROMPT + " variation 2")
    print(f"Case (a) 1024x1024 (8 steps, prompt changed): total {t_a_1024_8:.3f} s (sample {t_samp_a_1024_8:.3f} s)")

    # Case (b): prompt unchanged (encode cached by ComfyUI, no swap)
    print("\nMeasuring Case (b): prompt unchanged (steady state, no swap)...")
    comfy.model_management.load_models_gpu([unet])
    sync()

    step_times.clear()
    t_b_1024_16, t_samp_b_1024_16, _ = run_e2e_cached(1024, 1024, 16, pos)
    step_1024_steady = min(step_times) if step_times else (t_samp_b_1024_16 / 16)
    print(f"Case (b) 1024x1024 (16 steps, prompt cached): total {t_b_1024_16:.3f} s (sample {t_samp_b_1024_16:.3f} s, best step {step_1024_steady:.3f} s)")

    step_times.clear()
    t_b_1024_8, t_samp_b_1024_8, _ = run_e2e_cached(1024, 1024, 8, pos)
    print(f"Case (b) 1024x1024 (8 steps, prompt cached): total {t_b_1024_8:.3f} s (sample {t_samp_b_1024_8:.3f} s)")

    # 5. Benchmark runs at 768x1024
    print("\n--- 5. Generation 768x1024 ---")
    # Untimed warmup run per size (matching golem)
    print("Warmup 768x1024 (untimed)...")
    _ = run_e2e_changed(768, 1024, 16, PROMPT)

    # Case (a): prompt changed each picture (CLIP and DiT swap on 16 GB)
    print("\nMeasuring Case (a): prompt changed each picture (swapping)...")
    step_times.clear()
    t_a_768_16, t_samp_a_768_16, _ = run_e2e_changed(768, 1024, 16, PROMPT + " variation 3")
    print(f"Case (a) 768x1024 (16 steps, prompt changed): total {t_a_768_16:.3f} s (sample {t_samp_a_768_16:.3f} s)")

    step_times.clear()
    t_a_768_8, t_samp_a_768_8, _ = run_e2e_changed(768, 1024, 8, PROMPT + " variation 4")
    print(f"Case (a) 768x1024 (8 steps, prompt changed): total {t_a_768_8:.3f} s (sample {t_samp_a_768_8:.3f} s)")

    # Case (b): prompt unchanged (encode cached by ComfyUI, no swap)
    print("\nMeasuring Case (b): prompt unchanged (steady state, no swap)...")
    comfy.model_management.load_models_gpu([unet])
    sync()

    step_times.clear()
    t_b_768_16, t_samp_b_768_16, _ = run_e2e_cached(768, 1024, 16, pos)
    step_768_steady = min(step_times) if step_times else (t_samp_b_768_16 / 16)
    print(f"Case (b) 768x1024 (16 steps, prompt cached): total {t_b_768_16:.3f} s (sample {t_samp_b_768_16:.3f} s, best step {step_768_steady:.3f} s)")

    step_times.clear()
    t_b_768_8, t_samp_b_768_8, _ = run_e2e_cached(768, 1024, 8, pos)
    print(f"Case (b) 768x1024 (8 steps, prompt cached): total {t_b_768_8:.3f} s (sample {t_samp_b_768_8:.3f} s)")

    h1.remove()
    h2.remove()

    print("\n=== SUMMARY TABLE ===")
    print(f"Text Encoder (warm, positive): {t_enc_warm*1000:.1f} ms")
    print(f"Text Encoder (cold, includes CLIP upload to VRAM): {t_enc_cold*1000:.1f} ms")
    print(f"DiT Load (host + VRAM): {t_unet_load_total:.2f} s (host: {t_unet_host:.2f} s, vram upload: {t_unet_vram_upload:.2f} s)")
    print(f"VAE Decode: 1024x1024 (warm min): {t_dec_1024_warm*1000:.1f} ms, 768x1024 (warm min): {t_dec_768_warm*1000:.1f} ms")
    print(f"DiT Step (1024x1024, steady state without swap): {step_1024_steady:.3f} s")
    print(f"DiT Step (768x1024, steady state without swap): {step_768_steady:.3f} s")
    print("Case (a): prompt changed each picture (CLIP and DiT swap on 16 GB):")
    print(f"  1024x1024 (16 steps): {t_a_1024_16:.2f} s")
    print(f"  1024x1024 (8 steps):  {t_a_1024_8:.2f} s")
    print(f"  768x1024 (16 steps):  {t_a_768_16:.2f} s")
    print(f"  768x1024 (8 steps):   {t_a_768_8:.2f} s")
    print("Case (b): prompt unchanged (encode cached by ComfyUI, no swap):")
    print(f"  1024x1024 (16 steps): {t_b_1024_16:.2f} s")
    print(f"  1024x1024 (8 steps):  {t_b_1024_8:.2f} s")
    print(f"  768x1024 (16 steps):  {t_b_768_16:.2f} s")
    print(f"  768x1024 (8 steps):   {t_b_768_8:.2f} s")


if __name__ == "__main__":
    main()
