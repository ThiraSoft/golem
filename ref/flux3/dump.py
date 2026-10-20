"""Records what ComfyUI does with FLUX 3 Image Klein, for flux3/'s tests.

Drives ComfyUI's own nodes on its own three files:
- diffusion_models/flux3-image-klein-bf16.safetensors
- text_encoders/qwen3vl_4b_fp8_scaled.safetensors
- vae/flux2-vae.safetensors

and writes what goes in and out of every stage as raw little-endian float32
(or int32 for token ids), with meta.json giving each file's shape.

    /mnt/data/dev/ComfyUI/.venv/bin/python ref/flux3/dump.py \
        /mnt/data/dev/ComfyUI testdata/flux3 [stage ...]

Stages:
- tokens: three prompts (short, exactly 80 tokens, 130 tokens)
- sample: 256x256 t2i, seed 7, steps 16, euler/simple, alpha 16.0 (also records dit/)
- sample_cfg: 256x256 cfg 2 run with empty negative, seed 7
- vae: decode of sample/latent with up-block waypoints; encode of toast at 256x256
- i2i: i2i at 256x256 seed 7 with toast reference (ref_latent, ref_states, dit.out0, rgb)
- mi2i: two references (toast 256x256, great_wave 256x384) (ref_latents, ref_ids, rgb)
- accept: 1024x1024 t2i, seed 42, steps 16, the PNG
- t2i4: 2048x2048 t2i, seed 42 (first DiT call, latent, rgb, timings.json)
- i2i4: a phone photo with EXIF orientation 6, size matched at 4 MP, one
  reference at 1 MP, seed 7
- mi2i4: that photo matched plus three more references, 1 MP each, seed 7

The 4 MP stages are not in the default list.
"""

import json
import math
import os
import shutil
import time
import sys

import numpy as np
from PIL import Image, ImageOps

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
COMFY = sys.argv[1] if len(sys.argv) > 1 else "/mnt/data/dev/ComfyUI-flux3"
OUT = os.path.abspath(sys.argv[2]) if len(sys.argv) > 2 else os.path.join(ROOT, "testdata/flux3")
STAGES = sys.argv[3:] or ["tokens", "sample", "sample_cfg", "vae", "i2i", "mi2i", "accept"]

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
import node_helpers  # noqa: E402
import nodes  # noqa: E402
import comfy.sample  # noqa: E402
import comfy.samplers  # noqa: E402
from comfy_extras.nodes_edit_model import ReferenceMultiLatent  # noqa: E402
from comfy_extras.nodes_flux import EmptyFlux2LatentImage  # noqa: E402
from comfy_extras.nodes_model_advanced import ModelSamplingFlux  # noqa: E402
from comfy_extras.nodes_post_processing import ImageScaleToTotalPixels  # noqa: E402

UNET = "flux3-image-klein-bf16.safetensors"
CLIP = "qwen3vl_4b_fp8_scaled.safetensors"
VAE = "flux2-vae.safetensors"

# Upstream workflow defaults from ref/flux3/workflows/flux3_klein_bf16_t2i.json
STEPS = 16
CFG = 1.0
SAMPLER = "euler"
SCHEDULER = "simple"
ALPHA = 16.0
SHIFT = math.log(ALPHA)  # 2.77259

PROMPTS = {
    # p0: short prompt (< 80 unpadded tokens, pads to 80)
    "p0": "a cat",
    # p1: exactly 80 tokens unpadded (8 template + 72 text tokens = 80 tokens, 0 pads)
    "p1": (
        "a beautiful oil painting of a peaceful mountain landscape with a calm blue lake reflection "
        "tall green pine trees snow capped peaks soft morning mist golden sunlight breaking through the clouds "
        "flying birds in the distant sky small wooden cabin on the shore blooming wildflowers rough gray rocks "
        "detailed nature scene artstation concept art highly detailed sharp focus studio lighting vivid colors "
        "8k resolution cinematic atmospheric composition"
    ),
    # p2: 130 tokens unpadded (8 template + 122 text tokens = 130 tokens, pads to 160)
    "p2": (
        "a beautiful oil painting of a peaceful mountain landscape with a calm blue lake reflection "
        "tall green pine trees snow capped peaks soft morning mist golden sunlight breaking through the clouds "
        "flying birds in the distant sky small wooden cabin on the shore blooming wildflowers rough gray rocks "
        "detailed nature scene artstation concept art highly detailed sharp focus studio lighting vivid colors "
        "8k resolution cinematic atmospheric composition, ultra realistic digital illustration of an ancient "
        "mossy stone bridge spanning over a crystal clear rushing river with smooth rounded pebbles and autumn leaves, "
        "dramatic shadows casting across the lush valley during sunset with vibrant orange and purple gradient skies, "
        "epic fantasy masterpiece artwork"
    ),
}

PROMPT_ACCEPT = (
    "close up profile of a young male athlete with closed eyes pressing his "
    "sweaty forehead against a soccer ball, right hand gripping the ball, "
    "stark white background, athletic jacket, white ball with jagged red graphic "
    "patterns, high contrast studio lighting, highly detailed skin texture, "
    "sharp focus Absolutely no text, no typography, no letters, no words, "
    "no numbers, no logos, no watermark, no captions, pure imagery only."
)

PROMPT_I2I = "spread strawberry jam on the toast and add a few fresh blueberries on the plate"
PROMPT_MI2I = "a breakfast table with toast in front of the great wave"

REF_TOAST = os.path.join(ROOT, "ref/flux3/workflows/klein_ref_toast.jpg")
REF_GREAT_WAVE = os.path.join(ROOT, "ref/flux3/workflows/klein_ref_great_wave.jpg")

# The 4 MP stages: the workflows' ResolutionSelector and ImageScaleToTotalPixels
# at 4.0 MP, references at ReferenceMultiLatent's 1.0 MP each.
PROMPT_I2I4 = "turn the photo into a watercolor painting, keep the composition"
PROMPT_MI2I4 = (
    "the scene from the first image, with the toast from the second image on a table, "
    "painted in the colours of the third and fourth images"
)
# The photos sit in the main ComfyUI checkout, whatever COMFY the nodes come from.
INPUTS = "/mnt/data/dev/ComfyUI/input"
PHONE_PHOTO = os.path.join(INPUTS, "IMG_20191030_202255.jpg")
MI2I4_EXTRA = [
    (REF_TOAST, "ref1.jpg"),
    (os.path.join(INPUTS, "eyes_renarde.png"), "ref2.png"),
    (os.path.join(INPUTS, "Capture d'écran_20260803_084025.png"), "ref3.png"),
]

meta = {}


def save(rel, t):
    path = os.path.join(OUT, rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if isinstance(t, np.ndarray):
        a = np.ascontiguousarray(t)
    else:
        a = t.detach().float().cpu().contiguous().numpy()
    a.tofile(path)
    meta[rel] = list(a.shape)


def save_i32(rel, a):
    path = os.path.join(OUT, rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    arr = np.array(a, dtype=np.int32)
    arr.tofile(path)
    meta[rel] = list(arr.shape)


def cleanup_stale_files(out_dir):
    stale_patterns = [
        "sample/init", "sample/image", "sample/image.png",
        "sample_cfg/init", "sample_cfg/image", "sample_cfg/image.png",
        "i2i/init", "i2i/image", "i2i/image.png",
        "mi2i/init", "mi2i/image", "mi2i/image.png",
        "accept/init", "accept/image", "accept/image.png",
        "vae/z",
        "mi2i/ref_latent0", "mi2i/ref_latent1",
        "dit32/in.x", "dit32/in.t", "dit32/in.ctx", "dit32/in.next", "dit32/pe",
    ]
    for rel in stale_patterns:
        p = os.path.join(out_dir, rel)
        if os.path.exists(p):
            try:
                os.remove(p)
                print(f"Removed stale file: {rel}")
            except OSError:
                pass

    import re
    re_stale = [
        re.compile(r"^[^/]+/(x|denoised)\d+$"),
        re.compile(r"^vae/up\d+$"),
    ]
    for root, _, files in os.walk(out_dir):
        for f in files:
            rel = os.path.relpath(os.path.join(root, f), out_dir)
            if any(rx.match(rel) for rx in re_stale):
                try:
                    os.remove(os.path.join(out_dir, rel))
                    print(f"Removed stale file: {rel}")
                except OSError:
                    pass


def write_meta():
    cleanup_stale_files(OUT)
    path = os.path.join(OUT, "meta.json")
    old = {}
    if os.path.exists(path):
        with open(path) as f:
            old = json.load(f)

    import re
    stale_keys = {
        "sample/init", "sample/image",
        "sample_cfg/init", "sample_cfg/image",
        "i2i/init", "i2i/image",
        "mi2i/init", "mi2i/image",
        "accept/init", "accept/image",
        "vae/z",
        "mi2i/ref_latent0", "mi2i/ref_latent1",
        "dit32/in.x", "dit32/in.t", "dit32/in.ctx", "dit32/in.next", "dit32/pe",
    }
    re_stale_key = re.compile(r"^[^/]+/(x|denoised)\d+$|^vae/up\d+$")
    old = {k: v for k, v in old.items() if k not in stale_keys and not re_stale_key.match(k)}

    stage_prefixes = {
        "tokens": ["tokens/", "encoder/"],
        "sample": ["sample/", "dit/"],
        "sample_cfg": ["sample_cfg/"],
        "vae": ["vae/"],
        "i2i": ["i2i/"],
        "mi2i": ["mi2i/"],
        "accept": ["accept/"],
    }
    for stg in STAGES:
        prefixes = stage_prefixes.get(stg, [f"{stg}/"])
        old = {k: v for k, v in old.items() if not any(k.startswith(pfx) for pfx in prefixes)}

    old.update(meta)
    # Only keep entries that actually exist as files on disk
    final_meta = {k: v for k, v in old.items() if os.path.isfile(os.path.join(OUT, k))}
    with open(path, "w") as f:
        json.dump(final_meta, f, indent=1, sort_keys=True)


def load_models():
    print(f"Loading UNET: {UNET}")
    unet = nodes.UNETLoader().load_unet(UNET, "default")[0]
    print(f"Loading CLIP: {CLIP}")
    clip = nodes.CLIPLoader().load_clip(CLIP, "flux3", "default")[0]
    print(f"Loading VAE: {VAE}")
    vae = nodes.VAELoader().load_vae(VAE)[0]
    return unet, clip, vae


def stage_tokens(clip):
    print("=== Stage: tokens ===")
    tm = clip.cond_stage_model.qwen3vl_4b.transformer.model
    for key in ("p0", "p1", "p2"):
        text = PROMPTS[key]
        toks = clip.tokenize(text)["qwen3vl_4b"][0]
        ids = [int(t[0]) for t in toks]
        unpadded = [t for t in toks if t[0] != 151643]

        rec = {}

        def pre_hook(m, args, kwargs):
            x = args[0] if len(args) > 0 else kwargs.get("x")
            rec["hidden_l4"] = x.clone()

        h = tm.layers[4].register_forward_pre_hook(pre_hook, with_kwargs=True)
        cond = clip.encode_from_tokens_scheduled(clip.tokenize(text))
        h.remove()

        cond_tensor = cond[0][0][0]  # shape [T, 20480]
        hidden_l4 = rec["hidden_l4"][0]  # shape [T, 2560]

        save_i32(f"tokens/{key}.ids", ids)
        save(f"encoder/{key}.hidden_l4", hidden_l4)
        save(f"encoder/{key}.cond", cond_tensor)

        os.makedirs(os.path.join(OUT, "tokens"), exist_ok=True)
        with open(os.path.join(OUT, "tokens", f"{key}.json"), "w") as f:
            json.dump({
                "name": key,
                "text": text,
                "unpadded_len": len(unpadded),
                "total_len": len(ids),
                "ids": ids,
            }, f, indent=2)

        print(f"tokens {key}: unpadded={len(unpadded)} total={len(ids)} cond={tuple(cond_tensor.shape)}")


def run_sampling(unet, clip, vae, prompt, width, height, steps, cfg, seed, tag,
                 record_dit=False, extra_conds_fn=None, negative="",
                 pos=None, neg=None, waypoints=True, trajectory=True, dit_prefix="dit",
                 timings=None):
    print(f"=== Sampling: {tag} ({width}x{height}, seed={seed}, cfg={cfg}, steps={steps}) ===")
    rec = {"steps": []}
    drec = {}

    if pos is None:
        pos = nodes.CLIPTextEncode().encode(clip, prompt)[0]
    if neg is None:
        neg = nodes.CLIPTextEncode().encode(clip, negative)[0]
    if extra_conds_fn is not None:
        pos, neg = extra_conds_fn(pos, neg)

    patched_unet = ModelSamplingFlux().patch(unet, SHIFT, SHIFT, width, height)[0]
    latent = EmptyFlux2LatentImage.execute(width, height, 1)[0]

    real_prepare = comfy.sample.prepare_noise

    def prepare(latent_image, s, noise_inds=None):
        n = real_prepare(latent_image, s, noise_inds)
        rec["noise0"] = n.clone()
        return n

    comfy.sample.prepare_noise = prepare

    step_times = []

    def cb(step, x0, x, total):
        step_times.append(time.perf_counter())
        if trajectory:
            rec["steps"].append((step, x.clone(), x0.clone()))

    hooks = []
    dm = unet.model.diffusion_model

    if record_dit:
        def pre_dm(m, args, kwargs):
            if "in.x" not in drec:
                x_in = args[0] if len(args) > 0 else kwargs.get("x")
                t_in = args[1] if len(args) > 1 else kwargs.get("timestep")
                ctx_in = args[2] if len(args) > 2 else kwargs.get("context")
                opts = args[4] if len(args) > 4 else kwargs.get("transformer_options", {})
                drec["in.x"] = x_in.clone()
                drec["in.t"] = t_in.clone()
                drec["in.ctx"] = ctx_in.clone()
                sample_sigmas = opts.get("sample_sigmas", None)
                if sample_sigmas is not None:
                    i_sig = int((sample_sigmas - t_in[0]).abs().argmin())
                    target_sig = sample_sigmas[min(i_sig + 1, sample_sigmas.shape[0] - 1)]
                    drec["in.next"] = target_sig.reshape(1).clone()

        hooks.append(dm.register_forward_pre_hook(pre_dm, with_kwargs=True))

        def grab(name):
            def h(m, i, o):
                if name not in drec:
                    drec[name] = (o[0] if isinstance(o, tuple) else o).clone()
            return h

        if waypoints:
            hooks.append(dm.emb_in["image"].register_forward_hook(grab("wp.emb_img")))
            hooks.append(dm.txt_mode_blocks[-1].register_forward_hook(grab("wp.txt_early")))
            hooks.append(dm.content_mode_blocks["image"][-1].register_forward_hook(grab("wp.img_early")))
            hooks.append(dm.single_blocks[0].register_forward_hook(grab("wp.single_0")))
            hooks.append(dm.single_blocks[13].register_forward_hook(grab("wp.single_13")))
            hooks.append(dm.single_blocks[27].register_forward_hook(grab("wp.single_27")))
        hooks.append(dm.register_forward_hook(grab("out")))

    try:
        import latent_preview
        real_cb = latent_preview.prepare_callback

        def prep_cb(model, steps_cnt, x0_output_dict=None):
            return lambda step, x0, x, total: cb(step, x0, x, total)

        latent_preview.prepare_callback = prep_cb
        t_sample = time.perf_counter()
        samples = nodes.common_ksampler(
            patched_unet, seed, steps, cfg, SAMPLER, SCHEDULER, pos, neg, latent, denoise=1.0
        )[0]
        latent_preview.prepare_callback = real_cb
        # ComfyUI falls back to decode_tiled_ when a full decode does not fit:
        # record which one ran.
        tiled = []
        real_tiled = vae.decode_tiled_

        def decode_tiled_(*a, **kw):
            tiled.append(True)
            return real_tiled(*a, **kw)

        vae.decode_tiled_ = decode_tiled_
        t_decode = time.perf_counter()
        try:
            decoded = nodes.VAEDecode().decode(vae, samples)[0]
        finally:
            vae.decode_tiled_ = real_tiled
        t_end = time.perf_counter()
        if timings is not None:
            marks = [t_sample] + step_times
            timings["steps"] = [b - a for a, b in zip(marks, marks[1:])]
            timings["sample"] = t_decode - t_sample
            timings["decode"] = t_end - t_decode
            timings["decode_mode"] = "tiled" if tiled else "full"
            timings["end"] = t_end
    finally:
        comfy.sample.prepare_noise = real_prepare
        for h in hooks:
            h.remove()

    ms = patched_unet.get_model_object("model_sampling")
    sigmas = comfy.samplers.calculate_sigmas(ms, SCHEDULER, steps)

    save(f"{tag}/sigmas", sigmas)
    if "noise0" in rec:
        save(f"{tag}/noise0", rec["noise0"])

    for step_idx, x_val, x0_val in rec["steps"]:
        save(f"{tag}/x_{step_idx}", x_val)
        save(f"{tag}/denoised_{step_idx}", x0_val)

    save(f"{tag}/latent", samples["samples"])
    save(f"{tag}/rgb", decoded[0])

    arr = 255.0 * decoded[0].cpu().numpy()
    img_pil = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    os.makedirs(os.path.join(OUT, tag), exist_ok=True)
    img_pil.save(os.path.join(OUT, tag, f"{tag}.png"))

    if record_dit and "in.x" in drec and not waypoints:
        for k, v in drec.items():
            save(f"{dit_prefix}/{k}", v)
    elif record_dit and "in.x" in drec:
        for k, v in drec.items():
            save(f"dit/{k}", v)
        # Compute pe for dit/
        dev = dm.emb_in["image"].weight.device
        h_lat, w_lat = drec["in.x"].shape[-2:]
        txt_len = drec["in.ctx"].shape[1]
        ids = torch.cat([dm.image_ids(h_lat, w_lat, 0, dev), dm.text_ids(txt_len, dev)])
        pe = dm.rope(ids)
        save("dit/pe", pe)

    with open(os.path.join(OUT, tag, "request.json"), "w") as f:
        json.dump({
            "prompt": prompt,
            "negative": negative,
            "width": width,
            "height": height,
            "steps": steps,
            "cfg": cfg,
            "seed": seed,
            "sampler": SAMPLER,
            "scheduler": SCHEDULER,
            "alpha": ALPHA,
            "shift": SHIFT,
        }, f, indent=2)

    print(f"{tag} done: shape={tuple(samples['samples'].shape)}, steps={len(rec['steps'])}")
    return samples, decoded, pos, neg


def stage_sample(unet, clip, vae):
    run_sampling(
        unet, clip, vae, PROMPTS["p0"], 256, 256, STEPS, CFG, 7, "sample", record_dit=True
    )


def stage_sample_cfg(unet, clip, vae):
    # cfg 2 run with empty negative at 256x256, seed 7, prompt p2 (> 80 tokens)
    samples, decoded, pos, neg = run_sampling(
        unet, clip, vae, PROMPTS["p2"], 256, 256, STEPS, 2.0, 7, "sample_cfg",
        record_dit=False, negative=""
    )
    save("sample_cfg/cond", pos[0][0][0])
    save("sample_cfg/uncond", neg[0][0][0])


def stage_vae(vae):
    print("=== Stage: vae ===")
    sample_latent_path = os.path.join(OUT, "sample", "latent")
    if not os.path.exists(sample_latent_path):
        raise RuntimeError("sample/latent must exist before running stage_vae")

    shape = [1, 128, 16, 16]
    z = torch.from_numpy(np.fromfile(sample_latent_path, dtype=np.float32).reshape(shape))

    # Decode of sample/latent with waypoints per up block
    vrecs = {}
    fm = vae.first_stage_model
    dec = fm.decoder
    hooks = []
    for i in (3, 2, 1):
        def make_h(idx):
            def h(m, a, o):
                val = (o[0] if isinstance(o, list) else o).clone()
                vrecs[f"up_{idx}"] = val
            return h
        hooks.append(dec.up[i].upsample.register_forward_hook(make_h(i)))

    def h0(m, a, o):
        val = (o[0] if isinstance(o, list) else o).clone()
        vrecs["up_0"] = val
    hooks.append(dec.up[0].block[-1].register_forward_hook(h0))

    try:
        decoded = nodes.VAEDecode().decode(vae, {"samples": z})[0]
    finally:
        for hk in hooks:
            hk.remove()

    for k, v in vrecs.items():
        save(f"vae/{k}", v)

    # Encode of klein_ref_toast.jpg via ReferenceMultiLatent (megapixels=0.0625)
    t_toast = torch.from_numpy(np.array(Image.open(REF_TOAST).convert("RGB")).astype(np.float32) / 255.0).unsqueeze(0)
    saved_imgs = []
    orig_encode = vae.encode

    def hooked_encode(img):
        saved_imgs.append(img)
        return orig_encode(img)

    vae.encode = hooked_encode
    try:
        res = {"resolution": "resize", "megapixels": 0.0625}
        out = ReferenceMultiLatent.execute(
            positive=[[torch.zeros(1), {}]],
            resolution=res,
            negative=[[torch.zeros(1), {}]],
            vae=vae,
            images={"image1": t_toast}
        )
    finally:
        vae.encode = orig_encode

    enc_in = saved_imgs[0][0]
    enc_latent = out[0][0][1]["reference_latents"][0]

    save("vae/enc.in", enc_in)
    save("vae/enc.latent", enc_latent)
    print(f"vae encode toast: in={tuple(enc_in.shape)} latent={tuple(enc_latent.shape)}")


def stage_i2i(unet, clip, vae):
    print("=== Stage: i2i ===")
    t_toast = torch.from_numpy(np.array(Image.open(REF_TOAST).convert("RGB")).astype(np.float32) / 255.0).unsqueeze(0)
    saved_imgs = []
    orig_encode = vae.encode

    def hooked_encode(img):
        saved_imgs.append(img)
        return orig_encode(img)

    vae.encode = hooked_encode
    try:
        res = {"resolution": "resize", "megapixels": 0.0625}
        pos = nodes.CLIPTextEncode().encode(clip, PROMPT_I2I)[0]
        neg = nodes.CLIPTextEncode().encode(clip, "")[0]
        out = ReferenceMultiLatent.execute(
            positive=pos,
            resolution=res,
            negative=neg,
            vae=vae,
            images={"image1": t_toast}
        )
        pos, neg, _ = out[0], out[1], out[2]
    finally:
        vae.encode = orig_encode

    ref_rgb = saved_imgs[0][0]
    ref_latent = pos[0][1]["reference_latents"][0]
    save("i2i/ref_rgb", ref_rgb)
    save("i2i/ref_latent", ref_latent)

    dm = unet.model.diffusion_model
    i2i_rec = {}
    hooks = []

    # A forward hook that returns a value replaces the module's output: these
    # return None, so ComfyUI's trajectory is left alone.
    def hook_ref(m, i, o):
        if "ref_states" not in i2i_rec:
            i2i_rec["ref_states"] = (o[0] if isinstance(o, tuple) else o).clone()

    def hook_dit(m, i, o):
        if "dit.out0" not in i2i_rec:
            i2i_rec["dit.out0"] = (o[0] if isinstance(o, tuple) else o).clone()

    hooks.append(dm.content_mode_blocks["image_cond"][-1].register_forward_hook(hook_ref))
    hooks.append(dm.register_forward_hook(hook_dit))

    try:
        run_sampling(
            unet, clip, vae, PROMPT_I2I, 256, 256, STEPS, CFG, 7, "i2i",
            record_dit=False, pos=pos, neg=neg
        )
    finally:
        for h in hooks:
            h.remove()

    if "ref_states" in i2i_rec:
        save("i2i/ref_states", i2i_rec["ref_states"])
    if "dit.out0" in i2i_rec:
        save("i2i/dit.out0", i2i_rec["dit.out0"])


def stage_mi2i(unet, clip, vae):
    print("=== Stage: mi2i ===")
    t_toast = torch.from_numpy(np.array(Image.open(REF_TOAST).convert("RGB")).astype(np.float32) / 255.0).unsqueeze(0)
    t_wave = torch.from_numpy(np.array(Image.open(REF_GREAT_WAVE).convert("RGB")).astype(np.float32) / 255.0).unsqueeze(0)
    saved_imgs = []
    orig_encode = vae.encode

    def hooked_encode(img):
        saved_imgs.append(img)
        return orig_encode(img)

    vae.encode = hooked_encode
    try:
        res = {"resolution": "resize", "megapixels": 0.0625}
        pos = nodes.CLIPTextEncode().encode(clip, PROMPT_MI2I)[0]
        neg = nodes.CLIPTextEncode().encode(clip, "")[0]
        out = ReferenceMultiLatent.execute(
            positive=pos,
            resolution=res,
            negative=neg,
            vae=vae,
            images={"image1": t_toast, "image2": t_wave}
        )
        pos, neg, _ = out[0], out[1], out[2]
    finally:
        vae.encode = orig_encode

    ref_rgb0 = saved_imgs[0][0]
    ref_rgb1 = saved_imgs[1][0]
    save("mi2i/ref_rgb0", ref_rgb0)
    save("mi2i/ref_rgb1", ref_rgb1)

    ref_lats = pos[0][1]["reference_latents"]
    ref_latents_cat = torch.cat([lat.flatten(2).transpose(1, 2) for lat in ref_lats], dim=1)
    save("mi2i/ref_latents", ref_latents_cat)

    dm = unet.model.diffusion_model
    ref_shapes = [tuple(lat.shape[-2:]) for lat in ref_lats]
    ref_ids = dm.ref_ids(ref_shapes, "cpu")
    save("mi2i/ref_ids", ref_ids)

    run_sampling(
        unet, clip, vae, PROMPT_MI2I, 256, 256, STEPS, CFG, 7, "mi2i",
        record_dit=False, pos=pos, neg=neg
    )


def stage_accept(unet, clip, vae):
    print("=== Stage: accept ===")
    run_sampling(
        unet, clip, vae, PROMPT_ACCEPT, 1024, 1024, STEPS, CFG, 42, "accept", record_dit=False
    )


def to_tensor(im):
    return torch.from_numpy(np.array(im.convert("RGB")).astype(np.float32) / 255.0).unsqueeze(0)


def write_phone_exif6(path):
    """The phone photo re-saved with EXIF Orientation 6: oriented, a portrait."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    im = Image.open(PHONE_PHOTO)
    exif = im.getexif()
    exif[0x0112] = 6
    im.save(path, quality=95, exif=exif)
    oriented = ImageOps.exif_transpose(Image.open(path))
    assert oriented.size == (3120, 4160), oriented.size
    return oriented


def run_4mp(unet, clip, vae, tag, prompt, seed, images):
    """One 4 MP stage. images: oriented PIL images, the first one matched."""
    print(f"=== Stage: {tag} ===")
    timings = {}
    t0 = time.perf_counter()
    width, height = 2048, 2048
    ref_shapes = []
    pos = nodes.CLIPTextEncode().encode(clip, prompt)[0]
    neg = nodes.CLIPTextEncode().encode(clip, "")[0]
    if images:
        scaled = ImageScaleToTotalPixels.execute(to_tensor(images[0]), "lanczos", 4.0, 16)[0]
        height, width = int(scaled.shape[1]), int(scaled.shape[2])
        refs = {"image1": scaled}
        for i, im in enumerate(images[1:]):
            refs[f"image{i + 2}"] = to_tensor(im)
        saved = []
        orig_encode = vae.encode

        def hooked_encode(img):
            saved.append(img)
            return orig_encode(img)

        vae.encode = hooked_encode
        try:
            out = ReferenceMultiLatent.execute(
                positive=pos,
                resolution={"resolution": "resize", "megapixels": 1.0},
                negative=neg,
                vae=vae,
                images=refs,
            )
            pos, neg = out[0], out[1]
        finally:
            vae.encode = orig_encode
        save(f"{tag}/ref_rgb0", saved[0][0])
        lats = pos[0][1]["reference_latents"]
        ref_shapes = [list(lat.shape[-2:]) for lat in lats]
        save(f"{tag}/ref_latents", torch.cat([lat.flatten(2).transpose(1, 2) for lat in lats], dim=1))
        dm = unet.model.diffusion_model
        save(f"{tag}/ref_ids", dm.ref_ids([tuple(s) for s in ref_shapes], "cpu"))

    run_sampling(
        unet, clip, vae, prompt, width, height, STEPS, CFG, seed, tag,
        record_dit=True, pos=pos, neg=neg, waypoints=False, trajectory=False,
        dit_prefix=tag, timings=timings,
    )
    timings["total"] = timings.pop("end") - t0
    with open(os.path.join(OUT, tag, "timings.json"), "w") as f:
        json.dump(timings, f, indent=2)
    req_path = os.path.join(OUT, tag, "request.json")
    with open(req_path) as f:
        req = json.load(f)
    req["ref_shapes"] = ref_shapes
    with open(req_path, "w") as f:
        json.dump(req, f, indent=2)
    print(f"{tag}: {width}x{height}, refs {ref_shapes}, timings {timings}")


def stage_t2i4(unet, clip, vae):
    run_4mp(unet, clip, vae, "t2i4", PROMPT_ACCEPT, 42, [])


def stage_i2i4(unet, clip, vae):
    phone = write_phone_exif6(os.path.join(OUT, "i2i4", "ref0.jpg"))
    run_4mp(unet, clip, vae, "i2i4", PROMPT_I2I4, 7, [phone])


def stage_mi2i4(unet, clip, vae):
    images = [write_phone_exif6(os.path.join(OUT, "mi2i4", "ref0.jpg"))]
    for src, name in MI2I4_EXTRA:
        dst = os.path.join(OUT, "mi2i4", name)
        shutil.copyfile(src, dst)
        images.append(ImageOps.exif_transpose(Image.open(dst)))
    run_4mp(unet, clip, vae, "mi2i4", PROMPT_MI2I4, 7, images)


def main():
    torch.set_grad_enabled(False)
    unet, clip, vae = load_models()
    if "tokens" in STAGES:
        stage_tokens(clip)
    if "sample" in STAGES:
        stage_sample(unet, clip, vae)
    if "sample_cfg" in STAGES:
        stage_sample_cfg(unet, clip, vae)
    if "vae" in STAGES:
        stage_vae(vae)
    if "i2i" in STAGES:
        stage_i2i(unet, clip, vae)
    if "mi2i" in STAGES:
        stage_mi2i(unet, clip, vae)
    if "accept" in STAGES:
        stage_accept(unet, clip, vae)
    if "t2i4" in STAGES:
        stage_t2i4(unet, clip, vae)
    if "i2i4" in STAGES:
        stage_i2i4(unet, clip, vae)
    if "mi2i4" in STAGES:
        stage_mi2i4(unet, clip, vae)
    write_meta()
    print("dump.py complete. meta.json written.")


if __name__ == "__main__":
    main()
