"""Records what ComfyUI does with Krea 2, for krea2/'s tests.

Runs ComfyUI's own nodes on its own three files, the way the mobile front's
build_prompt wires them (LoRA off), and writes what goes in and out of every
stage as raw little-endian float32, with meta.json giving each file's shape.

    /mnt/data/dev/ComfyUI/.venv/bin/python ref/krea2/dump.py \
        /mnt/data/dev/ComfyUI testdata/krea2 [stage ...]

Stages: tokens, sample, accept (default: all three). sample also records one
DiT call and one VAE decode with their waypoints.
"""

import json
import os
import sys

import numpy as np

COMFY, OUT = sys.argv[1], os.path.abspath(sys.argv[2])
STAGES = sys.argv[3:] or ["tokens", "sample", "accept"]
sys.path.insert(0, COMFY)
os.chdir(COMFY)
# The flags start.sh runs ComfyUI with: they choose the dtypes and the attention.
sys.argv = [sys.argv[0], "--use-split-cross-attention", "--fp16-text-enc", "--fp16-vae",
            "--disable-mmap", "--reserve-vram", "1.5", "--deterministic"]

import torch  # noqa: E402

import comfy.options  # noqa: E402
comfy.options.enable_args_parsing()
import folder_paths  # noqa: E402,F401
import nodes  # noqa: E402
import comfy.k_diffusion.sampling as kds  # noqa: E402
import comfy.sample  # noqa: E402
import comfy.samplers  # noqa: E402

UNET = "krea2_turbo_fp8.safetensors"
CLIP = "qwen3vl_4b_fp8_scaled.safetensors"
VAE = "qwen_image_vae.safetensors"
NEGATIVE = "(ugly, anime, text, watermark, label, worst,sketch,censor, cg, cgi, rendered, 3d :1.0)"
PROMPTS = {
    "short": "a cat",
    "portrait": "portrait of a young woman with silver hair, gothic dress, facing the camera, plain grey background, soft light",
    "weighted": "a (red:1.3) cat on a ((wooden)) table, \\(film grain\\)",
}

meta = {}


def save(rel, t):
    path = os.path.join(OUT, rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    a = t.detach().float().cpu().contiguous().numpy()
    a.tofile(path)
    meta[rel] = list(a.shape)


def write_meta():
    path = os.path.join(OUT, "meta.json")
    old = {}
    if os.path.exists(path):
        with open(path) as f:
            old = json.load(f)
    old.update(meta)
    with open(path, "w") as f:
        json.dump(old, f, indent=1, sort_keys=True)


def first(x):
    return x[0] if isinstance(x, (tuple, list)) else x


def load():
    unet = nodes.UNETLoader().load_unet(UNET, "default")[0]
    clip = nodes.CLIPLoader().load_clip(CLIP, "krea2", "default")[0]
    vae = nodes.VAELoader().load_vae(VAE)[0]
    return unet, clip, vae


def text_model(clip):
    return clip.cond_stage_model.qwen3vl_4b.transformer.model


def stage_tokens(clip):
    tm = text_model(clip)
    for name, text in PROMPTS.items():
        toks = clip.tokenize(text)["qwen3vl_4b"]
        ids = [int(t[0]) for t in toks[0]]
        weights = [float(t[1]) for t in toks[0]]
        rec = {}
        hooks = [tm.embed_tokens.register_forward_hook(lambda m, i, o: rec.__setitem__("embed", o))]
        for n in (0, 1, 17, 34):
            hooks.append(tm.layers[n].register_forward_hook(
                lambda m, i, o, n=n: rec.__setitem__("layer%d" % n, first(o))))
        cond = clip.encode_from_tokens_scheduled(clip.tokenize(text))
        for h in hooks:
            h.remove()
        os.makedirs(os.path.join(OUT, "tokens"), exist_ok=True)
        with open(os.path.join(OUT, "tokens", name + ".json"), "w") as f:
            json.dump({"text": text, "ids": ids, "weights": weights}, f)
        for k, v in rec.items():
            save("encoder/%s/%s" % (name, k), v[0])
        save("encoder/%s/cond" % name, cond[0][0][0])
        print("tokens", name, len(ids), "cond", tuple(cond[0][0].shape))


def dit_hooks(unet, rec):
    dm = unet.model.diffusion_model
    hooks = []

    def pre(m, args, kwargs):
        if "in" not in rec:
            names = ["x", "timesteps", "context"]
            got = dict(zip(names, args))
            got.update({k: v for k, v in kwargs.items() if k in names})
            rec["in"] = tuple(got[n].clone() for n in names)
    hooks.append(dm.register_forward_pre_hook(pre, with_kwargs=True))

    def grab(name, fn=first):
        def h(m, i, o):
            if name not in rec:
                rec[name] = fn(o).clone()
        return h
    hooks.append(dm.first.register_forward_hook(grab("first")))
    hooks.append(dm.tproj.register_forward_hook(grab("tvec")))
    hooks.append(dm.tmlp.register_forward_hook(grab("tmlp")))
    hooks.append(dm.txtfusion.register_forward_hook(grab("txtfusion")))
    hooks.append(dm.txtmlp.register_forward_hook(grab("txtmlp")))
    hooks.append(dm.blocks[0].register_forward_hook(grab("block0")))
    hooks.append(dm.blocks[1].register_forward_hook(grab("block1")))
    hooks.append(dm.blocks[27].register_forward_hook(grab("block27")))
    hooks.append(dm.last.register_forward_hook(grab("last")))
    hooks.append(dm.register_forward_hook(grab("out")))
    return hooks


def vae_hooks(vae, rec):
    fm = vae.first_stage_model
    dec = fm.decoder
    hooks = []

    def grab(name):
        def h(m, i, o):
            if name not in rec:
                rec[name] = first(o).clone()
        return h

    def pre(m, args):
        if "z" not in rec:
            rec["z"] = args[0].clone()
    hooks.append(fm.conv2.register_forward_pre_hook(pre))
    hooks.append(fm.conv2.register_forward_hook(grab("conv2")))
    hooks.append(dec.conv1.register_forward_hook(grab("conv1")))
    for i, m in enumerate(dec.middle):
        hooks.append(m.register_forward_hook(grab("middle%d" % i)))
    for i, m in enumerate(dec.upsamples):
        hooks.append(m.register_forward_hook(grab("up%02d" % i)))
    hooks.append(dec.head.register_forward_hook(grab("head")))
    return hooks


def run(unet, clip, vae, prompt, width, height, steps, cfg, seed, tag, waypoints):
    rec = {"steps": [], "noise": []}
    pos = nodes.CLIPTextEncode().encode(clip, prompt)[0]
    neg = nodes.CLIPTextEncode().encode(clip, NEGATIVE)[0]
    latent = nodes.EmptyLatentImage().generate(width, height, 1)[0]
    save(tag + "/cond", pos[0][0][0])

    real_prepare = comfy.sample.prepare_noise

    def prepare(latent_image, seed, noise_inds=None):
        n = real_prepare(latent_image, seed, noise_inds)
        rec["init"] = n.clone()
        return n
    comfy.sample.prepare_noise = prepare

    real_default = kds.default_noise_sampler

    def default(x, seed=None):
        f = real_default(x, seed=seed)

        def g(s, sn):
            n = f(s, sn)
            rec["noise"].append(n.clone())
            return n
        return g
    kds.default_noise_sampler = default

    def cb(step, x0, x, total):
        rec["steps"].append((x.clone(), x0.clone()))

    hooks = []
    drec, vrec = {}, {}
    if waypoints:
        hooks += dit_hooks(unet, drec)
        hooks += vae_hooks(vae, vrec)
    try:
        import latent_preview
        real_cb = latent_preview.prepare_callback

        def prep_cb(model, steps, x0_output_dict=None):
            return lambda step, x0, x, total: cb(step, x0, x, total)
        latent_preview.prepare_callback = prep_cb
        samples = nodes.common_ksampler(unet, seed, steps, cfg, "er_sde", "simple", pos, neg, latent)[0]
        latent_preview.prepare_callback = real_cb
        image = nodes.VAEDecode().decode(vae, samples)[0]
    finally:
        comfy.sample.prepare_noise = real_prepare
        kds.default_noise_sampler = real_default
        for h in hooks:
            h.remove()


    ms = unet.get_model_object("model_sampling")
    sigmas = comfy.samplers.calculate_sigmas(ms, "simple", steps)
    save(tag + "/sigmas", sigmas)
    save(tag + "/init", rec["init"])
    for i, (x, x0) in enumerate(rec["steps"]):
        save(tag + "/x%d" % i, x)
        save(tag + "/denoised%d" % i, x0)
    for i, n in enumerate(rec["noise"]):
        save(tag + "/noise%d" % i, n)
    save(tag + "/latent", samples["samples"])
    save(tag + "/image", image[0])
    from PIL import Image
    arr = np.clip(255. * image[0].cpu().numpy(), 0, 255).astype(np.uint8)
    Image.fromarray(arr).save(os.path.join(OUT, tag, "image.png"))
    if waypoints:
        x, t, c = drec.pop("in")
        save("dit/x", x)
        save("dit/t", t)
        save("dit/context", c)
        for k, v in drec.items():
            save("dit/" + k, v)
        for k, v in vrec.items():
            save("vae/" + k, v)
    with open(os.path.join(OUT, tag, "request.json"), "w") as f:
        json.dump({"prompt": prompt, "negative": NEGATIVE, "width": width, "height": height,
                   "steps": steps, "cfg": cfg, "seed": seed}, f)
    print(tag, "done", tuple(samples["samples"].shape), "steps", len(rec["steps"]), "noise", len(rec["noise"]))


def main():
    unet, clip, vae = load()
    with torch.inference_mode():
        if "tokens" in STAGES:
            stage_tokens(clip)
        if "sample" in STAGES:
            run(unet, clip, vae, PROMPTS["short"], 256, 256, 8, 1.0, 7, "sample", True)
        if "accept" in STAGES:
            run(unet, clip, vae, PROMPTS["portrait"], 768, 1024, 8, 1.0, 42, "accept", False)
    write_meta()


main()
