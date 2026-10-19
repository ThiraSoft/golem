"""The DiT call dump.py recorded, again in float32 on the CPU.

ComfyUI runs the DiT in bf16 on ROCm/CUDA. This script runs ComfyUI's own
modules (comfy/ldm/flux3/model.py) in float32, one block at a time so that
the parameters never sit in memory together, on the inputs dump.py recorded,
and writes the same waypoints under dit32/.

    /mnt/data/dev/ComfyUI/.venv/bin/python ref/flux3/dump_f32.py \
        /mnt/data/dev/ComfyUI testdata/flux3 [stage]

With "dit" (default), it reads dit/in.* and writes dit32/ with the comparison
to the bf16 reference.
"""

import json
import math
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
COMFY = sys.argv[1] if len(sys.argv) > 1 else "/mnt/data/dev/ComfyUI-flux3"
OUT = os.path.abspath(sys.argv[2]) if len(sys.argv) > 2 else os.path.join(ROOT, "testdata/flux3")

sys.path.insert(0, COMFY)
os.chdir(COMFY)
sys.argv = [sys.argv[0], "--cpu"]

import torch  # noqa: E402
from safetensors import safe_open  # noqa: E402

import folder_paths  # noqa: E402
import comfy.model_management  # noqa: E402
import comfy.options  # noqa: E402
comfy.options.enable_args_parsing()
import comfy.ops  # noqa: E402
from comfy.ldm.flux.layers import EmbedND, LastLayer, MLPEmbedder, Modulation, timestep_embedding  # noqa: E402
from comfy.ldm.flux3.model import Block  # noqa: E402

torch.set_grad_enabled(False)
comfy.model_management.in_training = True  # use pure-torch apply_rope1 on CPU
ops = comfy.ops.disable_weight_init
F32 = torch.float32

unet_path = folder_paths.get_full_path("diffusion_models", "flux3-image-klein-bf16.safetensors")
if not unet_path:
    unet_path = os.path.join(COMFY, "models/diffusion_models/flux3-image-klein-bf16.safetensors")
UNET = unet_path

with open(os.path.join(OUT, "meta.json")) as f:
    meta = json.load(f)


def read(name):
    path = os.path.join(OUT, name)
    arr = np.fromfile(path, dtype=np.float32).reshape(meta[name])
    return torch.from_numpy(arr)


def save(name, t):
    a = t.detach().float().contiguous().numpy()
    path = os.path.join(OUT, name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    a.tofile(path)
    meta[name] = list(a.shape)


def rms_diff(a, b):
    # relative RMS error
    diff = (a.float() - b.float())
    return (torch.sqrt(torch.mean(diff ** 2)) / (torch.sqrt(torch.mean(b.float() ** 2)) + 1e-12)).item()


st = safe_open(UNET, "pt")


def load_block(prefix):
    blk = Block(3072, 24, 3.0, fused_qkv=False, dtype=F32, device="cpu", operations=ops)
    sd = {}
    for k in [
        "q_proj.weight", "k_proj.weight", "v_proj.weight",
        "mlp_in.weight", "attn_out.weight", "mlp_out.weight"
    ]:
        sd[k] = st.get_tensor(prefix + k).to(F32)

    q_scale_key = prefix + "norm.query_norm.scale"
    if q_scale_key in st.keys():
        sd["norm.query_norm.weight"] = st.get_tensor(q_scale_key).to(F32)
    else:
        sd["norm.query_norm.weight"] = st.get_tensor(prefix + "norm.query_norm.weight").to(F32)

    k_scale_key = prefix + "norm.key_norm.scale"
    if k_scale_key in st.keys():
        sd["norm.key_norm.weight"] = st.get_tensor(k_scale_key).to(F32)
    else:
        sd["norm.key_norm.weight"] = st.get_tensor(prefix + "norm.key_norm.weight").to(F32)

    missing, unexp = blk.load_state_dict(sd, strict=False)
    assert not missing, f"Missing keys in {prefix}: {missing}"
    return blk


def main():
    print("Running DiT in float32 on CPU...")
    x = read("dit/in.x").to(F32)
    t = read("dit/in.t").to(F32)
    context = read("dit/in.ctx").to(F32)
    t_next = read("dit/in.next").to(F32)
    pe = read("dit/pe").to(F32)

    for stale in ["dit32/in.x", "dit32/in.t", "dit32/in.ctx", "dit32/in.next", "dit32/pe"]:
        meta.pop(stale, None)
        p = os.path.join(OUT, stale)
        if os.path.exists(p):
            try:
                os.remove(p)
            except OSError:
                pass

    # 1. Image embedding
    emb_img = ops.Linear(128, 3072, bias=False, dtype=F32)
    emb_img.load_state_dict({"weight": st.get_tensor("emb_in.image.weight").to(F32)})
    img = emb_img(x.flatten(2).transpose(1, 2))  # [1, 256, 3072]
    save("dit32/wp.emb_img", img)

    # 2. Timestep embeddings (flow_map: vec + vec_target)
    ref_time_in = MLPEmbedder(256, 3072, bias=False, dtype=F32, operations=ops)
    ref_time_in.load_state_dict({
        "in_layer.weight": st.get_tensor("ref_time_in.in_layer.weight").to(F32),
        "out_layer.weight": st.get_tensor("ref_time_in.out_layer.weight").to(F32),
    })
    vec_target = ref_time_in(timestep_embedding(t_next, 256).to(F32))

    time_in = MLPEmbedder(256, 3072, bias=False, dtype=F32, operations=ops)
    time_in.load_state_dict({
        "in_layer.weight": st.get_tensor("time_in.in_layer.weight").to(F32),
        "out_layer.weight": st.get_tensor("time_in.out_layer.weight").to(F32),
    })
    t_cat = torch.cat([t, t.new_zeros(1)]).to(F32)
    vec = time_in(timestep_embedding(t_cat, 256).to(F32))
    vec = vec + vec_target
    vec_img, vec_zero = vec[:-1], vec[-1:]

    # 3. Text embedding
    txt_in = ops.Linear(20480, 3072, bias=False, dtype=F32)
    txt_in.load_state_dict({"weight": st.get_tensor("txt_in.weight").to(F32)})
    txt = txt_in(context)

    # 4. Text early stream
    early_mod_txt = Modulation(3072, double=False, bias=False, dtype=F32, operations=ops)
    early_mod_txt.load_state_dict({"lin.weight": st.get_tensor("early_stream_modulations.txt.lin.weight").to(F32)})
    mod_txt = early_mod_txt(vec_zero)[0]

    dm_pe = EmbedND(dim=128, theta=10000, axes_dim=[32, 32, 32, 32])
    text_ids = torch.zeros(txt.shape[1], 4, dtype=torch.float32)
    text_ids[:, 3] = torch.arange(txt.shape[1], dtype=torch.float32)
    pe_txt = dm_pe(text_ids.unsqueeze(0)).transpose(1, 2).contiguous()

    for i in range(5):
        blk = load_block(f"txt_mode_blocks.{i}.")
        txt = blk(txt, [(0, txt.shape[1], mod_txt)], pe_txt)
        del blk

    save("dit32/wp.txt_early", txt)

    # 5. Image early stream
    early_mod_img = Modulation(3072, double=False, bias=False, dtype=F32, operations=ops)
    early_mod_img.load_state_dict({"lin.weight": st.get_tensor("early_stream_modulations.image.lin.weight").to(F32)})
    mod_img = early_mod_img(vec_img)[0]

    pe_img = pe[:, :256]
    for i in range(5):
        blk = load_block(f"content_mode_blocks.image.{i}.")
        img = blk(img, [(0, 256, mod_img)], pe_img)
        del blk

    save("dit32/wp.img_early", img)

    # 6. Joint single stream blocks
    single_mod_img = Modulation(3072, double=False, bias=False, dtype=F32, operations=ops)
    single_mod_img.load_state_dict({"lin.weight": st.get_tensor("single_stream_modulations.image.lin.weight").to(F32)})
    s_mod_img = single_mod_img(vec_img)[0]

    single_mod_txt = Modulation(3072, double=False, bias=False, dtype=F32, operations=ops)
    single_mod_txt.load_state_dict({"lin.weight": st.get_tensor("single_stream_modulations.txt.lin.weight").to(F32)})
    s_mod_txt = single_mod_txt(vec_zero)[0]

    spans = [
        (0, 256, s_mod_img),
        (256, 336, s_mod_txt),
    ]
    combined = torch.cat([img, txt], dim=1)

    for i in range(28):
        blk = load_block(f"single_blocks.{i}.")
        combined = blk(combined, spans, pe)
        if i in (0, 13, 27):
            save(f"dit32/wp.single_{i}", combined)
        del blk

    # 7. Final layer
    last_layer = LastLayer(3072, 1, 128, bias=False, dtype=F32, operations=ops)
    last_layer.load_state_dict({
        "linear.weight": st.get_tensor("final_layer.image.linear.weight").to(F32),
        "adaLN_modulation.1.weight": st.get_tensor("final_layer.image.adaLN_modulation.1.weight").to(F32),
    })
    final = last_layer(combined[:, :256], vec_img)
    out = final.transpose(1, 2).reshape(1, 128, 16, 16)
    save("dit32/out", out)

    with open(os.path.join(OUT, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1, sort_keys=True)

    print("\n=== RMS comparison between bf16 (dit/) and float32 (dit32/) ===")
    for k in [
        "wp.emb_img", "wp.txt_early", "wp.img_early",
        "wp.single_0", "wp.single_13", "wp.single_27", "out"
    ]:
        bf16_val = read(f"dit/{k}")
        f32_val = read(f"dit32/{k}")
        diff = rms_diff(f32_val, bf16_val)
        print(f"  {k:15s}: RMS gap = {diff * 100:.4f}%")


if __name__ == "__main__":
    main()
