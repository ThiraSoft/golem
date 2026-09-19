"""The DiT call dump.py recorded, again in float32 on the CPU.

ComfyUI runs the DiT in bf16, whose rounding at the size of the text
fusion's largest values is sixteen; a comparison against it cannot tell a
mistake from bf16. This runs ComfyUI's own modules (comfy/ldm/krea2/model.py)
in float32, one block at a time so that twelve billion parameters never sit
in memory together, on the inputs dump.py recorded, and writes the same
waypoints under dit32/.

    /mnt/data/dev/ComfyUI/.venv/bin/python ref/krea2/dump_f32.py \
        /mnt/data/dev/ComfyUI testdata/krea2
"""

import json
import os
import sys

import numpy as np

COMFY, OUT = sys.argv[1], os.path.abspath(sys.argv[2])
sys.path.insert(0, COMFY)
os.chdir(COMFY)
sys.argv = [sys.argv[0], "--cpu"]

import torch  # noqa: E402
from safetensors import safe_open  # noqa: E402

import comfy.options  # noqa: E402
comfy.options.enable_args_parsing()
import comfy.ops  # noqa: E402
from comfy.ldm.krea2 import model as K  # noqa: E402
from comfy.ldm.flux.layers import timestep_embedding  # noqa: E402

torch.set_grad_enabled(False)
ops = comfy.ops.disable_weight_init
F32 = torch.float32
UNET = os.path.join(COMFY, "models/unet/krea2_turbo_fp8.safetensors")

with open(os.path.join(OUT, "meta.json")) as f:
    meta = json.load(f)


def read(name):
    return torch.from_numpy(np.fromfile(os.path.join(OUT, name), dtype=np.float32).reshape(meta[name]))


def save(name, t):
    a = t.detach().float().contiguous().numpy()
    path = os.path.join(OUT, name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    a.tofile(path)
    meta[name] = list(a.shape)


st = safe_open(UNET, "pt")


def load(module, prefix):
    sd = {k[len(prefix):]: st.get_tensor(k).to(F32) for k in st.keys() if k.startswith(prefix)}
    missing, unexpected = module.load_state_dict(sd, strict=False)
    assert not missing, missing
    return module.to(F32)


dm = K.SingleStreamDiT.__new__(K.SingleStreamDiT)
torch.nn.Module.__init__(dm)
features, heads, kvheads, patch, channels = 6144, 48, 12, 2, 16
headdim = features // heads
axes = [headdim - 12 * (headdim // 16), 6 * (headdim // 16), 6 * (headdim // 16)]
dm.patch, dm.channels, dm.tdim, dm.txtlayers, dm.txtdim = patch, channels, 256, 12, 2560
dm.pe_embedder = K.EmbedND(dim=headdim, theta=1000, axes_dim=axes)

x = read("dit/x")[:, :, 0]
t = read("dit/t")
context = read("dit/context")

first = load(ops.Linear(64, features, bias=True, dtype=F32), "first.")
tmlp = load(torch.nn.Sequential(ops.Linear(256, features, dtype=F32), torch.nn.GELU(approximate="tanh"), ops.Linear(features, features, dtype=F32)), "tmlp.")
tproj = load(torch.nn.Sequential(torch.nn.GELU(approximate="tanh"), ops.Linear(features, features * 6, dtype=F32)), "tproj.")
txtfusion = load(K.TextFusionTransformer(12, 2560, 20, 4, False, 20, dtype=F32, operations=ops), "txtfusion.")
txtmlp = load(torch.nn.Sequential(K.RMSNorm(2560, dtype=F32, operations=ops), ops.Linear(2560, features, dtype=F32),
                                  torch.nn.GELU(approximate="tanh"), ops.Linear(features, features, dtype=F32)), "txtmlp.")
last = load(K.LastLayer(features, patch, channels, dtype=F32, operations=ops), "last.")

img, imgpos, h_, w_ = dm.process_img(x)
img = first(img)
save("dit32/first", img)
tt = tmlp(timestep_embedding(t, 256).unsqueeze(1).to(F32))
save("dit32/tmlp", tt)
tvec = tproj(tt)
save("dit32/tvec", tvec)
ctx = txtfusion(context.reshape(1, context.shape[1], 12, 2560))
save("dit32/txtfusion", ctx)
ctx = txtmlp(ctx)
save("dit32/txtmlp", ctx)

txtlen = ctx.shape[1]
combined = torch.cat((ctx, img), dim=1)
pos = torch.cat((torch.zeros(1, txtlen, 3), imgpos), dim=1)
freqs = dm.pe_embedder(pos)
for i in range(28):
    block = load(K.SingleStreamBlock(features, heads, 4, False, kvheads, dtype=F32, operations=ops), "blocks.%d." % i)
    combined = block(combined, tvec, freqs, None)
    if i in (0, 1, 27):
        save("dit32/block%d" % i, combined)
    del block
    print("block", i, flush=True)
final = last(combined, tt)
save("dit32/last", final)
out = final[:, txtlen:txtlen + img.shape[1]]
out = out.reshape(1, h_, w_, channels, patch, patch).permute(0, 3, 1, 4, 2, 5).reshape(1, channels, h_ * patch, w_ * patch)
save("dit32/out", out.unsqueeze(2))

with open(os.path.join(OUT, "meta.json"), "w") as f:
    json.dump(meta, f, indent=1, sort_keys=True)
