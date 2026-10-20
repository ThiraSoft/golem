# FLUX 3 Klein, workflow parity (4 MP, reference size, 1 MP per reference)

> Rough plan, to be turned into a full spec + plan in a new conversation
> (superpowers:brainstorming then writing-plans, SDD with Gemini implementers
> and opus reviewers). FLUX 3 mode only: Krea 2 does not change.

**Goal:** golem and krea2-mobile draw FLUX 3 Klein pictures with the BFL
workflow settings (`ref/flux3/workflows/flux3_klein_bf16_{t2i,i2i,mi2i}.json`),
not the 1 MP shortcuts taken in the first port.

## Where golem differs from the workflows today

| Setting | Workflow | golem / krea2-mobile now |
|---|---|---|
| Output size, t2i | `ResolutionSelector` at **4.0 MP**, multiple 16 (1:1 = 2048²) | 1 MP max (`MaxImageTokens` 4096), default 1024² |
| Output size, i2i/mi2i | `match_input_image` true: first image through `ImageScaleToTotalPixels` (lanczos, 4.0 MP, steps 16), output = its size | size typed in the page |
| References | `ReferenceMultiLatent` resize, **1.0 MP each** | `MaxRefTokens` 4096 split by count (2 refs = 0.5 MP each) |
| First reference | the 4 MP rescaled image, then down to 1 MP | the uploaded image straight to 1 MP |

Shift (ln 16), CFG 1, euler/simple, 16 steps already match.

4 MP sizes from ComfyUI's formula (`round(r*sqrt(4 MiB/(a*b))/16)*16`):
1:1 2048×2048, 2:3 1680×2512, 3:4 1776×2368, 9:16 1536×2736, 3:2 2512×1680,
4:3 2368×1776, 16:9 2736×1536, 21:9 3136×1344. The largest in tokens is
2:3 (105×157 = 16485), so the image cap must be about 4.25 MP, not 4.0.
Worst case token count: 16485 image + 4 refs × 4096 + 960 text ≈ 33.8k.

## Hard parts (golem)

- **VRAM.** fp16 weights are 11.1 GB of 16 GB. The DiT arena is fp32 and
  sized from constants (`flux3/dit.go` ~L420): about 49k floats per token,
  i.e. ~6.6 GB at 33.8k tokens. Also past the 4 GiB single buffer limit
  (`vk.NewK2` one arena; the VAE already checks `1<<32/4`).
  Plan: size the arena from `Options` (max pixels, max refs) and run the
  MLP (`mlp`, `half`: 27.6k floats/token) in token chunks, which brings it
  to ~21.6k floats/token, ~2.9 GB. Keep fp16 (fidelity ruling); fp8 at
  4 MP only if the user agrees.
- **VAE.** Arena is ~655 KB per latent pixel: ~10.7 GB at 2048². Needs a
  tiled decode (ComfyUI `VAEDecodeTiled` style, feathered overlap) or a
  leaner arena. First check what ComfyUI itself does at 4 MP on this card
  (full or tiled fallback), that is the reference.
- **Attention at ~34k tokens.** Check the attn kernel has no N² scratch or
  dispatch limit; measure s/step (expect ~4× the 1 MP step, ~1 min/image).

## Tasks

0. **Fixtures at 4 MP.** Extend `ref/flux3/dump.py`: t2i 2048² seed fixed,
   i2i with a portrait phone photo and match_input_image, mi2i with 4 refs.
   Record ComfyUI timings and its decode mode. Run alone under
   `systemd-run --user --scope -p MemoryMax=24G -p MemorySwapMax=16G`.
   Restore the `testdata/flux3 -> /mnt/data/golem-testdata/flux3` symlink in
   `~/dev/golem` (lost with the worktree; fixtures are still on /mnt/data).
1. **DiT arena from options + chunked MLP.** `MaxImageTokens`/`MaxRefTokens`
   become per-pipeline values; check every buffer < device limit. 1 MP
   tests unchanged; block outputs at 4 MP match the fixture; VRAM measured.
2. **Tiled (or leaner) VAE decode** up to 4.25 MP; PSNR vs ComfyUI decode
   ≥ 35 dB; time measured.
3. **Pipeline.** Refs at 1 MP each with no split (up to 4); new request
   field `MatchImage bool`: output size = `ImageScaleToTotalPixels(images[0],
   4.0, 16)` on the EXIF-oriented image, and that rescaled image becomes
   reference 1 (then 1 MP as now). Default `MaxPixels` 4.25 MP. End to end
   t2i/i2i/mi2i at 4 MP vs fixtures (target ≥ 30 dB), bench vs ComfyUI.
4. **cmd/flux3 + golem-server.** `match_image` in the JSON, `-match-image`
   flag, FLUX 3 default size `2048x2048`; size past the cap gives a 400 with
   the limit. README perf tables at 4 MP.
5. **krea2-mobile, FLUX 3 only.** Preset row swaps to the 8 sizes above
   (plus a fast 1024² kept), default 2048² when FLUX is picked; Compose/Edit
   get a « taille de l'image 1 » switch, on by default, which hides
   width/height and sends `match_image`. Back to Krea 2 restores its presets.
   Server tests, then restart `krea2-mobile.service`.

## Watch out for

- Portrait photo with EXIF rotation under match_image: size from the
  oriented image.
- A tiny reference (e.g. 300×300) blown up to 4 MP: allowed, like ComfyUI.
- A panorama: match size must stay under the token cap; refuse cleanly.
- 4 refs + 4 MP: peak VRAM must fit with the DiT kept (`-flux3-keep`).
- Switching model back and forth in the page keeps the right presets.
