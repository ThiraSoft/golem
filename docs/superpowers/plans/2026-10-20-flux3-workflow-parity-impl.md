# FLUX 3 Klein workflow parity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** FLUX 3 Klein in golem and krea2-mobile draws at the BFL workflow settings: 4 MP output, match_input_image, 1 MP per reference, up to 4 references.

**Architecture:** The DiT arena is sized from the pipeline's limits and its MLP runs in token chunks so 34.7k tokens fit one storage buffer; the VAE decodes past 1 MP as an exact low-resolution trunk plus full-width bands sharing merged GroupNorm statistics (two passes); the pipeline gains MatchImage and per-reference 1 MP; golem-server and krea2-mobile carry `match_image` and the 4 MP sizes.

**Tech Stack:** Go 1.27, Vulkan compute (golem `vk` package, GLSL via glslc), ComfyUI Python for reference fixtures, vanilla JS page.

**Spec:** `docs/superpowers/specs/2026-10-20-flux3-workflow-parity-design.md`

## Global Constraints

- Repo for Tasks 1 to 5: `/home/ricardo/dev/golem-parity`, branch `flux3-workflow-parity`. Task 6: `/home/ricardo/dev/krea2-mobile`, branch `master`.
- FLUX 3 only: no change under `krea2/`, no behaviour change on any Krea 2 path.
- `DefaultMaxPixels = 4456448`; `MaxSide = 8192`; `RefTokens = 4096`; `MaxRefs = 4`; `MaxTextTokens = 960` (unchanged); `mlpChunk = 4096`; `vaeBandLatentPix = 4096`; `vaeBandOverlap = 2`.
- 4 MP sizes: 2048×2048, 1680×2512, 1776×2368, 1536×2736, 2512×1680, 2368×1776, 2736×1536, 3136×1344.
- Every arena is checked against `1<<32` bytes before allocation, with an error naming the limit.
- 1 MP behaviour unchanged: every existing flux3 test passes with its current thresholds.
- Never load a model in Python (fixtures are recorded by the controller). One GPU job at a time.
- Comments and identifiers in golem in English, in the style of the surrounding code; gofmt every Go file touched.
- No U+2014 (em dash) in any file written or commit message.
- Commit messages: golem `flux3: ...` / `vk: ...` / `cmd/...: ...` subject in English; krea2-mobile in French. End with:
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>` and `Claude-Session: https://claude.ai/code/session_0156r6qo6WjzDGTReMMYzzxJ`
- Forbidden: git push, reset, rebase, clean, stash, checkout of another branch; weakening a test; spawning subagents.
- GPU test commands use `-timeout 60m -count=1`.

## Review Focus

1. Portrait JPEG with EXIF orientation 6 under match_image: output size and reference 1 come from the oriented image (Task 4 test `TestMatchImageUsesOrientedSize`).
2. A panorama or tall strip whose match size or typed size passes `MaxSide` or `MaxPixels` is refused with a message naming the limit, before the DiT is opened (Task 4 `TestCheckRefusesPastSideAndPixels`, Task 5 server 400 test).
3. 4 refs at 4 MP with the DiT kept: peak VRAM measured, fits or the default precision moves to fp8 (Task 2 `TestDiT4MPMatchesComfyUI` logs it; ruling recorded).
4. Band edges: the first and last bands clip their overlap at the image edge, and a latent whose height is not a multiple of the core still covers every row (Task 3 `TestVAEBandedMatchesFull` with h=64 and a forced band budget that leaves a short last band).
5. Switching model back and forth in the page restores the right presets and sizes, and the match switch survives a reload (Task 6 manual check list plus `TestPicturesFluxDefaults`).

---

### Task 1: Fixtures at 4 MP (scripts only)

**Files:**
- Modify: `ref/flux3/dump.py`
- Modify: `ref/flux3/bench_comfy.py`
- Modify: `ref/flux3/README.md`

**Interfaces:**
- Produces (under `testdata/flux3/`, written later by the controller): `t2i4/`, `i2i4/`, `mi2i4/` with `request.json`, `noise0`, `sigmas`, `in.x`, `in.t`, `in.ctx`, `in.next`, `out`, `latent`, `rgb`, `<stage>.png`, `timings.json`; i2i4 and mi2i4 also `ref0.jpg` (and mi2i4 `ref1.jpg`, `ref2.png`, `ref3.png`), `ref_rgb0`, `ref_latents`, `ref_ids`. `request.json` holds `prompt`, `seed`, `steps`, `width`, `height`, `ref_shapes` (list of `[h, w]` latent shapes). Every float file has its shape in `meta.json` like the existing stages. `rgb` is HWC float in [0,1] like `accept/rgb`.

- [ ] **Step 1:** Add stages `t2i4`, `i2i4`, `mi2i4` to `STAGES` handling (not to the default list, so a plain run does not redo them) following `stage_i2i`/`stage_mi2i`/`stage_accept`:
  - `t2i4`: `PROMPT_ACCEPT`, 2048×2048, seed 42, `STEPS`, `CFG`.
  - `i2i4`: open `os.path.join(COMFY, "input", "IMG_20191030_202255.jpg")` with PIL, save it as `OUT/i2i4/ref0.jpg` with quality 95 and an EXIF whose tag 0x0112 is 6 (`exif = im.getexif(); exif[0x0112] = 6; im.save(..., exif=exif)`); reload it with `ImageOps.exif_transpose`, check its size is 3120×4160 (assert), make it a tensor, run `ImageScaleToTotalPixels.execute(image, "lanczos", 4.0, 16)` (import from `comfy_extras.nodes_post_processing`), take width/height from the result, `ReferenceMultiLatent.execute` with `{"resolution": "resize", "megapixels": 1.0}` and `images={"image1": scaled}`, prompt `"turn the photo into a watercolor painting, keep the composition"`, seed 7.
  - `mi2i4`: same `ref0.jpg` (copy from i2i4's construction, written as `OUT/mi2i4/ref0.jpg`), plus `REF_TOAST` copied to `ref1.jpg`, `COMFY/input/eyes_renarde.png` to `ref2.png`, `COMFY/input/Capture d'écran_20260803_084025.png` to `ref3.png`; image1 is the 4 MP scaled ref0, images 2 to 4 loaded with `exif_transpose` and `.convert("RGB")`; prompt `"the scene from the first image, with the toast from the second image on a table, painted in the colours of the third and fourth images"`, seed 7.
- [ ] **Step 2:** For these stages record the first DiT call with `record_dit=True` but without the block waypoint hooks (add a parameter `waypoints=False` to `run_sampling` that skips the `grab("wp.*")` hooks; keep `in.*` and `out`), saved under the stage prefix (`t2i4/in.x`, `t2i4/out`, ...). Save `ref_rgb0` (first image as handed to `vae.encode`), `ref_latents` (concatenated tokens like `mi2i/ref_latents`), `ref_ids`.
- [ ] **Step 3:** `timings.json`: wall seconds per sampler step (differences of `time.perf_counter()` taken in the step callback, first step from sampling start), decode seconds, total seconds from text encode to decoded image, and `decode_mode`: wrap `vae.decode_tiled_` with a function that sets a flag and calls the original, `"tiled"` if it ran, else `"full"`. Write `request.json` with the fields above.
- [ ] **Step 4:** `bench_comfy.py`: add a 2048×2048 16-step row with the same method as its existing 1024 rows.
- [ ] **Step 5:** README in `ref/flux3/`: document the three stages and the exact command
  `systemd-run --user --scope -q -p MemoryMax=24G -p MemorySwapMax=16G /mnt/data/dev/ComfyUI/.venv/bin/python ref/flux3/dump.py /mnt/data/dev/ComfyUI testdata/flux3 t2i4 i2i4 mi2i4`.
- [ ] **Step 6:** Verify syntax only: `python3 -m py_compile ref/flux3/dump.py ref/flux3/bench_comfy.py` prints nothing. Do NOT run either script.
- [ ] **Step 7:** Commit `ref/flux3: record 4 MP t2i, i2i and mi2i fixtures`.

**Controller step after review (not the implementer):** run the dump command above alone, then check `meta.json` has the three stages and `timings.json` exists in each.

---

### Task 2: DiT arena sized from limits, MLP in chunks

**Files:**
- Modify: `flux3/dit.go`
- Modify: `flux3/dit_test.go`

**Interfaces:**
- Consumes: fixtures `t2i4/in.*`, `t2i4/out`, `mi2i4/in.*`, `mi2i4/out`, `mi2i4/ref_latents`, `mi2i4/request.json` (`ref_shapes`).
- Produces:
  - `const RefTokens = 4096`, `const MaxRefs = 4`, `const mlpChunk = 4096`. `MaxImageTokens` and `MaxRefTokens` stay as the 1 MP defaults (4096) used by `OpenDiT`.
  - `func OpenDiTSized(d *vk.Device, path string, prec Precision, maxImageTokens, maxRefTokens int) (*DiT, error)`; `OpenDiT(d, path, prec...)` calls it with `MaxImageTokens, MaxRefTokens`.
  - `func ditArenaFloats(maxImageTokens, maxRefTokens int) int`: the arena size in floats for those limits (layout and size from the same code, so they cannot drift).
  - `(*DiT).MaxImageTokens() int`, `(*DiT).MaxRefTokens() int`.
  - `func vramUsed() (int64, bool)` in `dit_test.go` or a test helper file: sum of `/sys/class/drm/card*/device/mem_info_vram_used`, false when absent.

- [ ] **Step 1: Failing tests** in `flux3/dit_test.go`:
  - `TestDiTArenaFloats`: `ditArenaFloats(4096, 4096)*4` is below the current 1 MP arena (log both; the current one is `m.ArenaBytes()` today, about 2.0 GB, so assert `< 2_000_000_000`); `ditArenaFloats(17408, 16384)*4 < 1<<32`; `ditArenaFloats(40000, 16384)*4 >= 1<<32`.
  - `TestOpenDiTSizedRefusesPast4GiB`: `OpenDiTSized(d, path, PrecisionFP16, 40000, 16384)` returns an error whose text contains `"40000"` and `"4 GiB"`, and returns before uploading weights (assert it takes under 5 s).
  - `TestDiT4MPMatchesComfyUI`: for `t2i4` (no refs) and `mi2i4` (refs from `ref_latents` split by `request.json` `ref_shapes`, each reshaped back to `[128][h][w]` planes as `SetRefs` expects), `OpenDiTSized(d, path, PrecisionFP16, 17408, 16384)`, `SetText(0, in.ctx, tokens)`, `SetRefs`, `Step(0, in.x, H/16, W/16, in.t, in.next)`; `compareRMS(t, name, got, out, 0.03)`. Log the step time of a second identical step and `vramUsed()` read right after it (peak with weights and arena resident) in MB, plus `WeightBytes()` and `ArenaBytes()`.
- [ ] **Step 2:** Run `go test ./flux3 -run 'TestDiTArenaFloats|TestOpenDiTSizedRefusesPast4GiB|TestDiT4MPMatchesComfyUI' -timeout 60m -count=1 -v`; expect compile failure (RED).
- [ ] **Step 3: Implement.** Layout per the spec section 2: halves regions `nh` (N×3072 halves = N×1536 floats) and `half` (mlpChunk×9216 halves) first; then `mlp` (mlpChunk×2×9216 floats); then `x, q, kBuf, vBuf, att, out` (N×3072 each), `table` (N×128), and the per-slot text/ref regions and `img`, `imgOut` sized from the limits; `embImgBuf` removed (`emb_in.image` stores into `m.x` directly, the K2Copy into `x` goes). `recordBlock` order: norm, q/k/v products, qk-norm, rope, attention, attnOut store into `out`, then for each chunk `c0` in steps of `mlpChunk` up to `tokens`: `mlpIn` (`Cols = min(mlpChunk, tokens-c0)`, `X = 2*m.nh + c0*3072`) into `mlp`, SwiGLU into `half`, `mlpOut` with `Mode: vk.K2Add` into `out + c0*3072`; then gates. Bounds checks in `SetRefs` and `stepWithWaypoints` use the DiT's own limits and name them. The 4 GiB check happens before any weight is read, message: `flux3: a DiT arena for %d image and %d reference tokens is %d bytes, past the 4 GiB one storage buffer holds`.
- [ ] **Step 4:** Run the three new tests, then the whole package `go test ./flux3 -timeout 60m -count=1` (all existing tests must pass at their current thresholds; report TestDiTMatchesComfyUI and TestAcceptMatchesComfyUI figures before and after). Report the 4 MP step times and the VRAM figures verbatim: the controller makes the fp16/fp8 ruling from them.
- [ ] **Step 5:** `go vet ./flux3`, commit `flux3: size the DiT arena from its limits and run the MLP in chunks`.

---

### Task 3: Banded VAE decode

**Files:**
- Modify: `vk/krea2.go` (CopyStrided)
- Modify: `vk/krea2_test.go`
- Modify: `flux3/vae.go`
- Modify: `flux3/vae_test.go`

**Interfaces:**
- Consumes: fixtures `accept/latent`, `accept/rgb`, `t2i4/latent`, `t2i4/rgb`, `t2i4/timings.json`.
- Produces:
  - `func (k *K2) CopyStrided(r *Recorder, dst, dstStride, src, srcStride, n, count uint32)`: `count` regions of `n` floats, region i from `src+i*srcStride` to `dst+i*dstStride` (offsets in floats), one `vkCmdCopyBuffer` on the arena, then `r.Barrier()`.
  - `const vaeBandLatentPix = 4096`, `const vaeBandOverlap = 2`.
  - `OpenVAE(d, path, maxLatentPix)` unchanged signature; it now accepts `maxLatentPix` up to `DefaultMaxPixels/256` (17408) and lays out trunk + band buffers; `VAE.bandPix int` field (set to `vaeBandLatentPix` by `openVAE`; tests may lower it) decides `Decode`'s path: `h*w <= v.bandPix` full path as today, else banded.
  - `func mergeGroupStats(parts [][3]float64) [3]float64` (n, mean, m2 via Chan's formula), unit-tested.

- [ ] **Step 1: Failing tests.**
  - `vk`: `TestK2CopyStrided`: arena of 64 floats with 0..63 written, `CopyStrided(dst=40, dstStride=5, src=1, srcStride=10, n=3, count=3)` then read back: floats 40..42 = 1,2,3; 45..47 = 11,12,13; 50..52 = 21,22,23; the rest unchanged.
  - `flux3`: `TestMergeGroupStats`: two parts of a known array (e.g. 1..10 split 3/7) merged give n=10, mean=5.5, m2=82.5 within 1e-9; a part with n=0 is ignored.
  - `TestVAEBandedMatchesFull`: `accept/latent` (64×64) decoded by a VAE opened for 64*64 with the full path, and by the same VAE with `bandPix = 64*20` (band height 20, core 16, so bands with clipped overlap at both edges); PSNR banded vs full ≥ 40 dB, logged. Then `bandPix = 64*23` (core 19, last band short) ≥ 40 dB too.
  - `TestVAEDecode4MPMatchesComfyUI`: `OpenVAE(d, vaePath(), 17408)`, decode `t2i4/latent` (128×128) twice, PSNR vs `t2i4/rgb` ≥ 35 dB; log first and steady-state times beside `decode` seconds and `decode_mode` from `t2i4/timings.json`; log `ArenaBytes`-style total of the VAE arena.
  - `TestOpenVAERefusesPast4GiB`: `OpenVAE(d, vaePath(), 40000)` errors naming 4 GiB.
- [ ] **Step 2:** Run `go test ./vk -run TestK2CopyStrided -count=1` and `go test ./flux3 -run 'TestMergeGroupStats|TestVAEBanded|TestVAEDecode4MP|TestOpenVAERefusesPast4GiB' -timeout 60m -count=1 -v`: compile failure (RED).
- [ ] **Step 3: Implement `CopyStrided`** with `vkCmdCopyBuffer(r.cb, arena, arena, count, &regions[0])` (bytes = floats×4), reusing the `bufferCopy` type; panic on `count == 0` is not needed, return early.
- [ ] **Step 4: Implement banded decode in `flux3/vae.go`.** Arena: `b0 = b1 = max(65536*bandPix, 8192*maxLatentPix)`, `b2 = max(32768*bandPix, 8192*maxLatentPix)`, `trunk = 8192*maxLatentPix` only when `maxLatentPix > bandPix`, `statsBuf` as today, `statsAcc = 13 * maxBands * 32*64*3` floats with `maxBands = 64` (error if a decode needs more), `statsMerged = 13*32*3`; total checked `< 1<<32` bytes before `NewK2`, message `flux3: a VAE arena for %d latent pixels is %d bytes, past the 4 GiB one storage buffer holds`. The band-size limit comes from the `bandPix` used at open. Programs:
  - trunk program per `(h, w)`: the current decode program's steps up to and including level 2's three resnets (before its upsample), output copied into `trunk` (512 × 4h × 4w).
  - band programs per `(h, w, band index, pass)`: band latent rows `[r0, r1)` with overlap clipped to `[0, h)`; `CopyStrided` 512 planes of `(r1-r0)*4*4w` floats from `trunk + c*16hw + r0*4*4w` (plane stride `16hw`) into `b[0]` (plane stride `(r1-r0)*16w`); then level 2 upsample, level 1 (3 resnets + upsample), level 0 (3 resnets), norm_out with SiLU, conv_out. In pass A every norm runs `GStats` then `GNorm` as today and its stats (32×slices×3 floats at `statsBuf`) are copied into `statsAcc[norm][band]`; in pass B every norm runs only `GNorm` with `Slices: 1, Stats: statsMerged + norm*96`.
  - Between passes the host reads `statsAcc`, merges every norm's groups across bands and slices with `mergeGroupStats` (float64), writes `statsMerged` as float32 `(n, mean, m2)` triples.
  - After pass B for each band, read its 3 output planes' core rows (`[y0*16, y1*16)` of each plane) and place them into the full image planes; `toRGB` as today. Band height `bh = bandPix / w` must be > `2*vaeBandOverlap` or decode errors naming the width limit.
- [ ] **Step 5:** Run the new tests then `go test ./flux3 ./vk -timeout 60m -count=1`. Report PSNRs and times verbatim.
- [ ] **Step 6:** `go vet ./flux3 ./vk`, commit `flux3: decode past 1 MP in bands sharing GroupNorm statistics` (one commit may hold the vk helper, or two commits `vk: add K2.CopyStrided` first).

---

### Task 4: Pipeline at 4 MP, match_image, 1 MP per reference

**Files:**
- Modify: `flux3/pipeline.go`
- Modify: `flux3/paths.go`
- Modify: `flux3/pipeline_test.go`
- Modify: `flux3/bench_test.go` (only if a call to `Open`/`OpenDiT` there needs the new limits)

**Interfaces:**
- Consumes: `OpenDiTSized`, `RefTokens`, `MaxRefs` (Task 2); banded `OpenVAE` up to 17408 (Task 3); fixtures `t2i4`, `i2i4`, `mi2i4` (`ref0.jpg` ... `ref3.png`, `request.json`, `rgb`, `ref_rgb0`, `timings.json`).
- Produces:
  - `const DefaultMaxPixels = 4456448`, `const MaxSide = 8192`.
  - `Options.MaxRefs int` (0 means `MaxRefs`); `Options.MaxPixels` 0 means `DefaultMaxPixels`; `Open` refuses `MaxPixels > DefaultMaxPixels` naming the limit, and `MaxRefs` outside `[1, MaxRefs]`.
  - `Request.MatchImage bool` with tag `json:"match_image,omitempty"`.
  - `func ScaleToTotalPixels(img image.Image, mp float64, steps int) *image.RGBA`: ComfyUI's `ImageScaleToTotalPixels` (scale `sqrt(mp*1048576/(w*h))`, each side `math.RoundToEven(side*scale/steps)*steps`, lanczos via `lanczosKernel` after `ensureOpaque`).
  - `func (p *Pipeline) Check(r Request) error`: resolves MatchImage size (without encoding anything) and runs the size/steps/CFG checks; `Generate` calls it.
  - The precision default from the controller's ruling after Task 2 (carried in the dispatch).

- [ ] **Step 1: Failing tests** in `flux3/pipeline_test.go`:
  - `TestScaleToTotalPixelsSizes`: for a blank image of each aspect in the spec list scaled from 1000×1000-ish inputs (1:1 1000×1000, 2:3 1000×1500, 3:4 900×1200, 9:16 900×1600, 3:2 1500×1000, 4:3 1200×900, 16:9 1600×900, 21:9 2100×900), `ScaleToTotalPixels(img, 4.0, 16)` bounds equal the spec sizes 2048×2048, 1680×2512, 1776×2368, 1536×2736, 2512×1680, 2368×1776, 2736×1536, 3136×1344.
  - `TestMatchImageUsesOrientedSize`: `DecodeImage` of `testdata/flux3/i2i4/ref0.jpg` is 3120×4160; `Check` with `MatchImage: true` and that image sets nothing visible but a helper `p.resolve(&r)` (unexported, called by Check and Generate) gives Width, Height equal to `i2i4/request.json` width/height (expect 1776×2368).
  - `TestCheckRefusesPastSideAndPixels` (no GPU; a `Pipeline{o: Options{MaxPixels: DefaultMaxPixels}}` like the existing check tests): 8208×512 refused naming 8192; 2560×2560 refused naming 4456448; 1680×2512 accepted; `MatchImage` without images refused; 5 images refused naming 4; a 300×300 image with MatchImage resolves to 2048×2048 (accepted).
  - `TestPicture4MPMatchesComfyUI`, `TestI2I4MPMatchesComfyUI`, `TestMultiRef4MPMatchesComfyUI`: `Open(ComfyUI(...))` with defaults and `Keep: true`; requests from each `request.json` (prompt, seed, steps, CFG 1); i2i4/mi2i4 with `MatchImage: true` and images decoded with `DecodeImage` from the stage's `ref*` files in order; PSNR vs `rgb` ≥ 30 dB each; log golem timings beside `timings.json` (per-step mean, decode, total). For i2i4 also PSNR of golem's first reference RGB (from `ResizeReference(ScaleToTotalPixels(oriented, 4, 16), 1.0, RefTokens)`) vs `ref_rgb0`, logged, ≥ 35 dB.
  - Update `TestOpenRefusesExcessMaxPixels` to the new limit (`MaxPixels: 4096*4096` refused).
- [ ] **Step 2:** Run them: RED.
- [ ] **Step 3: Implement.** `Generate`: `p.resolve(&r)` first (MatchImage: size from `ScaleToTotalPixels(r.Images[0], 4.0, 16)` and `Images[0]` replaced by it, in a copy of the slice so the caller's slice is untouched), then `check`; references each `ResizeReference(img, mp, RefTokens)`; `OpenDiTSized(p.d, p.o.DiT, p.o.Precision, p.o.MaxPixels/256, p.o.MaxRefs*RefTokens)`; `OpenVAE(p.d, p.o.VAE, p.o.MaxPixels/256)`. Error texts name the limit (`MaxSide`, `MaxPixels`, `MaxRefs`) like the current ones, without "1 MP DiT limit".
- [ ] **Step 4:** Run the new tests, then `go test ./flux3 -timeout 60m -count=1`. Report PSNRs and timings verbatim.
- [ ] **Step 5:** `go vet ./flux3`, commit `flux3: draw at 4 MP with match_image and 1 MP references`.

---

### Task 5: cmd/flux3, golem-server, README

**Files:**
- Modify: `cmd/flux3/main.go`
- Modify: `cmd/golem-server/generations.go`, `cmd/golem-server/generations_test.go`
- Modify: `flux3/README.md`, the FLUX 3 lines of the top-level `README.md` if any mention 1 MP or 1024

**Interfaces:**
- Consumes: `Request.MatchImage`, `Pipeline.Check`, `DefaultMaxPixels`, `MaxSide` (Task 4).
- Produces: JSON field `match_image` (bool) on `/v1/images/generations`; `PictureRequest.MatchImage bool`; FLUX 3 default size `"2048x2048"`.

- [ ] **Step 1: Failing tests** in `cmd/golem-server/generations_test.go`, with the existing fake-imager pattern of the flux3 tests there (lines ~243 and ~295):
  - `TestFlux3MatchImageCarried`: a request with `match_image: true` and one image reaches the imager with `MatchImage == true`.
  - `TestFlux3DefaultSize`: the flux3 imager's `Defaults()` size is `"2048x2048"`.
  - `TestFlux3SizePastLimitIs400`: with an imager whose optional `Check(PictureRequest) error` refuses (fake returning an error naming 4456448), the handler answers 400 with that message and the imager's Generate is never called. The interface is `type checker interface{ Check(PictureRequest) error }`, implemented by `flux3Imager` through `Pipeline.Check(toFlux3(r))`; `lockedImager` forwards it when the inner imager has it.
- [ ] **Step 2:** `go test ./cmd/golem-server -run 'TestFlux3' -count=1`: RED.
- [ ] **Step 3: Implement.** `generationRequest.MatchImage bool json:"match_image"` → `PictureRequest.MatchImage` → `toFlux3`. Krea 2: a request with `match_image` true and Krea 2 loaded is refused with 400 like images are. cmd/flux3: `-match-image` bool (needs at least one `-image`), defaults `-width 2048 -height 2048`, `MaxPixels` left 0 (default); README usage comment updated.
- [ ] **Step 4:** README `flux3/README.md`: reference section rewritten (1 MP each, up to 4, match_image), limits (4.25 MP, side 8192), and 4 MP rows in the performance and VRAM tables from the figures measured in Tasks 2 to 4 and `timings.json` (the controller passes them in the dispatch); fp8 table updated if the ruling made fp8 the 4 MP default. Run `cmd/flux3` once: `go run ./cmd/flux3 -prompt "a red fox in the snow, photo" -seed 42 -out /tmp/claude-1000/flux3-4mp.png` and report its timings.
- [ ] **Step 5:** `go vet ./flux3 ./cmd/flux3 ./cmd/golem-server && go test ./cmd/golem-server -count=1`, commit `cmd: FLUX 3 at 2048x2048 by default, match_image in the server and the CLI`.

---

### Task 6: krea2-mobile, FLUX 3 sizes and « taille de l'image 1 »

**Files:**
- Modify: `/home/ricardo/dev/krea2-mobile/web/index.html`
- Modify: `/home/ricardo/dev/krea2-mobile/server.go`, `jobs.go` (the `picture` struct)
- Modify: `/home/ricardo/dev/krea2-mobile/server_test.go`
- Modify: `/home/ricardo/dev/krea2-mobile/README.md` (the « Taille » line)

**Interfaces:**
- Consumes: golem-server's `match_image` and FLUX 3 default 2048x2048 (Task 5).
- Produces: `frontParams.MatchImage bool json:"match_image"`; `picture.MatchImage bool json:"match_image,omitempty"`.

- [ ] **Step 1: Failing Go tests** in `server_test.go`, in its existing style:
  - `TestPicturesFluxDefaults`: FLUX 3 model, no width/height → `Size == "2048x2048"`, steps 16; Krea 2 model → `"1024x1024"`, steps 8 (unchanged).
  - `TestPicturesFluxClamp`: FLUX 3 width 9000 → clamped to 8192; Krea 2 width 9000 → 4096 as today.
  - `TestComposeCarriesMatchImage`: a compose request with `match_image: true` and one image produces shots with `MatchImage == true` and the JSON sent to golem contains `"match_image":true`; Generate mode never sets it.
- [ ] **Step 2:** `cd /home/ricardo/dev/krea2-mobile && go test ./... -count=1`: RED.
- [ ] **Step 3: Implement server.go/jobs.go.**
- [ ] **Step 4: Page.** Two preset sets in JS: `KREA_PRESETS` = the six current buttons (labels and sizes as now), `FLUX_PRESETS` = Square 2048², Portrait 2:3 1680×2512, Portrait 3:4 1776×2368, Port. 9:16 1536×2736, Land. 3:2 2512×1680, Land. 4:3 2368×1776, Land. 16:9 2736×1536, Pano 21:9 3136×1344, Rapide 1024². `renderPresets()` rebuilds `#presetRow` from the set of the picked model and rebinds clicks to `syncPresets`. On switch to FLUX: if the size is a Krea 2 preset or 768×1024, set 2048×2048; on switch to Krea 2: if the size is not one of its presets, set 768×1024. Call `renderPresets()` at load after the models list fills. Compose/Edit (FLUX only): a card with the label « taille de l'image 1 » and a `.switch` like the LoRA one (id `matchToggle`), on by default, its state in localStorage key `mobileMatchImage`; when on, hide the Width/Height fields and `#presetRow` in compose/edit, and the request body for `/mobile/api/compose` and `/mobile/api/edit` carries `match_image: true`; when off, the typed size is sent. Generate mode always shows sizes and never sends it. Width/Height inputs step 16.
- [ ] **Step 5:** README « Taille » line: FLUX 3 up to 4.25 MP (2048×2048 by default, côté max 8192), Krea 2 up to 1024 × 1024; Compose/Edit « taille de l'image 1 ».
- [ ] **Step 6:** `go vet ./... && go test ./... -count=1`; `node -e` syntax check of the page script: extract the `<script>` body to a temp file under `/tmp/claude-1000/` and run `node --check` on it. Commit on master `FLUX 3 : tailles 4 MP, taille de l'image 1 en Compose et Edit`.
- [ ] **Manual check list (report each as checked in code, not by running a browser):** switch Krea 2 → FLUX → Krea 2 restores the Krea 2 six presets and a Krea 2 size; reload with FLUX saved shows FLUX presets; switch off survives reload; Edit with switch on hides size fields.
