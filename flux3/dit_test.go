package flux3

import (
	"math"
	"os"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/ThiraSoft/golem/krea2"
)

func readPeakRSS() int64 {
	b, err := os.ReadFile("/proc/self/status")
	if err != nil {
		return 0
	}
	for _, line := range strings.Split(string(b), "\n") {
		if strings.HasPrefix(line, "VmHWM:") {
			fields := strings.Fields(line)
			if len(fields) >= 2 {
				kb, _ := strconv.ParseInt(fields[1], 10, 64)
				return kb * 1024
			}
		}
	}
	return 0
}

const defaultDiTPath = "/mnt/data/dev/ComfyUI/models/diffusion_models/flux3-image-klein-bf16.safetensors"

func ditPath() string {
	if p := os.Getenv("FLUX3_DIT"); p != "" {
		return p
	}
	return defaultDiTPath
}

func TestRopeMatchesComfyUI(t *testing.T) {
	fix := loadFixtures(t)
	wantPE := fix.read(t, "dit/pe")

	// 1. Image 16x16 + 80 text tokens against dit/pe (shape [1, 336, 1, 64, 2, 2] = 86016 floats).
	imgIDs := ImageIDs(16, 16, 0)
	txtIDs := TextIDs(80)
	ids336 := append(imgIDs, txtIDs...)

	gotPE := EmbedPE(ids336)
	if len(gotPE) != len(wantPE) {
		t.Fatalf("EmbedPE len = %d, want %d", len(gotPE), len(wantPE))
	}
	var maxDiff float32
	for i := range gotPE {
		d := float32(math.Abs(float64(gotPE[i] - wantPE[i])))
		if d > maxDiff {
			maxDiff = d
		}
	}
	if maxDiff > 1e-6 {
		t.Errorf("EmbedPE diff against dit/pe = %g, want <= 1e-6", maxDiff)
	}

	// 2. Image 16x16 + one ref (13x19) + 80 text tokens (583 tokens).
	refIDs := RefIDs([][2]int{{13, 19}})
	ids583 := append(append(ImageIDs(16, 16, 0), refIDs...), TextIDs(80)...)
	got583 := EmbedPE(ids583)
	if len(got583) != 583*256 {
		t.Fatalf("EmbedPE 583 len = %d, want %d", len(got583), 583*256)
	}

	// Verify RopeTable consistency with EmbedPE.
	table := RopeTable(ids583)
	if len(table) != 583*128 {
		t.Fatalf("RopeTable len = %d, want %d", len(table), 583*128)
	}
	for tok := 0; tok < 583; tok++ {
		for pair := 0; pair < 64; pair++ {
			cosVal := got583[tok*256+pair*4+0]
			sinVal := got583[tok*256+pair*4+2]
			if table[tok*128+pair] != cosVal {
				t.Fatalf("tok %d pair %d: table cos %g != EmbedPE cos %g", tok, pair, table[tok*128+pair], cosVal)
			}
			if table[tok*128+64+pair] != sinVal {
				t.Fatalf("tok %d pair %d: table sin %g != EmbedPE sin %g", tok, pair, table[tok*128+64+pair], sinVal)
			}
		}
	}
}

func TestDiTMatchesComfyUI(t *testing.T) {
	path := ditPath()
	needFile(t, path)
	f := loadFixtures(t)
	d := device(t)

	// Pre-encode i2i prompt using encoder so we have the real conditioning for i2i
	needFile(t, krea2.EncoderPath())
	tok := tokenizer(t)
	enc, err := OpenEncoder(d, krea2.EncoderPath(), false)
	if err != nil {
		t.Fatal(err)
	}
	ids, real := Tokenize(tok, "spread strawberry jam on the toast and add a few fresh blueberries on the plate")
	i2iCtx, err := enc.EncodeMasked(ids, real)
	enc.Close()
	if err != nil {
		t.Fatal(err)
	}

	inX := f.read(t, "dit/in.x")
	inT := f.read(t, "dit/in.t")[0]
	inNext := f.read(t, "dit/in.next")[0]
	inCtx := f.read(t, "dit/in.ctx")
	tokens := len(inCtx) / CondWidth

	compareWp := func(name string, got []float32, tol float64) float64 {
		t.Helper()
		want32 := f.read(t, "dit32/"+name)
		gap32 := rmsError(got, want32)
		if gap32 > tol {
			t.Errorf("%s vs dit32: rms error %.4f%% exceeds tol %.4f%%", name, gap32*100, tol*100)
		}
		if f.has("dit/" + name) {
			wantBF16 := f.read(t, "dit/"+name)
			gapBF16 := rmsError(got, wantBF16)
			t.Logf("%s vs dit (bf16): rms error %.4f%% (vs dit32: %.4f%%)", name, gapBF16*100, gap32*100)
		}
		return gap32
	}

	// 1. FP16 product path: bf16 -> fp16 weights, K2 MM fp16 path
	t.Log("=== Testing FP16 DiT ===")
	m16, err := OpenDiT(d, path, PrecisionFP16)
	if err != nil {
		t.Fatalf("OpenDiT FP16: %v", err)
	}
	t.Logf("FP16 DiT weights on GPU: %d MB", m16.WeightBytes()/(1024*1024))

	if err := m16.SetText(0, inCtx, tokens); err != nil {
		t.Fatalf("m16.SetText: %v", err)
	}
	out16, wps16, err := m16.stepWithWaypoints(0, inX, 16, 16, inT, inNext, true)
	if err != nil {
		t.Fatalf("m16.stepWithWaypoints: %v", err)
	}

	// FP16 must be within 2% (0.02) of dit32 at every waypoint and out
	const tolFP16 = 0.02
	compareWp("wp.emb_img", wps16["wp.emb_img"], tolFP16)
	compareWp("wp.txt_early", wps16["wp.txt_early"], tolFP16)
	compareWp("wp.img_early", wps16["wp.img_early"], tolFP16)
	compareWp("wp.single_0", wps16["wp.single_0"], tolFP16)
	compareWp("wp.single_13", wps16["wp.single_13"], tolFP16)
	compareWp("wp.single_27", wps16["wp.single_27"], tolFP16)
	compareWp("out", out16, tolFP16)

	// FP16 Reference stream test
	i2iTokens := len(i2iCtx) / CondWidth
	if err := m16.SetText(0, i2iCtx, i2iTokens); err != nil {
		t.Fatalf("m16.SetText (i2i): %v", err)
	}
	refLatent := f.read(t, "i2i/ref_latent")
	if err := m16.SetRefs(0, [][]float32{refLatent}, [][2]int{{13, 19}}); err != nil {
		t.Fatalf("m16.SetRefs: %v", err)
	}
	i2iX := f.read(t, "i2i/x_0")
	outI2I16, wpsI2I16, err := m16.stepWithWaypoints(0, i2iX, 16, 16, inT, inNext, true)
	if err != nil {
		t.Fatalf("m16.stepWithWaypoints (i2i): %v", err)
	}
	compareRMS(t, "FP16 i2i/ref_states", wpsI2I16["ref_states"], f.read(t, "i2i/ref_states"), 0.02)
	gapOut0_16 := rmsError(outI2I16, f.read(t, "i2i/dit.out0"))
	t.Logf("FP16 i2i/dit.out0 vs bf16: rms error %.4f%%", gapOut0_16*100)
	compareRMS(t, "FP16 i2i/dit.out0", outI2I16, f.read(t, "i2i/dit.out0"), 0.03)

	m16.Close()

	// 2. FP8 product path: bf16 -> fp8 e4m3 weights, K2 MM fp8 path
	t.Log("=== Testing FP8 DiT ===")
	m8, err := OpenDiT(d, path, PrecisionFP8)
	if err != nil {
		t.Fatalf("OpenDiT FP8: %v", err)
	}
	defer m8.Close()
	t.Logf("FP8 DiT weights on GPU: %d MB", m8.WeightBytes()/(1024*1024))

	if err := m8.SetText(0, inCtx, tokens); err != nil {
		t.Fatalf("m8.SetText: %v", err)
	}
	out8, wps8, err := m8.stepWithWaypoints(0, inX, 16, 16, inT, inNext, true)
	if err != nil {
		t.Fatalf("m8.stepWithWaypoints: %v", err)
	}

	// FP8 asserted at measured gap vs dit32 + 0.5 percentage point (0.005)
	compareWp("wp.emb_img", wps8["wp.emb_img"], 0.02657+0.005)
	compareWp("wp.txt_early", wps8["wp.txt_early"], 0.02903+0.005)
	compareWp("wp.img_early", wps8["wp.img_early"], 0.00835+0.005)
	compareWp("wp.single_0", wps8["wp.single_0"], 0.02898+0.005)
	compareWp("wp.single_13", wps8["wp.single_13"], 0.12965+0.005)
	compareWp("wp.single_27", wps8["wp.single_27"], 0.05274+0.005)
	compareWp("out", out8, 0.04392+0.005)

	// FP8 Reference stream test
	if err := m8.SetText(0, i2iCtx, i2iTokens); err != nil {
		t.Fatalf("m8.SetText (i2i): %v", err)
	}
	if err := m8.SetRefs(0, [][]float32{refLatent}, [][2]int{{13, 19}}); err != nil {
		t.Fatalf("m8.SetRefs: %v", err)
	}
	outI2I8, wpsI2I8, err := m8.stepWithWaypoints(0, i2iX, 16, 16, inT, inNext, true)
	if err != nil {
		t.Fatalf("m8.stepWithWaypoints (i2i): %v", err)
	}
	compareRMS(t, "FP8 i2i/ref_states", wpsI2I8["ref_states"], f.read(t, "i2i/ref_states"), 0.00797+0.005)
	gapOut0_8 := rmsError(outI2I8, f.read(t, "i2i/dit.out0"))
	t.Logf("FP8 i2i/dit.out0 vs bf16: rms error %.4f%%", gapOut0_8*100)
	compareRMS(t, "FP8 i2i/dit.out0", outI2I8, f.read(t, "i2i/dit.out0"), 0.06126+0.005)
}

func TestEarlyBlocksCached(t *testing.T) {
	path := ditPath()
	needFile(t, path)
	f := loadFixtures(t)
	d := device(t)

	m, err := OpenDiT(d, path)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()

	// 1. Path (a): Cached path (flowMap=false: SetText runs early blocks once, Step reads txtStates)
	m.flowMap = false
	m.inStepEarly = false
	m.zeroTarget = false

	inCtx := f.read(t, "dit/in.ctx")
	tokens := len(inCtx) / CondWidth
	inX := f.read(t, "dit/in.x")
	inT := f.read(t, "dit/in.t")[0]

	dispatchesBefore := m.earlyDispatches
	if err := m.SetText(0, inCtx, tokens); err != nil {
		t.Fatalf("SetText: %v", err)
	}
	dispatchesAfterSet := m.earlyDispatches
	if dispatchesAfterSet <= dispatchesBefore {
		t.Fatalf("SetText without flow_map did not record early dispatches: %d -> %d", dispatchesBefore, dispatchesAfterSet)
	}
	recordedEarlyTxt := dispatchesAfterSet - dispatchesBefore

	// Step 1: should run 0 early block dispatches (cached states are reused)
	dispatchesBeforeStep1 := m.earlyDispatches
	outCached, err := m.Step(0, inX, 16, 16, inT, inT)
	if err != nil {
		t.Fatalf("Step 1 (cached): %v", err)
	}
	if m.earlyDispatches != dispatchesBeforeStep1 {
		t.Errorf("Step 1 unexpectedly ran early block dispatches: %d != %d", m.earlyDispatches, dispatchesBeforeStep1)
	}

	// Step 2: cached step program, 0 early block dispatches
	dispatchesBeforeStep2 := m.earlyDispatches
	if _, err := m.Step(0, inX, 16, 16, inT, inT); err != nil {
		t.Fatalf("Step 2 (cached): %v", err)
	}
	if m.earlyDispatches != dispatchesBeforeStep2 {
		t.Errorf("Step 2 unexpectedly ran early block dispatches: %d != %d", m.earlyDispatches, dispatchesBeforeStep2)
	}

	// 2. Path (b): in-Step path p0 forced to run with zero vec_target
	m.Trim()
	m.flowMap = false
	m.inStepEarly = true
	m.zeroTarget = true

	if err := m.SetText(0, inCtx, tokens); err != nil {
		t.Fatalf("SetText (inStepEarly): %v", err)
	}
	outInStep, err := m.Step(0, inX, 16, 16, inT, inT)
	if err != nil {
		t.Fatalf("Step (inStepEarly): %v", err)
	}

	// Both DiT outputs must agree to floating-point rounding
	compareRMS(t, "cached path vs in-Step path DiT output", outCached, outInStep, 1e-3)

	// 3. With flowMap = true: each of two Steps runs the early text blocks once (2x the per-run count).
	m.Trim()
	m.flowMap = true
	m.inStepEarly = false
	m.zeroTarget = false

	dispatchesBeforeFlowSet := m.earlyDispatches
	if err := m.SetText(0, inCtx, tokens); err != nil {
		t.Fatalf("SetText with flow_map: %v", err)
	}
	// With flow_map, SetText does not run early blocks
	if m.earlyDispatches != dispatchesBeforeFlowSet {
		t.Errorf("SetText with flow_map unexpectedly ran early block dispatches: %d != %d", m.earlyDispatches, dispatchesBeforeFlowSet)
	}

	// Step 1 with flow_map runs early text blocks once
	dispatchesBeforeFlowStep1 := m.earlyDispatches
	if _, err := m.Step(0, inX, 16, 16, inT, inT); err != nil {
		t.Fatalf("Step 1 with flow_map: %v", err)
	}
	perRun := m.earlyDispatches - dispatchesBeforeFlowStep1
	key := stepKey{slot: 0, h: 16, w: 16, refTokens: 0, txtTokens: tokens, inStepEarly: false, zeroTarget: false}
	p0EarlyCount := m.stepProgs[key][0].earlyDispatches
	if perRun != p0EarlyCount {
		t.Fatalf("Step 1 with flow_map ran %d early dispatches, want p0 early count %d", perRun, p0EarlyCount)
	}

	// Step 2 with flow_map runs early text blocks once again
	dispatchesBeforeFlowStep2 := m.earlyDispatches
	if _, err := m.Step(0, inX, 16, 16, inT, inT); err != nil {
		t.Fatalf("Step 2 with flow_map: %v", err)
	}
	step2Run := m.earlyDispatches - dispatchesBeforeFlowStep2
	if step2Run != p0EarlyCount {
		t.Errorf("Step 2 with flow_map ran %d early block dispatches, want per-run count %d", step2Run, p0EarlyCount)
	}
	totalFlowSteps := m.earlyDispatches - dispatchesBeforeFlowStep1
	if totalFlowSteps != 2*p0EarlyCount {
		t.Errorf("Two Steps with flow_map ran %d early block dispatches, want 2x per-run count (%d)", totalFlowSteps, 2*p0EarlyCount)
	}
	t.Logf("Early dispatches run: non-flow_map SetText=%d (Steps=0), flow_map per-Step=%d (2 Steps=%d)", recordedEarlyTxt, perRun, totalFlowSteps)
}

func TestStepTimings(t *testing.T) {
	path := ditPath()
	needFile(t, path)
	f := loadFixtures(t)
	d := device(t)

	inCtx := f.read(t, "dit/in.ctx")
	tokens := len(inCtx) / CondWidth
	inX := f.read(t, "dit/in.x")
	inT := f.read(t, "dit/in.t")[0]
	inNext := f.read(t, "dit/in.next")[0]
	latent1024 := make([]float32, 4096*ditInChannels)

	measure := func(prec Precision, name string) {
		m, err := OpenDiT(d, path, prec)
		if err != nil {
			t.Fatal(err)
		}
		defer m.Close()

		wMB := m.WeightBytes() / (1024 * 1024)
		aMB := m.ArenaBytes() / (1024 * 1024)
		peakVRAM := wMB + aMB
		t.Logf("=== %s: Weights %d MB, Arena %d MB, Peak VRAM %d MB ===", name, wMB, aMB, peakVRAM)

		if err := m.SetText(0, inCtx, tokens); err != nil {
			t.Fatal(err)
		}

		// Warmup
		if _, err := m.Step(0, inX, 16, 16, inT, inNext); err != nil {
			t.Fatal(err)
		}
		start256 := time.Now()
		const iters256 = 5
		for i := 0; i < iters256; i++ {
			if _, err := m.Step(0, inX, 16, 16, inT, inNext); err != nil {
				t.Fatal(err)
			}
		}
		dur256 := time.Since(start256) / iters256
		t.Logf("%s Step timing at 256x256 (256 img tokens + %d text tokens): %v per step", name, tokens, dur256)

		// 1024x1024
		if _, err := m.Step(0, latent1024, 64, 64, inT, inNext); err != nil {
			t.Fatal(err)
		}
		start1024 := time.Now()
		const iters1024 = 3
		for i := 0; i < iters1024; i++ {
			if _, err := m.Step(0, latent1024, 64, 64, inT, inNext); err != nil {
				t.Fatal(err)
			}
		}
		dur1024 := time.Since(start1024) / iters1024
		t.Logf("%s Step timing at 1024x1024 (4096 img tokens + %d text tokens): %v per step", name, tokens, dur1024)

		// 1024x1024 with 2 CFG slots
		if err := m.SetText(1, inCtx, tokens); err != nil {
			t.Fatal(err)
		}
		if _, err := m.Step(1, latent1024, 64, 64, inT, inNext); err != nil {
			t.Fatal(err)
		}
		startCFG := time.Now()
		const itersCFG = 3
		for i := 0; i < itersCFG; i++ {
			if _, err := m.Step(0, latent1024, 64, 64, inT, inNext); err != nil {
				t.Fatal(err)
			}
			if _, err := m.Step(1, latent1024, 64, 64, inT, inNext); err != nil {
				t.Fatal(err)
			}
		}
		durCFG := time.Since(startCFG) / itersCFG
		t.Logf("%s 2-slot CFG step timing at 1024x1024: %v per step", name, durCFG)
	}

	measure(PrecisionFP8, "FP8")
	measure(PrecisionFP16, "FP16")
}

func TestStepAfterTrimNeedsText(t *testing.T) {
	path := ditPath()
	needFile(t, path)
	f := loadFixtures(t)
	d := device(t)

	m, err := OpenDiT(d, path)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()

	inCtx := f.read(t, "dit/in.ctx")
	tokens := len(inCtx) / CondWidth
	if err := m.SetText(0, inCtx, tokens); err != nil {
		t.Fatal(err)
	}

	inX := f.read(t, "dit/in.x")
	inT := f.read(t, "dit/in.t")[0]
	inNext := f.read(t, "dit/in.next")[0]
	out1, err := m.Step(0, inX, 16, 16, inT, inNext)
	if err != nil {
		t.Fatalf("Step before Trim: %v", err)
	}

	m.Trim()

	// Step after Trim must fail until SetText is called again.
	if _, err := m.Step(0, inX, 16, 16, inT, inNext); err == nil {
		t.Fatal("Step after Trim succeeded, want error until SetText is called")
	}

	if err := m.SetText(0, inCtx, tokens); err != nil {
		t.Fatalf("SetText after Trim: %v", err)
	}

	out2, err := m.Step(0, inX, 16, 16, inT, inNext)
	if err != nil {
		t.Fatalf("Step after SetText: %v", err)
	}

	wantOut := f.read(t, "dit/out")
	compareRMS(t, "out after Trim+SetText vs dit/out", out2, wantOut, 0.05)
	compareRMS(t, "out after Trim+SetText vs out1", out2, out1, 1e-4)
}

func TestBoundsChecks(t *testing.T) {
	path := ditPath()
	needFile(t, path)
	d := device(t)

	m, err := OpenDiT(d, path)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()

	// SetText bounds checks
	oversizedTxt := make([]float32, (MaxTextTokens+1)*ditTxtIn)
	if err := m.SetText(0, oversizedTxt, MaxTextTokens+1); err == nil {
		t.Error("SetText with tokens > MaxTextTokens succeeded, want error")
	}

	// SetRefs bounds checks
	dummyLat := make([]float32, 16*16*ditInChannels)
	if err := m.SetRefs(0, [][]float32{dummyLat}, [][2]int{}); err == nil {
		t.Error("SetRefs with len(shapes) != len(latents) succeeded, want error")
	}
	oversizedRef := make([]float32, (MaxRefTokens+1)*ditInChannels)
	if err := m.SetRefs(0, [][]float32{oversizedRef}, [][2]int{{1, MaxRefTokens + 1}}); err == nil {
		t.Error("SetRefs with total ref tokens > MaxRefTokens succeeded, want error")
	}

	// Precision validation check
	if _, err := OpenDiT(d, path, Precision(99)); err == nil {
		t.Error("OpenDiT with invalid Precision succeeded, want error")
	}
}

func TestOpenDiTPeakRSS(t *testing.T) {
	path := ditPath()
	needFile(t, path)
	d := device(t)

	resetPeakRSS := func() {
		if err := os.WriteFile("/proc/self/clear_refs", []byte("5\n"), 0644); err != nil {
			t.Fatalf("clear_refs: %v", err)
		}
	}

	measure := func(prec Precision, name string) {
		resetPeakRSS()
		m, err := OpenDiT(d, path, prec)
		if err != nil {
			t.Fatal(err)
		}
		defer m.Close()
		hwm := readPeakRSS()
		t.Logf("Peak host RSS during %s OpenDiT: VmHWM = %d MB (%d bytes)", name, hwm/(1024*1024), hwm)
	}

	measure(PrecisionFP8, "FP8")
	measure(PrecisionFP16, "FP16")
}

func TestDiTVRAMFootprint(t *testing.T) {
	path := ditPath()
	needFile(t, path)
	d := device(t)

	for _, prec := range []struct {
		name string
		p    Precision
	}{
		{"FP8", PrecisionFP8},
		{"FP16", PrecisionFP16},
	} {
		m, err := OpenDiT(d, path, prec.p)
		if err != nil {
			t.Fatalf("OpenDiT(%s) failed: %v", prec.name, err)
		}
		wb := m.WeightBytes()
		ab := m.ArenaBytes()
		total := wb + ab
		t.Logf("%s: weights %d bytes (%.2f GB), arena %d bytes (%.2f GB), total %d bytes (%.2f GB)",
			prec.name, wb, float64(wb)/(1024*1024*1024), ab, float64(ab)/(1024*1024*1024), total, float64(total)/(1024*1024*1024))
		m.Close()
	}
}
