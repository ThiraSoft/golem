package flux3

import (
	"math"
	"os"
	"testing"
	"time"

	"github.com/ThiraSoft/golem/krea2"
)

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

	m, err := OpenDiT(d, path)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()

	inX := f.read(t, "dit/in.x")
	inT := f.read(t, "dit/in.t")[0]
	inNext := f.read(t, "dit/in.next")[0]
	inCtx := f.read(t, "dit/in.ctx")
	tokens := len(inCtx) / CondWidth

	if err := m.SetText(0, inCtx, tokens); err != nil {
		t.Fatalf("SetText: %v", err)
	}

	out, wps, err := m.StepWithWaypoints(0, inX, 16, 16, inT, inNext)
	if err != nil {
		t.Fatalf("StepWithWaypoints: %v", err)
	}

	// Compare waypoints against dit32/ and log gap to dit/ (bf16).
	compareWp := func(name string, got []float32, tol float64) {
		t.Helper()
		want32 := f.read(t, "dit32/"+name)
		compareRMS(t, name+" vs dit32", got, want32, tol)
		if f.has("dit/" + name) {
			wantBF16 := f.read(t, "dit/"+name)
			gap := rmsError(got, wantBF16)
			t.Logf("%s vs dit (bf16): rms error %.4f%%", name, gap*100)
		}
	}

	// FP8 e4m3 per-tensor scale tolerances:
	// emb_in has 2.66% intrinsic fp8 quantization error; deeper layers accumulate quantization noise.
	compareWp("wp.emb_img", wps["wp.emb_img"], 0.03)
	compareWp("wp.txt_early", wps["wp.txt_early"], 0.03)
	compareWp("wp.img_early", wps["wp.img_early"], 0.02)
	compareWp("wp.single_0", wps["wp.single_0"], 0.03)
	compareWp("wp.single_13", wps["wp.single_13"], 0.15)
	compareWp("wp.single_27", wps["wp.single_27"], 0.06)
	compareWp("out", out, 0.05)

	// Reference stream test against i2i/ref_states and i2i/dit.out0.
	i2iTokens := len(i2iCtx) / CondWidth
	if err := m.SetText(0, i2iCtx, i2iTokens); err != nil {
		t.Fatalf("SetText (i2i): %v", err)
	}
	refLatent := f.read(t, "i2i/ref_latent")
	if err := m.SetRefs(0, [][]float32{refLatent}, [][2]int{{13, 19}}); err != nil {
		t.Fatalf("SetRefs: %v", err)
	}
	i2iX := f.read(t, "i2i/x_0")
	outI2I, wpsI2I, err := m.StepWithWaypoints(0, i2iX, 16, 16, inT, inNext)
	if err != nil {
		t.Fatalf("StepWithWaypoints (i2i): %v", err)
	}
	wantRefStates := f.read(t, "i2i/ref_states")
	compareRMS(t, "i2i/ref_states", wpsI2I["ref_states"], wantRefStates, 0.02)
	wantOut0 := f.read(t, "i2i/dit.out0")
	gapOut0 := rmsError(outI2I, wantOut0)
	t.Logf("i2i/dit.out0 vs bf16: rms error %.4f%%", gapOut0*100)
	compareRMS(t, "i2i/dit.out0", outI2I, wantOut0, 0.07)
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

	// Simulate non-flow_map mode: early blocks should run once during SetText/SetRefs,
	// and remain cached across Step calls.
	m.flowMap = false

	inCtx := f.read(t, "dit/in.ctx")
	tokens := len(inCtx) / CondWidth

	dispatchesBefore := m.EarlyDispatches
	if err := m.SetText(0, inCtx, tokens); err != nil {
		t.Fatalf("SetText: %v", err)
	}
	dispatchesAfterSet := m.EarlyDispatches
	if dispatchesAfterSet <= dispatchesBefore {
		t.Errorf("SetText without flow_map did not dispatch early blocks: %d -> %d", dispatchesBefore, dispatchesAfterSet)
	}

	inX := f.read(t, "dit/in.x")
	inT := f.read(t, "dit/in.t")[0]
	// First step
	if _, err := m.Step(0, inX, 16, 16, inT, inT); err != nil {
		t.Fatalf("Step 1: %v", err)
	}
	dispatchesStep1 := m.EarlyDispatches
	if dispatchesStep1 != dispatchesAfterSet {
		t.Errorf("Step 1 unexpectedly dispatched early blocks: %d != %d", dispatchesStep1, dispatchesAfterSet)
	}

	// Second step
	if _, err := m.Step(0, inX, 16, 16, inT, inT); err != nil {
		t.Fatalf("Step 2: %v", err)
	}
	dispatchesStep2 := m.EarlyDispatches
	if dispatchesStep2 != dispatchesAfterSet {
		t.Errorf("Step 2 unexpectedly dispatched early blocks: %d != %d", dispatchesStep2, dispatchesAfterSet)
	}
}

func TestStepTimings(t *testing.T) {
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

	// 1. Warmup and measure at 256x256 (256 tokens)
	inX := f.read(t, "dit/in.x")
	inT := f.read(t, "dit/in.t")[0]
	inNext := f.read(t, "dit/in.next")[0]
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
	t.Logf("Step timing at 256x256 (256 img tokens + %d text tokens): %v per step", tokens, dur256)

	// 2. Measure at 1024x1024 (4096 img tokens + 80 text tokens)
	latent1024 := make([]float32, 4096*ditInChannels)
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
	t.Logf("Step timing at 1024x1024 (4096 img tokens + %d text tokens): %v per step", tokens, dur1024)
}
