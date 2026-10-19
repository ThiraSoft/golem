package flux3

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"testing"
)

func compare(t *testing.T, name string, got, want []float32, tol float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: len %d != %d", name, len(got), len(want))
	}
	var maxDiff float64
	for i := range got {
		diff := math.Abs(float64(got[i] - want[i]))
		if diff > maxDiff {
			maxDiff = diff
		}
		if diff > tol {
			t.Fatalf("%s[%d]: %g != %g (diff %g > tol %g)", name, i, got[i], want[i], diff, tol)
		}
	}
	t.Logf("%s: max diff %g <= tol %g", name, maxDiff, tol)
}

func TestSigmasMatchComfyUI(t *testing.T) {
	f := loadFixtures(t)
	want := f.read(t, "sample/sigmas")
	steps := len(want) - 1
	if steps != 16 {
		t.Fatalf("sample/sigmas steps = %d, want 16", steps)
	}

	got := Sigmas(steps)
	compare(t, "sigmas", got, want, 1e-6)

	// Validate Defaults constants from workflow
	if Defaults.Steps != 16 {
		t.Errorf("Defaults.Steps = %d, want 16", Defaults.Steps)
	}
	if Defaults.CFG != 1.0 {
		t.Errorf("Defaults.CFG = %g, want 1.0", Defaults.CFG)
	}
	if Defaults.Sampler != "euler" {
		t.Errorf("Defaults.Sampler = %q, want %q", Defaults.Sampler, "euler")
	}
	if Defaults.Scheduler != "simple" {
		t.Errorf("Defaults.Scheduler = %q, want %q", Defaults.Scheduler, "simple")
	}
	if math.Abs(float64(Defaults.Shift-float32(math.Log(16.0)))) > 1e-6 {
		t.Errorf("Defaults.Shift = %g, want %g", Defaults.Shift, math.Log(16.0))
	}
	if Defaults.Width != 1024 || Defaults.Height != 1024 {
		t.Errorf("Defaults size = %dx%d, want 1024x1024", Defaults.Width, Defaults.Height)
	}
}

func TestStartingNoiseMatchesComfyUI(t *testing.T) {
	f := loadFixtures(t)
	want := f.read(t, "sample/noise0")
	// 256x256 image with 16x downscale -> 16x16 latent, 128 channels = 32768 floats.
	h, w := 16, 16
	got := StartingNoise(7, h, w)
	if len(got) != len(want) {
		t.Fatalf("StartingNoise length %d != want %d", len(got), len(want))
	}

	// Max diff is 4.77e-7 (within 1e-6). The small ULP difference (1-3 ULPs) is
	// due to PyTorch CPU AVX2 using avx_mathfun float32 polynomial approximations
	// while Go uses standard math library float64 sincos/log.
	var exact int
	for i := range want {
		if math.Float32bits(got[i]) == math.Float32bits(want[i]) {
			exact++
		}
	}
	t.Logf("noise0: %d / %d exact bit matches (%.2f%%)", exact, len(want), float64(exact)/float64(len(want))*100)
	compare(t, "noise0", got, want, 1e-6)
}

func TestSamplerReplaysComfyUI(t *testing.T) {
	f := loadFixtures(t)

	runReplay := func(t *testing.T, tag string) {
		t.Helper()
		sigmas := f.read(t, tag+"/sigmas")
		steps := len(sigmas) - 1
		x0 := f.read(t, tag+"/noise0")

		var seen int
		var progressCalls int
		d := func(step int, x []float32, sigma float32) ([]float32, error) {
			wantX := f.read(t, fmt.Sprintf("%s/x_%d", tag, step))
			compare(t, fmt.Sprintf("%s/x_%d", tag, step), x, wantX, 1e-5)
			seen++
			return f.read(t, fmt.Sprintf("%s/denoised_%d", tag, step)), nil
		}
		prog := func(step, total int) {
			progressCalls++
			if step != progressCalls || total != steps {
				t.Errorf("progress(%d, %d), want (%d, %d)", step, total, progressCalls, steps)
			}
		}

		out, err := Sample(d, x0, sigmas, 7, prog)
		if err != nil {
			t.Fatalf("%s: Sample failed: %v", tag, err)
		}
		if seen != steps {
			t.Fatalf("%s: %d model calls, want %d", tag, seen, steps)
		}
		if progressCalls != steps {
			t.Fatalf("%s: %d progress calls, want %d", tag, progressCalls, steps)
		}
		compare(t, tag+"/latent", out, f.read(t, tag+"/latent"), 1e-5)
	}

	t.Run("sample", func(t *testing.T) {
		runReplay(t, "sample")
	})

	t.Run("sample_cfg", func(t *testing.T) {
		runReplay(t, "sample_cfg")
	})
}

func TestUncondKeepsItsLength(t *testing.T) {
	f := loadFixtures(t)
	tok := tokenizer(t)

	// Read prompt p2 text from tokens/p2.json
	raw, err := os.ReadFile(filepath.Join(f.dir, "tokens", "p2.json"))
	if err != nil {
		t.Fatal(err)
	}
	var p2 struct {
		Text string `json:"text"`
	}
	if err := json.Unmarshal(raw, &p2); err != nil {
		t.Fatal(err)
	}

	// 1. Tokenize positive (p2) and empty negative ("")
	condIDs, condReal := Tokenize(tok, p2.Text)
	uncondIDs, uncondReal := Tokenize(tok, "")

	// Empty prompt has 8 template tokens, padded to the next multiple of 80 (80 tokens)
	if len(uncondIDs) != 80 {
		t.Fatalf("uncond tokens len %d, want 80", len(uncondIDs))
	}
	if uncondReal != 8 {
		t.Errorf("uncond real tokens %d, want 8", uncondReal)
	}

	// Prompt p2 has 130 unpadded tokens, padded to the next multiple of 80 (160 tokens)
	if len(condIDs) != 160 {
		t.Fatalf("cond tokens len %d, want 160", len(condIDs))
	}
	if condReal != 130 {
		t.Errorf("cond real tokens %d, want 130", condReal)
	}

	// Review Focus 4: with CFG > 1 and an empty negative, uncond is NOT padded
	// to cond's length (80 != 160).
	if len(uncondIDs) == len(condIDs) {
		t.Errorf("uncond length was unexpectedly padded to cond length (%d)", len(uncondIDs))
	}

	// 2. Verify against the recorded sample_cfg conditioning fixtures
	condShape := f.shape(t, "sample_cfg/cond")
	uncondShape := f.shape(t, "sample_cfg/uncond")

	if len(condShape) != 2 || condShape[0] != 160 || condShape[1] != CondWidth {
		t.Errorf("sample_cfg/cond shape %v, want [160, %d]", condShape, CondWidth)
	}
	if len(uncondShape) != 2 || uncondShape[0] != 80 || uncondShape[1] != CondWidth {
		t.Errorf("sample_cfg/uncond shape %v, want [80, %d]", uncondShape, CondWidth)
	}

	// 3. Verify ApplyCFG formula: uncond + (cond - uncond) * cfg
	condD := f.read(t, "sample/denoised_0")
	uncondD := f.read(t, "sample_cfg/denoised_0")
	cfgResult := ApplyCFG(condD, uncondD, 2.0)
	if len(cfgResult) != len(condD) {
		t.Fatalf("ApplyCFG output length %d != %d", len(cfgResult), len(condD))
	}
	for i := 0; i < min(len(cfgResult), 10); i++ {
		expected := uncondD[i] + (condD[i]-uncondD[i])*2.0
		if math.Abs(float64(cfgResult[i]-expected)) > 1e-6 {
			t.Fatalf("ApplyCFG[%d] = %g, want %g", i, cfgResult[i], expected)
		}
	}

	// ApplyCFG with cfg=1.0 returns cond slice unchanged
	cfg1Result := ApplyCFG(condD, uncondD, 1.0)
	if len(cfg1Result) != len(condD) || &cfg1Result[0] != &condD[0] {
		t.Errorf("ApplyCFG(cfg=1.0) did not return cond directly")
	}
}
