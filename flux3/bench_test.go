package flux3

import (
	"encoding/json"
	"math/rand"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/ThiraSoft/golem/internal/heavy"
	"github.com/ThiraSoft/golem/token/bytebpe"
)

// TestAcceptMatchesComfyUI compares the acceptance fixture (1024x1024, seed 42,
// 16 steps, Euler/simple) against golem generated output for both FP8 and FP16.
func TestAcceptMatchesComfyUI(t *testing.T) {
	f := loadFixtures(t)
	needFile(t, DiTPath())
	needFile(t, EncoderPath())
	needFile(t, VAEPath())
	needFile(t, filepath.Join(TokenizerDir(), "vocab.json"))

	raw, err := os.ReadFile(filepath.Join(f.dir, "accept", "request.json"))
	if err != nil {
		t.Fatal(err)
	}
	var r Request
	if err := json.Unmarshal(raw, &r); err != nil {
		t.Fatal(err)
	}

	wantRGB := f.read(t, "accept/rgb")

	// Measure FP16
	t.Log("=== Testing FP16 on accept/ ===")
	p16, err := Open(Options{Keep: true, Precision: PrecisionFP16})
	if err != nil {
		t.Fatalf("Open FP16 failed: %v", err)
	}
	defer p16.Close()

	img16, tm16, err := p16.Generate(r, nil)
	if err != nil {
		t.Fatalf("FP16 Generate failed: %v", err)
	}
	val16 := psnr(img16, wantRGB)
	t.Logf("accept/ FP16 (1024x1024, 16 steps): PSNR %.2f dB; encode %v, load %v, sample %v, decode %v, total %v",
		val16, tm16.Encode, tm16.Load, tm16.Sample, tm16.Decode, tm16.Total)

	p16.Close()

	// Measure FP8
	t.Log("=== Testing FP8 on accept/ ===")
	p8, err := Open(Options{Keep: true, Precision: PrecisionFP8})
	if err != nil {
		t.Fatalf("Open FP8 failed: %v", err)
	}
	defer p8.Close()

	img8, tm8, err := p8.Generate(r, nil)
	if err != nil {
		t.Fatalf("FP8 Generate failed: %v", err)
	}
	val8 := psnr(img8, wantRGB)
	t.Logf("accept/ FP8 (1024x1024, 16 steps): PSNR %.2f dB; encode %v, load %v, sample %v, decode %v, total %v",
		val8, tm8.Encode, tm8.Load, tm8.Sample, tm8.Decode, tm8.Total)

	gap := val16 - val8
	t.Logf("PSNR comparison on accept/: FP16 = %.2f dB, FP8 = %.2f dB, gap = %.2f dB", val16, val8, gap)
	if gap > 3.0 {
		t.Logf("DECISION: FP8 is %.2f dB below FP16 (> 3 dB threshold), default should be FP16", gap)
	} else {
		t.Logf("DECISION: FP8 is %.2f dB below FP16 (<= 3 dB threshold), FP8 remains acceptable", gap)
	}

	if val16 < 25.0 {
		t.Errorf("accept/ FP16 PSNR %.2f dB < 25 dB", val16)
	}
	if val8 < 20.0 {
		t.Errorf("accept/ FP8 PSNR %.2f dB < 20 dB", val8)
	}
}

// TestFullSizeTimings measures timings at full sizes (1024x1024 and 768x1024)
// for encoder, DiT load, DiT per step, VAE decode, and total picture with/without Keep.
func TestFullSizeTimings(t *testing.T) {
	heavy.Skip(t, "a measurement at full size")
	needFile(t, DiTPath())
	needFile(t, EncoderPath())
	needFile(t, VAEPath())
	needFile(t, filepath.Join(TokenizerDir(), "vocab.json"))

	d := device(t)

	// 1. Text Encoder timings
	tok, err := bytebpe.LoadFiles(TokenizerDir())
	if err != nil {
		t.Fatal(err)
	}
	prompt := "close up profile of a young male athlete with closed eyes pressing his sweaty forehead against a soccer ball, right hand gripping the ball, stark white background, athletic jacket, white ball with jagged red graphic patterns, high contrast studio lighting, highly detailed skin texture, sharp focus Absolutely no text, no typography, no letters, no words, no numbers, no logos, no watermark, no captions, pure imagery only."
	ids, real := Tokenize(tok, prompt)

	tEncStart := time.Now()
	enc, err := OpenEncoder(d, EncoderPath(), true)
	if err != nil {
		t.Fatal(err)
	}
	tEncLoad := time.Since(tEncStart)

	tEncFirst := time.Now()
	cond, err := enc.EncodeMasked(ids, real)
	if err != nil {
		t.Fatal(err)
	}
	tEncFirstRun := time.Since(tEncFirst)

	tEncSecond := time.Now()
	if _, err := enc.EncodeMasked(ids, real); err != nil {
		t.Fatal(err)
	}
	tEncSecondRun := time.Since(tEncSecond)
	t.Logf("Text encoder: load %v, first encode (cold cache) %v, second encode (warm) %v",
		tEncLoad, tEncFirstRun, tEncSecondRun)
	enc.Close()

	// 2. DiT load timings (FP8 vs FP16)
	for _, prec := range []struct {
		name string
		p    Precision
	}{
		{"FP8", PrecisionFP8},
		{"FP16", PrecisionFP16},
	} {
		tLoad1 := time.Now()
		m1, err := OpenDiT(d, DiTPath(), prec.p)
		if err != nil {
			t.Fatal(err)
		}
		durLoad1 := time.Since(tLoad1)
		m1.Close()

		tLoad2 := time.Now()
		m2, err := OpenDiT(d, DiTPath(), prec.p)
		if err != nil {
			t.Fatal(err)
		}
		durLoad2 := time.Since(tLoad2)
		m2.Close()
		t.Logf("DiT %s load: first load %v, second load %v", prec.name, durLoad1, durLoad2)
	}

	// 3. VAE decode timings at 1024x1024 and 768x1024
	for _, sz := range [][2]int{{1024, 1024}, {768, 1024}} {
		w, h := sz[0], sz[1]
		latW, latH := w/16, h/16
		v, err := OpenVAE(d, VAEPath(), latW*latH)
		if err != nil {
			t.Fatal(err)
		}
		rnd := rand.New(rand.NewSource(42))
		latent := make([]float32, 128*latW*latH)
		for i := range latent {
			latent[i] = float32(rnd.NormFloat64())
		}
		// Warmup
		if _, err := v.Decode(latent, latH, latW); err != nil {
			t.Fatal(err)
		}
		tStart := time.Now()
		for iter := 0; iter < 3; iter++ {
			if _, err := v.Decode(latent, latH, latW); err != nil {
				t.Fatal(err)
			}
		}
		vDur := time.Since(tStart) / 3
		t.Logf("VAE decode %dx%d: %v", w, h, vDur)
		v.Close()
	}

	// 4. DiT step timings (FP8 and FP16 at 1024x1024 and 768x1024)
	for _, prec := range []struct {
		name string
		p    Precision
	}{
		{"FP8", PrecisionFP8},
		{"FP16", PrecisionFP16},
	} {
		m, err := OpenDiT(d, DiTPath(), prec.p)
		if err != nil {
			t.Fatal(err)
		}
		tokens := len(ids)
		if err := m.SetText(0, cond, tokens); err != nil {
			t.Fatal(err)
		}

		for _, sz := range [][2]int{{1024, 1024}, {768, 1024}} {
			w, h := sz[0], sz[1]
			latW, latH := w/16, h/16
			rnd := rand.New(rand.NewSource(42))
			latent := make([]float32, 128*latW*latH)
			for i := range latent {
				latent[i] = float32(rnd.NormFloat64())
			}

			// Warmup step
			if _, err := m.Step(0, latent, latH, latW, 1.0, 0.9); err != nil {
				t.Fatal(err)
			}

			best := time.Hour
			for iter := 0; iter < 4; iter++ {
				tStep := time.Now()
				if _, err := m.Step(0, latent, latH, latW, 0.5, 0.4); err != nil {
					t.Fatal(err)
				}
				elapsed := time.Since(tStep)
				if elapsed < best {
					best = elapsed
				}
			}
			t.Logf("DiT %s step %dx%d: %v", prec.name, w, h, best)
		}
		m.Close()
	}

	// 5. End-to-end Pipeline timings with Keep and without Keep
	for _, prec := range []struct {
		name string
		p    Precision
	}{
		{"FP8", PrecisionFP8},
		{"FP16", PrecisionFP16},
	} {
		for _, steps := range []int{16, 8} {
			for _, sz := range [][2]int{{1024, 1024}, {768, 1024}} {
				w, h := sz[0], sz[1]
				p, err := Open(Options{Keep: true, Precision: prec.p})
				if err != nil {
					t.Fatal(err)
				}

				r := Request{
					Prompt: prompt,
					Width:  w,
					Height: h,
					Steps:  steps,
					CFG:    1.0,
					Seed:   42,
				}

				// Warmup run
				_, _, err = p.Generate(r, nil)
				if err != nil {
					t.Fatalf("Generate warm failed: %v", err)
				}

				// Measured run with Keep (warm)
				_, tmWarm, err := p.Generate(r, nil)
				if err != nil {
					t.Fatalf("Generate warm failed: %v", err)
				}
				t.Logf("Pipeline %s %dx%d (%d steps, Keep=true): total %v (encode %v, load %v, sample %v, decode %v)",
					prec.name, w, h, steps, tmWarm.Total, tmWarm.Encode, tmWarm.Load, tmWarm.Sample, tmWarm.Decode)
				p.Close()

				// Measured run without Keep (cold DiT load) for 16 steps
				if steps == 16 {
					pCold, err := Open(Options{Keep: false, Precision: prec.p})
					if err != nil {
						t.Fatal(err)
					}
					_, tmCold, err := pCold.Generate(r, nil)
					if err != nil {
						t.Fatalf("Generate cold failed: %v", err)
					}
					t.Logf("Pipeline %s %dx%d (%d steps, Keep=false): total %v (encode %v, load %v, sample %v, decode %v)",
						prec.name, w, h, steps, tmCold.Total, tmCold.Encode, tmCold.Load, tmCold.Sample, tmCold.Decode)
					pCold.Close()
				}
			}
		}
	}
}
