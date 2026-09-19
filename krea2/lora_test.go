package krea2

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/ThiraSoft/golem/internal/heavy"
)

// recordedLoRA is the path of the LoRA ref/krea2/dump.py recorded with,
// which its request.json names: the front's, rank 32.
func recordedLoRA(t testing.TB) string {
	t.Helper()
	f := loadFixtures(t)
	raw, err := os.ReadFile(filepath.Join(f.dir, "lora", "sample", "request.json"))
	if err != nil {
		t.Skipf("no LoRA recorded (%v): see ref/krea2/README.md", err)
	}
	var r Request
	if err := json.Unmarshal(raw, &r); err != nil || r.Lora == "" {
		t.Fatalf("lora/sample/request.json names no LoRA (%v)", err)
	}
	path := filepath.Join(LoRADir(), r.Lora)
	needFile(t, path)
	return path
}

func TestOpenLoRA(t *testing.T) {
	path := recordedLoRA(t)
	l, err := OpenLoRA(path)
	if err != nil {
		t.Fatal(err)
	}
	// Eight products in each of 28 blocks and 4 text blocks, rank 32.
	if len(l.deltas) != 8*32 || l.Rank() != 32 {
		t.Fatalf("%d products, rank %d", len(l.deltas), l.Rank())
	}
	d := l.deltas["blocks.3.mlp.down"]
	if d == nil || d.out != ditWidth || d.in != ditFFN || d.alpha != 1 {
		t.Fatalf("blocks.3.mlp.down: %+v", d)
	}
	d = l.deltas["txtfusion.refiner_blocks.1.attn.wk"]
	if d == nil || d.out != txtWidth || d.in != txtWidth {
		t.Fatalf("txtfusion.refiner_blocks.1.attn.wk: %+v", d)
	}
}

// One DiT call with the front's LoRA against ComfyUI's bypass loader, which
// adds the change to each product's answer as this does, in bf16. Without a
// LoRA the same call is 1.8% from the arithmetic; this is held to what the
// call without one is held to.
func TestDiTWithLoRAMatchesComfyUI(t *testing.T) {
	heavy.Skip(t, "uploads twelve gigabytes of DiT")
	f := loadFixtures(t)
	needFile(t, DiTPath())
	l, err := OpenLoRA(recordedLoRA(t))
	if err != nil {
		t.Fatal(err)
	}
	m, err := OpenDiT(device(t), DiTPath())
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	if err := m.SetLoRA(l, 1); err != nil {
		t.Fatal(err)
	}
	cond := f.read(t, "lora/dit/context")
	if err := m.SetText(0, cond, len(cond)/CondWidth); err != nil {
		t.Fatal(err)
	}
	fused, err := m.Fused()
	if err != nil {
		t.Fatal(err)
	}
	compareRMS(t, "txtfusion", fused, f.read(t, "lora/dit/txtfusion"), 2e-2)
	text, err := m.Text()
	if err != nil {
		t.Fatal(err)
	}
	compareRMS(t, "txtmlp", text, f.read(t, "lora/dit/txtmlp"), 5e-2)
	shape := f.shape(t, "lora/dit/x")
	x, sigma := f.read(t, "lora/dit/x"), f.read(t, "lora/dit/t")[0]
	out, err := m.Step(0, x, shape[3], shape[4], sigma)
	if err != nil {
		t.Fatal(err)
	}
	compareRMS(t, "out", out, f.read(t, "lora/dit/out"), 4e-2)
	if f.has("lora/dit32/out") {
		compareRMS(t, "txtfusion against float32", fused, f.read(t, "lora/dit32/txtfusion"), 1e-3)
		compareRMS(t, "txtmlp against float32", text, f.read(t, "lora/dit32/txtmlp"), 1e-3)
		compareRMS(t, "out against float32", out, f.read(t, "lora/dit32/out"), 1.5e-2)
	}
	// And the call without it is the recorded call without it: taking the
	// LoRA off leaves nothing behind.
	if err := m.SetLoRA(nil, 0); err != nil {
		t.Fatal(err)
	}
	if err := m.SetText(0, cond, len(cond)/CondWidth); err != nil {
		t.Fatal(err)
	}
	bare, err := m.Step(0, x, shape[3], shape[4], sigma)
	if err != nil {
		t.Fatal(err)
	}
	compareRMS(t, "out without the LoRA", bare, f.read(t, "dit/out"), 4e-2)
	// The first call of the run with the LoRA and of the run without are
	// the same noise and the same text: what the LoRA changes is what it
	// changes in ComfyUI.
	if !f.has("dit/x") || rmsError(x, f.read(t, "dit/x")) != 0 {
		t.Skip("the two recorded calls do not start from the same latent")
	}
	delta := make([]float32, len(out))
	want := f.read(t, "lora/dit/out")
	for i, v := range f.read(t, "dit/out") {
		delta[i] = out[i] - bare[i]
		want[i] -= v
	}
	compareRMS(t, "what the LoRA changes", delta, want, 0.4)
}

// Pictures with the front's LoRA against ComfyUI's, and at two more seeds
// without it, which is what they are to be held beside. The portrait at
// seed 42 without a LoRA is 30.7 dB from ComfyUI's, and at seeds 43 and 44
// 24: 42 is a seed where the two happen to stay close. With the LoRA they
// are 21 dB at 42 and 23 to 25 at 43 and 44, as close as without, and
// ComfyUI's own two loaders part by 23, 31 and 26 dB on the same three.
func TestLoRAPictureMatchesComfyUI(t *testing.T) {
	heavy.Skip(t, "draws pictures with the twelve-gigabyte DiT")
	f := loadFixtures(t)
	needFile(t, DiTPath())
	p, err := Open(Options{Keep: true})
	if err != nil {
		t.Fatal(err)
	}
	defer p.Close()
	for _, run := range []struct {
		tag string
		min float64
	}{{"lora/bypass1", 20}, {"lora/fused1", 18}, {"lora/fused05", 15},
		{"seeds/43/none", 22}, {"seeds/43/bypass1", 22}, {"seeds/43/fused1", 22},
		{"seeds/44/none", 22}, {"seeds/44/bypass1", 22}, {"seeds/44/fused1", 22}} {
		raw, err := os.ReadFile(filepath.Join(f.dir, run.tag, "request.json"))
		if err != nil {
			t.Logf("%s: not recorded", run.tag)
			continue
		}
		var r Request
		if err := json.Unmarshal(raw, &r); err != nil {
			t.Fatal(err)
		}
		img, tm, err := p.Generate(r, nil)
		if err != nil {
			t.Fatal(err)
		}
		got := psnr(img, f.read(t, run.tag+"/image"))
		t.Logf("%s: strength %g: PSNR %.2f dB; load %v, sample %v", run.tag, r.LoraStrength, got, tm.Load, tm.Sample)
		if out := os.Getenv("GOLEM_KREA2_OUT"); out != "" {
			writePNG(t, filepath.Join(out, strings.ReplaceAll(run.tag, "/", "_")+".png"), img)
		}
		if got < run.min {
			t.Errorf("%s: PSNR %.2f dB against ComfyUI's picture, want %.0f", run.tag, got, run.min)
		}
	}
}
