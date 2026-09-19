package krea2

import (
	"testing"

	"github.com/ThiraSoft/golem/internal/heavy"
)

// One DiT call from the recorded run: the text's part, then the step.
//
// ComfyUI computes it in bf16, which on this call is 1.8% (rms) from the
// same modules run in float32 by ref/krea2/dump_f32.py. This is held to the
// float32 run, which it sits well inside of, and to ComfyUI's by the size of
// ComfyUI's own error.
func TestDiTMatchesComfyUI(t *testing.T) {
	heavy.Skip(t, "uploads twelve gigabytes of DiT")
	f := loadFixtures(t)
	needFile(t, DiTPath())
	m, err := OpenDiT(device(t), DiTPath())
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	cond := f.read(t, "dit/context")
	if err := m.SetText(0, cond, len(cond)/CondWidth); err != nil {
		t.Fatal(err)
	}
	fused, err := m.Fused()
	if err != nil {
		t.Fatal(err)
	}
	compareRMS(t, "txtfusion", fused, f.read(t, "dit32/txtfusion"), 1e-3)
	compareRMS(t, "txtfusion against bf16", fused, f.read(t, "dit/txtfusion"), 2e-2)
	text, err := m.Text()
	if err != nil {
		t.Fatal(err)
	}
	compareRMS(t, "txtmlp", text, f.read(t, "dit32/txtmlp"), 1e-3)
	compareRMS(t, "txtmlp against bf16", text, f.read(t, "dit/txtmlp"), 5e-2)
	shape := f.shape(t, "dit/x")
	out, err := m.Step(0, f.read(t, "dit/x"), shape[3], shape[4], f.read(t, "dit/t")[0])
	if err != nil {
		t.Fatal(err)
	}
	compareRMS(t, "out", out, f.read(t, "dit32/out"), 1.5e-2)
	compareRMS(t, "out against bf16", out, f.read(t, "dit/out"), 4e-2)

}
