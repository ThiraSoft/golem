package krea2

import (
	"fmt"
	"testing"

	"github.com/ThiraSoft/golem/internal/heavy"
)

// Every step of the recorded run, the DiT given the very latent ComfyUI gave
// it: what the model answers at each noise level, apart from what the steps
// before did.
func TestEveryStepMatchesComfyUI(t *testing.T) {
	heavy.Skip(t, "uploads twelve gigabytes of DiT")
	f := loadFixtures(t)
	needFile(t, DiTPath())
	m, err := OpenDiT(device(t), DiTPath())
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	cond := f.read(t, "sample/cond")
	if err := m.SetText(0, cond, len(cond)/CondWidth); err != nil {
		t.Fatal(err)
	}
	sigmas := f.read(t, "sample/sigmas")
	sigmas[0] = float32(1 / (1 + (1/(1-1e-4)-1)/1*expShift()))
	shape := f.shape(t, "sample/x0")
	h, w := shape[3], shape[4]
	for i := 0; i < len(sigmas)-1; i++ {
		x := f.read(t, fmt.Sprintf("sample/x%d", i))
		v, err := m.Step(0, x, h, w, sigmas[i])
		if err != nil {
			t.Fatal(err)
		}
		for k := range v {
			v[k] = x[k] - sigmas[i]*v[k]
		}
		compareRMS(t, fmt.Sprintf("denoised%d at sigma %.4f", i, sigmas[i]), v, f.read(t, fmt.Sprintf("sample/denoised%d", i)), 1)
		if ref := fmt.Sprintf("dit32_%d/out", i); f.has(ref) {
			vel, _ := m.Step(0, x, h, w, sigmas[i])
			compareRMS(t, fmt.Sprintf("velocity%d against float32", i), vel, f.read(t, ref), 1)
			want := f.read(t, fmt.Sprintf("sample/denoised%d", i))
			bf := make([]float32, len(want))
			for k := range want {
				bf[k] = (x[k] - want[k]) / sigmas[i]
			}
			compareRMS(t, fmt.Sprintf("ComfyUI's velocity%d against float32", i), bf, f.read(t, ref), 1)
		}
	}
}
