package flux3

// The recordings ref/flux3/dump.py writes under testdata/flux3: raw float32
// files, their shapes in meta.json.

import (
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"
)

type fixtures struct {
	dir    string
	shapes map[string][]int
}

func loadFixtures(t testing.TB) *fixtures {
	t.Helper()
	dir := filepath.Join("..", "testdata", "flux3")
	raw, err := os.ReadFile(filepath.Join(dir, "meta.json"))
	if err != nil {
		t.Fatalf("no recording (%v): see ref/flux3/README.md", err)
	}
	f := &fixtures{dir: dir}
	if err := json.Unmarshal(raw, &f.shapes); err != nil {
		t.Fatal(err)
	}
	return f
}

func (f *fixtures) has(name string) bool { _, ok := f.shapes[name]; return ok }

func (f *fixtures) shape(t testing.TB, name string) []int {
	t.Helper()
	s, ok := f.shapes[name]
	if !ok {
		t.Fatalf("%s not recorded: see ref/flux3/README.md", name)
	}
	return s
}

func (f *fixtures) read(t testing.TB, name string) []float32 {
	t.Helper()
	f.shape(t, name)
	b, err := os.ReadFile(filepath.Join(f.dir, name))
	if err != nil {
		t.Fatal(err)
	}
	out := make([]float32, len(b)/4)
	for i := range out {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[i*4:]))
	}
	return out
}

func rmsError(got, want []float32) float64 {
	var d, n float64
	for i := range want {
		e := float64(got[i] - want[i])
		d += e * e
		n += float64(want[i]) * float64(want[i])
	}
	return math.Sqrt(d / (n + 1e-30))
}

func compareRMS(t testing.TB, name string, got, want []float32, tolerance float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: %d values, want %d", name, len(got), len(want))
	}
	e := rmsError(got, want)
	if !(e <= tolerance) {
		t.Errorf("%s: rms error %.4f%%, beyond %.4f%%", name, e*100, tolerance*100)
		return
	}
	t.Logf("%s: rms error %.4f%%", name, e*100)
}

func TestFixturesPresent(t *testing.T) {
	f := loadFixtures(t)
	required := []string{
		"tokens/p0.ids", "tokens/p1.ids", "tokens/p2.ids",
		"encoder/p0.hidden_l4", "encoder/p1.hidden_l4", "encoder/p2.hidden_l4",
		"encoder/p0.cond", "encoder/p1.cond", "encoder/p2.cond",
		"sample/sigmas", "sample/noise0", "sample/latent", "sample/rgb",
		"sample_cfg/cond", "sample_cfg/uncond",
		"dit/in.x", "dit/in.t", "dit/in.ctx", "dit/in.next", "dit/pe",
		"dit/wp.emb_img", "dit/wp.txt_early", "dit/wp.img_early",
		"dit/wp.single_0", "dit/wp.single_13", "dit/wp.single_27", "dit/out",
		"dit32/in.x", "dit32/in.t", "dit32/in.ctx", "dit32/in.next", "dit32/pe",
		"dit32/wp.emb_img", "dit32/wp.txt_early", "dit32/wp.img_early",
		"dit32/wp.single_0", "dit32/wp.single_13", "dit32/wp.single_27", "dit32/out",
		"vae/up_0", "vae/up_1", "vae/up_2", "vae/up_3",
		"vae/enc.in", "vae/enc.latent",
		"i2i/ref_latent", "i2i/ref_states", "i2i/dit.out0", "i2i/rgb",
		"mi2i/ref_latents", "mi2i/ref_ids", "mi2i/rgb",
		"accept/latent", "accept/rgb",
	}
	for _, req := range required {
		if !f.has(req) {
			t.Errorf("missing fixture: %s", req)
		}
	}
}
