package yue2

// The recordings ref/yue2/dump_vae.cpp writes under testdata/yue2: raw float32
// files, their shapes in meta.json.

import (
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

type fixtures struct {
	dir    string
	shapes map[string][]int
}

func loadFixtures(t testing.TB) *fixtures {
	t.Helper()
	dir := filepath.Join("..", "testdata", "yue2")
	raw, err := os.ReadFile(filepath.Join(dir, "meta.json"))
	if err != nil {
		t.Skipf("no recording (%v): see ref/yue2/README.md", err)
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
		t.Skipf("%s not recorded: see ref/yue2/README.md", name)
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

func (f *fixtures) readInt32(t testing.TB, name string) []int32 {
	t.Helper()
	f.shape(t, name)
	b, err := os.ReadFile(filepath.Join(f.dir, name))
	if err != nil {
		t.Fatal(err)
	}
	out := make([]int32, len(b)/4)
	for i := range out {
		out[i] = int32(binary.LittleEndian.Uint32(b[i*4:]))
	}
	return out
}

func (f *fixtures) readF16(t testing.TB, name string) []uint16 {
	t.Helper()
	f.shape(t, name)
	b, err := os.ReadFile(filepath.Join(f.dir, name))
	if err != nil {
		t.Fatal(err)
	}
	out := make([]uint16, len(b)/2)
	for i := range out {
		out[i] = binary.LittleEndian.Uint16(b[i*2:])
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
		t.Errorf("%s: rms error %.4f%%, beyond %.4f%%\n got:  %v\n want: %v", name, e*100, tolerance*100, got[:min(len(got), 5)], want[:min(len(want), 5)])
		return
	}
	t.Logf("%s: rms error %.4f%%", name, e*100)
}

func snrDB(got, want []float32) float64 {
	if len(got) != len(want) {
		return math.NaN()
	}
	var num, den float64
	for i := range want {
		d := float64(got[i] - want[i])
		num += d * d
		den += float64(want[i]) * float64(want[i])
	}
	if num <= 0 {
		return math.Inf(1)
	}
	return 10 * math.Log10(den/(num+1e-30))
}

func TestFixturesPresent(t *testing.T) {
	f := loadFixtures(t)
	required := []string{
		"vae/latent",
		"vae/conv1",
		"vae/block_0",
		"vae/block_1",
		"vae/block_2",
		"vae/block_3",
		"vae/block_4",
		"vae/block_5",
		"vae/audio",
		"vae/b0_ct",
		"vae/b0_ru0",
		"vae/b0_ru1",
		"vae/b0_ru2",
		"vae_f32acc/conv1",
		"vae_f32acc/b0_ct",
		"vae_f32acc/b0_ru0",
		"vae_f32acc/b0_ru1",
		"vae_f32acc/b0_ru2",
		"vae_f32acc/block_0",
		"vae_f32acc/block_1",
		"vae_f32acc/block_2",
		"vae_f32acc/block_3",
		"vae_f32acc/block_4",
		"vae_f32acc/block_5",
		"vae_f32acc/audio",
		"vae_tiled/latent",
		"vae_tiled/audio",
		"nar/ar_ids",
		"nar/kv_k_l0",
		"nar/kv_k_l13",
		"nar/kv_k_l27",
		"nar/kv_v_l0",
		"nar/kv_v_l13",
		"nar/kv_v_l27",
		"nar/kv_all_k.f16",
		"nar/kv_all_v.f16",
		"nar/noise",
		"nar/temb_t",
		"nar/hidden_after_input",
		"nar/layer0_sa_output",
		"nar/hidden_after_layer0",
		"nar/hidden_after_layer7",
		"nar/hidden_after_layer14",
		"nar/hidden_after_layer21",
		"nar/hidden_after_layer27",
		"nar/v_first_0",
		"nar/v_second_0",
		"nar/xt_0",
		"nar/xt_1",
		"nar/xt_7",
		"nar/xt_31",
		"nar/latent",
		"rng/normal_37",
		"rng/normal_6400",
	}
	for _, req := range required {
		if !f.has(req) {
			t.Errorf("missing fixture: %s", req)
			continue
		}
		shape := f.shape(t, req)
		expectedLen := 1
		for _, dim := range shape {
			expectedLen *= dim
		}
		var gotLen int
		if req == "nar/ar_ids" {
			gotLen = len(f.readInt32(t, req))
		} else if strings.HasSuffix(req, ".f16") {
			gotLen = len(f.readF16(t, req))
		} else {
			gotLen = len(f.read(t, req))
		}
		if gotLen != expectedLen {
			t.Errorf("%s: length %d != expected %d from shape %v", req, gotLen, expectedLen, shape)
		}
	}
}

func TestRNGFixtures(t *testing.T) {
	f := loadFixtures(t)
	if !f.has("rng/normal_6400") || !f.has("nar/noise") {
		t.Skip("rng fixtures not present")
	}
	normal6400 := f.read(t, "rng/normal_6400")
	noise := f.read(t, "nar/noise")
	if len(normal6400) != len(noise) {
		t.Fatalf("length mismatch: %d vs %d", len(normal6400), len(noise))
	}
	for i := range normal6400 {
		if normal6400[i] != noise[i] {
			t.Fatalf("mismatch at %d: %v vs %v", i, normal6400[i], noise[i])
		}
	}
}
