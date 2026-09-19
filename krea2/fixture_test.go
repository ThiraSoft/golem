package krea2

// The recordings ref/krea2/dump.py writes under testdata/krea2: raw float32
// files, their shapes in meta.json. A test that finds none skips.

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
	dir := filepath.Join("..", "testdata", "krea2")
	raw, err := os.ReadFile(filepath.Join(dir, "meta.json"))
	if err != nil {
		t.Skipf("no recording (%v): see ref/krea2/README.md", err)
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
		t.Skipf("%s not recorded: see ref/krea2/README.md", name)
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

// compare fails when the largest gap exceeds tolerance times the reference's
// root mean square, and logs where it sits otherwise.
func compare(t testing.TB, name string, got, want []float32, tolerance float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: %d values, want %d", name, len(got), len(want))
	}
	var norm float64
	for _, v := range want {
		norm += float64(v) * float64(v)
	}
	scale := math.Sqrt(norm/float64(len(want))) + 1e-12
	worst, at := 0.0, 0
	for i := range got {
		if g := math.Abs(float64(got[i] - want[i])); g > worst || math.IsNaN(g) {
			worst, at = g, i
			if math.IsNaN(g) {
				break
			}
		}
	}
	rel := worst / scale
	if !(rel <= tolerance) {
		t.Errorf("%s: gap %.3g at %d (got %g, want %g), %.4f%% of the scale, beyond %.4f%%; rms error %.4f%%",
			name, worst, at, got[at], want[at], rel*100, tolerance*100, rmsError(got, want)*100)
		return
	}
	t.Logf("%s: max gap %.3g, %.4f%% of the scale; rms error %.4f%%", name, worst, rel*100, rmsError(got, want)*100)
}

// rmsError is the root mean square of the difference over that of the
// reference: what a reference computed in bf16 is fairly held to, where one
// entry's gap measures the reference's own rounding at its largest values.
func rmsError(got, want []float32) float64 {
	var d, n float64
	for i := range want {
		e := float64(got[i] - want[i])
		d += e * e
		n += float64(want[i]) * float64(want[i])
	}
	return math.Sqrt(d / (n + 1e-30))
}

// compareRMS fails when the rms error passes tolerance.
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
