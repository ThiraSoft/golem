package tha3

// Parity with the upstream PyTorch code, through the recordings ref/tha3/dump.py
// writes under testdata/tha3/<pose>/. Each network is fed what it was fed
// there and every waypoint it emits is compared in the order it is produced,
// so the first one out of tolerance says where a mistake is.

import (
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/ThiraSoft/golem/internal/kyutai/reference"
)

const tolerance = 1e-3

var runs = []string{"neutral", "aaa", "wink", "head", "mix"}

type fixtures struct {
	dir     string
	Pose    []float32        `json:"pose"`
	Tensors map[string][]int `json:"tensors"`
}

func loadFixtures(t *testing.T, run string) *fixtures {
	t.Helper()
	dir := filepath.Join("..", "testdata", "tha3", run)
	raw, err := os.ReadFile(filepath.Join(dir, "fixtures.json"))
	if err != nil {
		t.Skipf("no recording for %q (%v): see ref/tha3/README.md", run, err)
	}
	f := &fixtures{dir: dir}
	if err := json.Unmarshal(raw, f); err != nil {
		t.Fatal(err)
	}
	return f
}

func (f *fixtures) has(name string) bool { _, ok := f.Tensors[name]; return ok }

func (f *fixtures) raw(t *testing.T, name string) []float32 {
	t.Helper()
	b, err := os.ReadFile(filepath.Join(f.dir, name+".bin"))
	if err != nil {
		t.Fatal(err)
	}
	out := make([]float32, len(b)/4)
	for i := range out {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[i*4:]))
	}
	return out
}

// tensor reads a recorded [1, C, H, W] activation.
func (f *fixtures) tensor(t *testing.T, name string) Tensor {
	t.Helper()
	shape, ok := f.Tensors[name]
	if !ok || len(shape) != 4 || shape[0] != 1 {
		t.Fatalf("%s: recorded shape %v, want [1 C H W]", name, shape)
	}
	return Tensor{C: shape[1], H: shape[2], W: shape[3], Data: f.raw(t, name)}
}

// pose reads a recorded [1, n] pose argument.
func (f *fixtures) pose(t *testing.T, name string) []float32 { return f.raw(t, name) }

// checker compares every waypoint that has a recording under net+"."+name,
// and stops the test at the first one out of tolerance.
func (f *fixtures) checker(t *testing.T, net string) tracer {
	return func(name string, got Tensor) {
		t.Helper()
		full := name
		if net != "" {
			full = net + "." + name
		}
		if !f.has(full) {
			return
		}
		want := f.tensor(t, full)
		if got.C != want.C || got.H != want.H || got.W != want.W {
			t.Fatalf("%s: %dx%dx%d, want %dx%dx%d", full, got.C, got.H, got.W, want.C, want.H, want.W)
		}
		reference.Compare(t, full, got.Data, want.Data, tolerance)
		if t.Failed() {
			t.FailNow()
		}
	}
}

func openNetwork(t *testing.T, network string) *weights {
	t.Helper()
	w, err := openWeights(Dir(), network)
	if err != nil {
		t.Skipf("weights not found (%v): see ref/tha3/README.md", err)
	}
	t.Cleanup(func() { w.close() })
	return w
}

// Building every body from the real files checks every name and shape the
// code expects, before any arithmetic is compared.
func TestBodiesLoad(t *testing.T) {
	cases := []struct {
		network string
		build   func(w *weights)
	}{
		{netEyebrowDecomposer, func(w *weights) { newEncoderDecoder(w, "body", 128, 4, 0, 64, 16, 6, ReLU) }},
		{netEyebrowCombiner, func(w *weights) { newEncoderDecoder(w, "body", 128, 8, 12, 64, 16, 6, ReLU) }},
		{netFaceMorpher, func(w *weights) { newEncoderDecoder(w, "body", 192, 4, 27, 64, 24, 6, ReLU) }},
		{netRotator, func(w *weights) { newResizeEncoderDecoder(w, "encoder_decoder", 256, 10, 64, 32, 6, leaky) }},
		{netEditor, func(w *weights) { newUNet(w, "body", 512, 16, 32, 64, 6, leaky) }},
	}
	for _, c := range cases {
		w := openNetwork(t, c.network)
		c.build(w)
		if err := w.err(); err != nil {
			t.Errorf("%s: %v", c.network, err)
		}
	}
}
