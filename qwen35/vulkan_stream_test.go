package qwen35

import (
	"fmt"
	"math"
	"sort"
	"testing"
	"time"

	"github.com/ThiraSoft/golem/internal/heavy"
)

// rel is the relative difference between two site vectors, which is what a
// salience is compared by: the scale it produces is the vector raised to a
// power, so an agreement here is an agreement about the file.
func rel(a, b []float32) float64 {
	var num, den float64
	for i := range a {
		d := float64(a[i] - b[i])
		num += d * d
		den += float64(a[i]) * float64(a[i])
	}
	return math.Sqrt(num / (den + 1e-30))
}

func worstSite(t *testing.T, a, b map[string][]float32) (string, float64) {
	t.Helper()
	if len(a) != len(b) {
		t.Fatalf("%d sites against %d", len(a), len(b))
	}
	worst, at := 0.0, ""
	for k, va := range a {
		vb, ok := b[k]
		if !ok {
			t.Fatalf("site %q is in one and not the other", k)
		}
		if len(va) != len(vb) {
			t.Fatalf("site %q is %d wide against %d", k, len(va), len(vb))
		}
		if d := rel(va, vb); d > worst {
			worst, at = d, k
		}
	}
	return at, worst
}

func calibTokens(n int) []int32 {
	ids := make([]int32, n)
	for i := range ids {
		ids[i] = int32(100 + i*7)
	}
	return ids
}

// TestStreamedCalibrationIsWindowIndependent is the test the streamed path
// exists to pass. Two window sizes run the same kernels over the same weights
// in the same order; the only thing that differs is where the hidden states
// leave the card and come back, which is the whole of what the streaming adds.
// So the two measurements have to agree, and agree far more closely than any
// two arithmetics would.
//
// It is the guard against the failure this design can actually have: a window
// that carries the wrong state across its seam — the output norm's gains folded
// in, a residual left unclosed, a delta net's state not forgotten between runs.
// Every one of those is invisible in a single window and changes the answer in
// several.
func TestStreamedCalibrationIsWindowIndependent(t *testing.T) {
	heavy.Skip(t, "it calibrates a twenty-seven billion parameter checkpoint on the processor")
	m, err := Open(qwen38, 256)
	if err != nil {
		t.Skipf("open: %v", err)
	}
	defer m.Close()

	runs := [][]int32{calibTokens(96), calibTokens(64)}
	t0 := time.Now()
	wide, rows, err := m.CalibrateStreamed(runs, 0, 256)
	if err != nil {
		t.Fatalf("the card's own window: %v", err)
	}
	fmt.Printf("the card's own window: %d sites over %d rows in %v\n",
		len(wide), rows, time.Since(t0).Round(time.Millisecond))

	t0 = time.Now()
	narrow, rows2, err := m.CalibrateStreamed(runs, 2, 256)
	if err != nil {
		t.Fatalf("two blocks a window: %v", err)
	}
	fmt.Printf("two blocks a window: %d sites over %d rows in %v\n",
		len(narrow), rows2, time.Since(t0).Round(time.Millisecond))

	if rows != rows2 {
		t.Fatalf("%d rows against %d", rows, rows2)
	}
	at, worst := worstSite(t, wide, narrow)
	fmt.Printf("worst site %q at %.3e\n", at, worst)
	// The two differ only in the order the card is handed its work, so this is
	// float rounding and nothing else. A seam that carried the wrong state
	// would not be near this.
	if worst > 1e-5 {
		t.Fatalf("site %q differs by %.3e between a four-block window and a two-block one", at, worst)
	}
}

// TestStreamedCalibrationMatchesResident holds the float path to the quantized
// one it stands in for. The weights are the same numbers — the float form is
// the Q4_0 widened, not a different matrix — so what is left between them is
// what vulkan_test.go's comment already names: the resident path quantizes the
// activation of every projection to Q8_0 where this one keeps floats.
//
// What that leaves is not uniform, and the shape of it is the point:
//
//   - 0/qkv is the input of the first block, which is the embedding and has met
//     no kernel of either path. It has to agree exactly, and it does. This is
//     the assertion that catches a float path computing something else;
//   - across the model the two sit within half a percent, which is the Q8_0
//     activation and nothing else;
//   - the tail is the `down` sites, which read the output of the SwiGLU. x·σ(x)
//     is steep near zero, so a small difference in what it is fed is a large
//     relative one in what it makes, on columns whose values are small. The
//     first block is the worst of them at a fifth, and the next is a
//     seventeenth. That is the nonlinearity, not a disagreement about the
//     arithmetic — and the streamed path is the more exact of the two, since it
//     is the one that did not round the activation to eight bits.
func TestStreamedCalibrationMatchesResident(t *testing.T) {
	heavy.Skip(t, "uploads the model twice")
	m, err := Open(qwen38, 256)
	if err != nil {
		t.Skipf("open: %v", err)
	}
	defer m.Close()

	runs := [][]int32{calibTokens(96)}
	streamed, rows, err := m.CalibrateStreamed(runs, 0, 256)
	if err != nil {
		t.Fatalf("streamed: %v", err)
	}

	r, err := Open(qwen38, 256)
	if err != nil {
		t.Skipf("open: %v", err)
	}
	defer r.Close()
	if err := r.UseVulkanStack(); err != nil {
		t.Skipf("no resident stack to compare against: %v", err)
	}
	if err := r.StartVulkanCalibration(); err != nil {
		t.Fatalf("resident calibration: %v", err)
	}
	r.Reset()
	r.ForwardBatch(runs[0], 0)
	r.CountVulkanCalibration(len(runs[0]))
	resident, rrows, err := r.VulkanCalibrationSums()
	if err != nil {
		t.Fatalf("resident sums: %v", err)
	}
	if rows != rrows {
		t.Fatalf("%d rows streamed against %d resident", rows, rrows)
	}
	if len(streamed) != len(resident) {
		t.Fatalf("%d sites streamed against %d resident", len(streamed), len(resident))
	}

	// The site both paths reach before either has done anything to it.
	if d := rel(streamed["0/qkv"], resident["0/qkv"]); d > 1e-6 {
		t.Fatalf("the first block is fed differently by the two paths, at %.3e — the float path is not reading the model the resident one reads", d)
	}

	all := make([]float64, 0, len(streamed))
	worst, at := 0.0, ""
	for k, a := range streamed {
		b, ok := resident[k]
		if !ok {
			t.Fatalf("site %q is in one and not the other", k)
		}
		d := rel(a, b)
		all = append(all, d)
		if d > worst {
			worst, at = d, k
		}
	}
	sort.Float64s(all)
	median := all[len(all)/2]
	fmt.Printf("%d sites, median %.4f, worst %q at %.4f\n", len(all), median, at, worst)

	// The body of the model, which is where the salience of nearly every matrix
	// comes from. Half a percent is the eight-bit activation; a percent and a
	// half would be something else.
	if median > 0.015 {
		t.Fatalf("the two paths sit %.4f apart across the model, which is more than the Q8_0 activation between them", median)
	}
	// And the tail, which the SwiGLU owns. Loose, and it has to be: what it
	// refuses is a path that answers something unrelated, not one that rounds
	// differently.
	if worst > 0.30 {
		t.Fatalf("site %q differs by %.4f between the streamed float path and the resident Q4_0 one", at, worst)
	}
}

// The BF16 checkpoint, which is the one a conversion reads and the one no
// kernel on the card can hold whole. It is the only checkpoint that exercises
// what the streamed path is for.
const qwen38BF16 = "/mnt/data/LLMs_models/unsloth/Qwen3.8-27B-GGUF/BF16/Qwen3.8-27B-BF16-merged.gguf"

// TestForwardStreamedMatchesCPU holds the streamed path to the processor on the
// same checkpoint. Both compute in floats over the same BF16 weights —
// nn/matrix.go's BF16 branch reads the float activation, not its Q8_0 form —
// so there is no approximation on either side to explain a difference away
// with. What is left is the order the sums are taken in, and that is small.
//
// This is what a reference has to pass before anything is measured against it.
// A logit dump is the number every divergence in compress/README.md is taken
// from, and a wrong one looks exactly like a right one.
func TestForwardStreamedMatchesCPU(t *testing.T) {
	heavy.Skip(t, "streams a fifty-four gigabyte checkpoint")
	m, err := Open(qwen38BF16, 256)
	if err != nil {
		t.Skipf("open: %v", err)
	}
	defer m.Close()
	if !m.FloatWeights() {
		t.Fatalf("%s is not a float checkpoint", qwen38BF16)
	}

	// Short, because the processor is what costs here: it reads the whole
	// checkpoint for every window of positions.
	ids := calibTokens(24)
	t0 := time.Now()
	got, err := m.ForwardStreamed([][]int32{ids}, 0, 256)
	if err != nil {
		t.Fatalf("streamed: %v", err)
	}
	fmt.Printf("streamed %d positions in %v\n", len(ids), time.Since(t0).Round(time.Millisecond))

	c, err := Open(qwen38BF16, 256)
	if err != nil {
		t.Skipf("open: %v", err)
	}
	defer c.Close()
	t0 = time.Now()
	c.Reset()
	want := c.ForwardBatch(ids, 0)
	fmt.Printf("processor %d positions in %v\n", len(ids), time.Since(t0).Round(time.Millisecond))

	worst, at := 0.0, -1
	for i := range ids {
		if d := rel(got[0][i], want[i]); d > worst {
			worst, at = d, i
		}
	}
	fmt.Printf("worst position %d at %.3e\n", at, worst)
	if worst > 5e-3 {
		t.Fatalf("position %d differs by %.3e between the streamed path and the processor", at, worst)
	}
}
