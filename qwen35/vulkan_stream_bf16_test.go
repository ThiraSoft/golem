package qwen35

// The bfloat16 window against the widened one it replaced.
//
// vk/matvec_bf16_test.go holds the kernel to the float kernel bit for bit, which
// is the arithmetic. This holds the *plumbing*: that a bfloat16 checkpoint is
// recognised, that its bytes go up untouched, that every projection that should
// read them binds the bfloat16 product and the two decay projections beside them
// do not, and that a window sized at two bytes a weight is still a window the
// card holds.
//
// None of that is exercised by anything else on this machine — the only qwen35
// checkpoints here besides this one are Q4_0 and Q4_K_M, and those take the
// widened path exactly as they did before.
//
// It runs the same positions twice over the same file and asks for the same
// answer. That is a stronger assertion than a tolerance against the processor
// and a very much cheaper one: widening a bfloat16 to a float is a shift, the
// kernel does the shift the host used to, and the products are then summed in
// the same order by the same shader. What is left to differ is nothing.

import (
	"fmt"
	"os"
	"testing"
	"time"
)

func TestStreamedBF16MatchesWidened(t *testing.T) {
	if testing.Short() {
		t.Skip("streams a fifty-two gigabyte checkpoint twice")
	}
	if _, err := os.Stat(qwen38BF16); err != nil {
		t.Skipf("%s is not there", qwen38BF16)
	}

	// Few positions on purpose. Both passes walk the whole trunk whatever the
	// corpus is — the cost here is the checkpoint crossing the bus, not the
	// arithmetic over it — so the positions only have to be enough that a
	// projection reading the wrong half of a word could not agree by accident.
	ids := calibTokens(8)

	run := func(widened bool) ([][][]float32, time.Duration) {
		t.Helper()
		wideF32Only = widened
		defer func() { wideF32Only = false }()

		m, err := Open(qwen38BF16, 256)
		if err != nil {
			t.Skipf("open: %v", err)
		}
		defer m.Close()
		if !m.FloatWeights() {
			t.Fatalf("%s is not a float checkpoint", qwen38BF16)
		}
		form := m.wideForm()
		start := time.Now()
		out, err := m.ForwardStreamed([][]int32{ids}, 0, 256)
		if err != nil {
			t.Fatalf("streamed (%v): %v", form, err)
		}
		took := time.Since(start)
		fmt.Printf("%v: %d positions in %v, %d bytes a block\n",
			form, len(ids), took.Round(time.Millisecond), m.StreamBlockBytes())
		return out, took
	}

	// The widened pass first, so that the bfloat16 one is not the pass that
	// warms the page cache for the other.
	want, slow := run(true)
	got, fast := run(false)
	fmt.Printf("bfloat16 against widened: %v against %v\n",
		fast.Round(time.Millisecond), slow.Round(time.Millisecond))

	if len(got) != len(want) || len(got[0]) != len(want[0]) {
		t.Fatalf("%d runs of %d positions against %d of %d",
			len(got), len(got[0]), len(want), len(want[0]))
	}
	worst, at := 0.0, -1
	for i := range want[0] {
		if d := rel(got[0][i], want[0][i]); d > worst {
			worst, at = d, i
		}
	}
	fmt.Printf("worst position %d at %.3e\n", at, worst)
	// Not zero, and it should not be asked to be: the two passes size their
	// windows differently — two bytes a weight against four — so a block does
	// not always fall in the same window, and a window boundary is where the
	// state crosses to the host and back as float32. What that costs is the
	// rounding of a state, not of a weight.
	if worst > 1e-5 {
		t.Fatalf("position %d differs by %.3e between the bfloat16 window and the widened one", at, worst)
	}
}
