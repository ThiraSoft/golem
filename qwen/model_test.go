package qwen

import (
	"os"
	"testing"
)

func openModel(t *testing.T) *Model {
	t.Helper()
	m, err := Open(modelPath(t), 4096)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { m.Close() })
	return m
}

// Every block's output, against the reference's recording of it. A divergence
// says which block it began in, which is the whole point of recording them all.
func TestFullStackMatchesReference(t *testing.T) {
	f := loadFixture(t, "layers")
	m := openModel(t)

	hidden := m.ForwardBatch(f.Tokens, 0)
	last := len(f.Tokens) - 1

	for i := 0; i < f.NLayer; i++ {
		name := "l_out-" + itoa(i)
		compareRelative(t, name+" at the last position", m.BlockOutput(i), f.lastColumn(t, name), 1e-2)
	}
	compareRelative(t, "result_norm", hidden[last], f.lastColumn(t, "result_norm"), 2e-2)
}

// The logits, against the sixty-four the reference kept, and the token it would
// have drawn.
func TestLogitsMatchReference(t *testing.T) {
	f := loadFixture(t, "layers")
	m := openModel(t)

	hidden := m.ForwardBatch(f.Tokens, 0)
	out := make([]float32, m.Cfg.Vocab)
	m.Logits(hidden[len(hidden)-1], out)

	got := make([]float32, len(f.LogitsTop))
	want := make([]float32, len(f.LogitsTop))
	for i, probe := range f.LogitsTop {
		got[i], want[i] = out[probe.ID], probe.Logit
	}
	compare(t, "the top 64 logits", got, want, 0.5)

	if best := argmax(out); best != f.Argmax {
		t.Errorf("argmax %d, want %d", best, f.Argmax)
	}
}

// The long run: several hundred positions in, where an indexing mistake in the
// cache that a five-token prompt hides finally shows.
func TestLongRunMatchesReference(t *testing.T) {
	f := loadFixture(t, "long")
	m := openModel(t)

	if len(f.Tokens) < 600 {
		t.Fatalf("the long fixture holds %d tokens", len(f.Tokens))
	}
	hidden := m.ForwardBatch(f.Tokens, 0)
	last := len(f.Tokens) - 1

	// The recording kept only the last column of each waypoint.
	for _, il := range []int{0, 14, 27} {
		name := "l_out-" + itoa(il)
		compareRelative(t, name, m.BlockOutput(il), f.tensor(t, name), 1e-2)
	}
	compareRelative(t, "result_norm", hidden[last], f.tensor(t, "result_norm"), 3e-2)
}

// A batch of positions and the same positions one at a time must give the same
// answer: the batch exists to read the weights once, not to change the
// arithmetic.
func TestBatchAgreesWithOneAtATime(t *testing.T) {
	f := loadFixture(t, "layers")
	m := openModel(t)

	batched := m.ForwardBatch(f.Tokens, 0)
	last := append([]float32(nil), batched[len(batched)-1]...)

	m.Reset()
	var stepped []float32
	for i, tok := range f.Tokens {
		stepped = m.Forward(tok, i)
	}
	compare(t, "the last hidden state, batched against stepped", stepped, last, 1e-3)
}

// Reset must forget the conversation, or a second prompt reads the first one's
// keys and answers something that is nearly right.
func TestResetForgets(t *testing.T) {
	f := loadFixture(t, "layers")
	m := openModel(t)

	first := append([]float32(nil), m.ForwardBatch(f.Tokens, 0)[len(f.Tokens)-1]...)
	m.ForwardBatch(f.Tokens, len(f.Tokens)) // dirty the cache
	m.Reset()
	second := m.ForwardBatch(f.Tokens, 0)[len(f.Tokens)-1]

	compare(t, "the hidden state after a reset", second, first, 0)
}

func TestOpenRefusesAnotherArchitecture(t *testing.T) {
	path := os.Getenv("GOLEM_MODEL")
	if path == "" {
		t.Skip("set GOLEM_MODEL to a Gemma 4 GGUF to run this test")
	}
	if _, err := Open(path, 4096); err == nil {
		t.Fatal("a gemma4 file was opened as qwen3")
	}
}

func argmax(x []float32) int32 {
	best := 0
	for i := 1; i < len(x); i++ {
		if x[i] > x[best] {
			best = i
		}
	}
	return int32(best)
}
