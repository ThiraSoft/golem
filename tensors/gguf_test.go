package tensors

import (
	"os"
	"testing"
)

// modelPath is the GGUF the tests read, named by GOLEM_MODEL. It is not
// versioned — three gigabytes have no place in a git history — so its absence
// skips rather than fails.
func modelPath(t *testing.T) string {
	t.Helper()
	path := os.Getenv("GOLEM_MODEL")
	if path == "" {
		t.Skip("set GOLEM_MODEL to a GGUF to run this test")
	}
	if _, err := os.Stat(path); err != nil {
		t.Skipf("GOLEM_MODEL names %s, which is not there", path)
	}
	return path
}

func TestGGUFMetadata(t *testing.T) {
	g, err := OpenGGUF(modelPath(t))
	if err != nil {
		t.Fatal(err)
	}
	defer g.Close()

	arch, err := g.String("general.architecture")
	if err != nil {
		t.Fatal(err)
	}
	if arch != "gemma4" {
		t.Errorf("architecture %q, want gemma4", arch)
	}

	blocks, err := g.Uint32("gemma4.block_count")
	if err != nil {
		t.Fatal(err)
	}
	if blocks != 35 {
		t.Errorf("block_count %d, want 35", blocks)
	}

	eps, err := g.Float32("gemma4.attention.layer_norm_rms_epsilon")
	if err != nil {
		t.Fatal(err)
	}
	if eps <= 0 || eps > 1e-5 {
		t.Errorf("rms epsilon %g, want about 1e-6", eps)
	}

	// A scalar must read as a slice of one: the 12B stores this key as an array.
	kv, err := g.Uint32Slice("gemma4.attention.head_count_kv")
	if err != nil {
		t.Fatal(err)
	}
	if len(kv) != 1 || kv[0] != 1 {
		t.Errorf("head_count_kv %v, want [1]", kv)
	}

	window, err := g.BoolSlice("gemma4.attention.sliding_window_pattern")
	if err != nil {
		t.Fatal(err)
	}
	if len(window) != 35 {
		t.Fatalf("sliding_window_pattern has %d entries, want 35", len(window))
	}
	// Four windowed blocks then one global, repeating.
	for i, windowed := range window {
		if want := i%5 != 4; windowed != want {
			t.Errorf("block %d: windowed=%v, want %v", i, windowed, want)
		}
	}

	tokens, err := g.Strings("tokenizer.ggml.tokens")
	if err != nil {
		t.Fatal(err)
	}
	if len(tokens) != 262144 {
		t.Errorf("%d tokens, want 262144", len(tokens))
	}
	if tokens[2] != "<bos>" {
		t.Errorf("token 2 is %q, want <bos>", tokens[2])
	}
}

func TestGGUFTensors(t *testing.T) {
	g, err := OpenGGUF(modelPath(t))
	if err != nil {
		t.Fatal(err)
	}
	defer g.Close()

	if len(g.Tensors) != 541 {
		t.Errorf("%d tensors, want 541", len(g.Tensors))
	}

	// A windowed block: 8 query heads of 256.
	q, err := g.Get("blk.0.attn_q.weight")
	if err != nil {
		t.Fatal(err)
	}
	if want := []int{1536, 2048}; !equalInts(q.Shape, want) {
		t.Errorf("blk.0.attn_q shape %v, want %v", q.Shape, want)
	}
	if q.DType != "Q4_0" {
		t.Errorf("blk.0.attn_q dtype %q, want Q4_0", q.DType)
	}
	// 2048 rows of 1536 weights: 48 blocks of 18 bytes each.
	if want := 2048 * (1536 / 32) * 18; len(q.Raw) != want {
		t.Errorf("blk.0.attn_q holds %d bytes, want %d", len(q.Raw), want)
	}

	// A global block: 8 query heads of 512, one key head of 512.
	k, err := g.Get("blk.4.attn_k.weight")
	if err != nil {
		t.Fatal(err)
	}
	if want := []int{1536, 512}; !equalInts(k.Shape, want) {
		t.Errorf("blk.4.attn_k shape %v, want %v", k.Shape, want)
	}

	// Past block 20 the keys and values are shared, so they are simply absent.
	if _, err := g.Get("blk.20.attn_k.weight"); err == nil {
		t.Error("blk.20.attn_k.weight should not exist: it is shared")
	}

	if _, err := g.Get("no.such.tensor"); err == nil {
		t.Error("a missing tensor should be an error")
	}
}

func equalInts(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
