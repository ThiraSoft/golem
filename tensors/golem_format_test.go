package tensors

import (
	"bytes"
	"encoding/binary"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// golemFixture writes a small, well-formed golem file whose metadata the
// caller can spoil one key at a time. Everything here is synthetic: the guard
// has to be provable without the multi-gigabyte artifacts a real conversion
// produces, or it is only tested when somebody happens to have one.
func golemFixture(t *testing.T, dtype string, edits map[string]any) string {
	t.Helper()
	meta := map[string]any{
		"general.architecture": "qwen3",
		"general.alignment":    uint32(32),
		"golem.format":         GolemFormat,
		"golem.hadamard_group": uint32(128),
		"golem.scale_block":    uint32(64),
		"golem.trellis.seq":    uint32(golemSeq),
		"golem.trellis.bits":   uint32(golemTierBits[dtype]),
		"golem.trellis.state":  uint32(golemState),
	}
	for k, v := range edits {
		if v == nil {
			delete(meta, k)
			continue
		}
		meta[k] = v
	}
	out := []OutTensor{
		{Name: "blk.0.attn_q.weight", Shape: []int{golemSeq, 2}, DType: dtype,
			Data: make([]byte, 2*blockGeometry[dtype][1])},
	}
	path := filepath.Join(t.TempDir(), "fixture.golem")
	if err := WriteGGUF(path, meta, out); err != nil {
		t.Fatal(err)
	}
	return path
}

// The three layers are only worth their cost if each refuses something the
// others let through, so each case here spoils exactly one of them and the
// file is otherwise valid.
func TestGolemFileGuards(t *testing.T) {
	cases := []struct {
		name  string
		dtype string
		edits map[string]any
		want  string // a fragment of the error, so the message stays diagnostic
	}{
		{"a valid file", "T3G", nil, ""},
		{"a valid four-bit file", "T4G", nil, ""},
		{"a valid five-bit file", "T5G", nil, ""},
		{"no golem.format at all", "T3G",
			map[string]any{"golem.format": nil}, "no golem.format"},
		{"a golem.format from another codec", "T3G",
			map[string]any{"golem.format": "lattice/1"}, `is "lattice/1"`},
		{"a golem.format from a later layout", "T3G",
			map[string]any{"golem.format": "trellis/2"}, `is "trellis/2"`},
		{"a sequence length this build does not code", "T3G",
			map[string]any{"golem.trellis.seq": uint32(96)}, "96 weights as one path"},
		{"a state width this build does not carry", "T3G",
			map[string]any{"golem.trellis.state": uint32(10)}, "10 bits of trellis state"},
		{"a body tier no tensor in the file is", "T3G",
			map[string]any{"golem.trellis.bits": uint32(5)}, "no tensor in the file is that tier"},
		{"a lattice-era key", "T3G",
			map[string]any{"golem.d4.hadamard_group": uint32(128)}, "lattice-era converter"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			g, err := OpenGGUF(golemFixture(t, c.dtype, c.edits))
			if err == nil {
				g.Close()
			}
			switch {
			case c.want == "" && err != nil:
				t.Fatalf("a well-formed file was refused: %v", err)
			case c.want != "" && err == nil:
				t.Fatalf("opened without complaint; expected an error naming %q", c.want)
			case c.want != "" && !strings.Contains(err.Error(), c.want):
				t.Fatalf("error is %v, expected one naming %q", err, c.want)
			}
		})
	}
}

// A file with none of golem's tensor types is somebody else's GGUF and must
// open with no golem key at all — the guard is about private types, not about
// every file the reader is pointed at.
func TestPlainGGUFNeedsNoGolemKeys(t *testing.T) {
	path := filepath.Join(t.TempDir(), "plain.gguf")
	err := WriteGGUF(path, map[string]any{"general.architecture": "qwen3"},
		[]OutTensor{{Name: "output_norm.weight", Shape: []int{4}, DType: "F32",
			Data: make([]byte, 16)}})
	if err != nil {
		t.Fatal(err)
	}
	g, err := OpenGGUF(path)
	if err != nil {
		t.Fatalf("a plain GGUF was refused: %v", err)
	}
	g.Close()
}

// The type numbers are the second layer, and their arithmetic is the whole
// reason they were chosen: "glm" in the top three bytes so a hex dump names
// the format, and the rate in the low byte so a new tier needs no decision.
// A renumbering that broke either property would still pass every other test
// in this package.
func TestGolemTypeNumbersSpellThemselves(t *testing.T) {
	for id, name := range ggmlTypes {
		k, private := golemTierBits[name]
		if !private {
			if id>>8 == golemTypeBase>>8 {
				t.Fatalf("%s is not golem's and sits at %#x", name, id)
			}
			continue
		}
		if id>>8 != golemTypeBase>>8 {
			t.Fatalf("%s is %#x, which does not begin with ASCII \"glm\"", name, id)
		}
		if int(id&0xFF) != k {
			t.Fatalf("%s is %#x, whose low byte says %d bits a weight and it codes %d", name, id, id&0xFF, k)
		}
	}
	// The retired numbering must not come back under another name.
	for _, id := range []uint32{1000, 1001, 1002, 1003, 1004} {
		if name, ok := ggmlTypes[id]; ok {
			t.Fatalf("%d is retired and now reads as %s", id, name)
		}
	}
}

// A file written before 2026-08-31 carries type 1000, which is now nobody's.
// The reader has to say so: "unsupported ggml type 1000" sends whoever reads
// it looking for a missing decoder, when the answer is to rebuild the file.
func TestRetiredTypeNumbersSayToRebuild(t *testing.T) {
	path := golemFixture(t, "T3G", nil)
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var was, now [4]byte
	binary.LittleEndian.PutUint32(was[:], golemT3G)
	binary.LittleEndian.PutUint32(now[:], 1000)
	i := bytes.Index(b, was[:])
	if i < 0 {
		t.Fatal("the fixture does not carry the type number it was written with")
	}
	copy(b[i:], now[:])
	retired := filepath.Join(t.TempDir(), "retired.golem")
	if err := os.WriteFile(retired, b, 0o644); err != nil {
		t.Fatal(err)
	}
	g, err := OpenGGUF(retired)
	if err == nil {
		g.Close()
		t.Fatal("a retired type number opened without complaint")
	}
	if !strings.Contains(err.Error(), "rebuilt with golemquant") {
		t.Fatalf("error is %v, which does not say to rebuild the file", err)
	}
}
