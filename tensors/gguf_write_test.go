package tensors

import (
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

// A file written and read back has to be the file that went in. The converter
// carries a checkpoint's whole metadata across — vocabulary, merges, chat
// template — and a value that changes type on the way through is a model that
// tokenizes differently for no visible reason.
func TestGGUFRoundTrip(t *testing.T) {
	meta := map[string]any{
		"general.architecture":     "qwen3",
		"general.alignment":        uint32(32),
		"qwen3.block_count":        uint32(36),
		"qwen3.rope.freq_base":     float32(1000000),
		"qwen3.attention.eps":      float32(1e-6),
		"tokenizer.ggml.tokens":    []any{"a", "b", "cc"},
		"tokenizer.ggml.bos_id":    int32(11),
		"tokenizer.ggml.add_bos":   true,
		"tokenizer.ggml.scores":    []any{float32(1), float32(-2)},
		"golem.format":             GolemFormat,
		"golem.hadamard_group":     uint32(128),
		"golem.scale_block":        uint32(64),
		"golem.trellis.seq":        uint32(128),
		"golem.trellis.bits":       uint32(4),
		"golem.trellis.state":      uint32(12),
		"general.quantization_ver": uint64(2),
	}
	tensors := []OutTensor{
		{Name: "output_norm.weight", Shape: []int{4}, DType: "F32",
			Data: []byte{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}},
		{Name: "blk.0.attn_q.weight", Shape: []int{128, 2}, DType: "T4G",
			Data: make([]byte, 2*67)},
	}
	for i := range tensors[1].Data {
		tensors[1].Data[i] = byte(i * 7)
	}

	path := filepath.Join(t.TempDir(), "round.golem")
	if err := WriteGGUF(path, meta, tensors); err != nil {
		t.Fatal(err)
	}
	g, err := OpenGGUF(path)
	if err != nil {
		t.Fatal(err)
	}
	defer g.Close()

	if !reflect.DeepEqual(g.Meta, meta) {
		for k, want := range meta {
			if got := g.Meta[k]; !reflect.DeepEqual(got, want) {
				t.Errorf("%s: read %#v, wrote %#v", k, got, want)
			}
		}
		for k := range g.Meta {
			if _, ok := meta[k]; !ok {
				t.Errorf("%s appeared from nowhere", k)
			}
		}
	}
	for _, want := range tensors {
		got, err := g.Get(want.Name)
		if err != nil {
			t.Fatal(err)
		}
		if got.DType != want.DType {
			t.Errorf("%s is %s, wrote %s", want.Name, got.DType, want.DType)
		}
		if !reflect.DeepEqual(got.Shape, want.Shape) {
			t.Errorf("%s has shape %v, wrote %v", want.Name, got.Shape, want.Shape)
		}
		if !reflect.DeepEqual([]byte(got.Raw), want.Data) {
			t.Errorf("%s: %d bytes back, wrote %d", want.Name, len(got.Raw), len(want.Data))
		}
	}
	if _, err := os.Stat(path); err != nil {
		t.Fatal(err)
	}
}

// TestGGUFRefusesLatticeEra checks the guard that catches the one tensor type a
// D4G file and a T3G file cannot be told apart by size or type number alone:
// both are 26 bytes per 64 weights. A file still carrying a golem.d4.* key must
// fail to open rather than decode through the trellis as nonsense.
func TestGGUFRefusesLatticeEra(t *testing.T) {
	meta := map[string]any{
		"general.architecture":    "qwen3",
		"general.alignment":       uint32(32),
		"golem.d4.hadamard_group": uint32(128),
	}
	tensors := []OutTensor{
		{Name: "blk.0.attn_q.weight", Shape: []int{128, 2}, DType: "T3G",
			Data: make([]byte, 2*52)},
	}
	path := filepath.Join(t.TempDir(), "lattice-era.golem")
	if err := WriteGGUF(path, meta, tensors); err != nil {
		t.Fatal(err)
	}
	if _, err := OpenGGUF(path); err == nil {
		t.Fatal("a golem.d4.* key opened without complaint")
	}
}

// TestGGUFRefusesLatticeEraEvenWithTrellisBits covers the transitional files
// on disk that carry both a golem.d4.* key and golem.trellis.bits — written
// after the trellis existed but before this branch renamed the metadata keys.
// The d4 key alone has to refuse the file: trellis.bits being present too must
// not be read as permission.
func TestGGUFRefusesLatticeEraEvenWithTrellisBits(t *testing.T) {
	meta := map[string]any{
		"general.architecture":    "qwen3",
		"general.alignment":       uint32(32),
		"golem.d4.hadamard_group": uint32(128),
		"golem.trellis.bits":      uint32(4),
	}
	tensors := []OutTensor{
		{Name: "blk.0.attn_q.weight", Shape: []int{128, 2}, DType: "T3G",
			Data: make([]byte, 2*52)},
	}
	path := filepath.Join(t.TempDir(), "transitional.golem")
	if err := WriteGGUF(path, meta, tensors); err != nil {
		t.Fatal(err)
	}
	if _, err := OpenGGUF(path); err == nil {
		t.Fatal("a golem.d4.* key opened without complaint even with golem.trellis.bits present")
	}
}

// TestGGUFRefusesTrellisWithoutBits checks the other half of the same guard:
// trellis-tier tensors with no golem.trellis.bits to say which tier wrote
// them. The file is otherwise well formed and declares golem.format, so this
// is layer three refusing on its own.
func TestGGUFRefusesTrellisWithoutBits(t *testing.T) {
	meta := map[string]any{
		"general.architecture": "qwen3",
		"general.alignment":    uint32(32),
		"golem.format":         GolemFormat,
		"golem.trellis.seq":    uint32(128),
		"golem.trellis.state":  uint32(12),
	}
	tensors := []OutTensor{
		{Name: "blk.0.attn_q.weight", Shape: []int{128, 2}, DType: "T3G",
			Data: make([]byte, 2*52)},
	}
	path := filepath.Join(t.TempDir(), "no-bits.golem")
	if err := WriteGGUF(path, meta, tensors); err != nil {
		t.Fatal(err)
	}
	if _, err := OpenGGUF(path); err == nil {
		t.Fatal("a trellis tensor with no golem.trellis.bits opened without complaint")
	}
}
