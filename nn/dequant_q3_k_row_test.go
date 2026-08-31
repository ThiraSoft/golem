package nn

import (
	"encoding/binary"
	"math"
	"os"
	"testing"

	"github.com/ThiraSoft/golem/tensors"
)

// TestDequantizeQ3_KRowCorrelatesWithBF16 is the probe the hand-built block in
// dequant_q3_k_test.go cannot be: that fixture pins two weights and, by its
// own report, cannot fail on a wrong traversal order or a wrong scale
// unpacking. Q3_K used to be an internal read-only path; this branch now
// publishes numbers about llama.cpp's Q3_K on the landing page, so a wrong
// reader here would be a wrong public claim, not just a wrong local one.
//
// A correct reader dequantizes a real row into something close to the same
// row of the bf16 original — a traversal or scale-unpacking bug does not
// produce a plausible-looking wrong answer, it destroys the correlation.
func TestDequantizeQ3_KRowCorrelatesWithBF16(t *testing.T) {
	const (
		bf16Path = "/mnt/data/LLMs_models/unsloth/Qwen3-4B-GGUF/Qwen3-4B-BF16.gguf"
		q3kPath  = "/mnt/data/LLMs_models/unsloth/Qwen3-4B-GGUF/Qwen3-4B-Q3_K_M.gguf"
	)
	if _, err := os.Stat(bf16Path); err != nil {
		t.Skipf("%s is not there", bf16Path)
	}
	if _, err := os.Stat(q3kPath); err != nil {
		t.Skipf("%s is not there", q3kPath)
	}

	ref, err := tensors.OpenGGUF(bf16Path)
	if err != nil {
		t.Fatal(err)
	}
	defer ref.Close()
	quant, err := tensors.OpenGGUF(q3kPath)
	if err != nil {
		t.Fatal(err)
	}
	defer quant.Close()

	// The M mix keeps the attention projections at three bits and spends the
	// FFN's wider matrices at higher rates, so this is one of the tensors
	// still actually stored as Q3_K.
	const name = "blk.0.attn_q.weight"
	rt, err := ref.Get(name)
	if err != nil {
		t.Fatalf("bf16 reference has no %s: %v", name, err)
	}
	qt, err := quant.Get(name)
	if err != nil {
		t.Fatalf("Q3_K_M file has no %s: %v", name, err)
	}
	if qt.DType != "Q3_K" {
		t.Skipf("%s is stored as %s in the Q3_K_M file, not Q3_K", name, qt.DType)
	}
	cols := rt.Shape[0]
	if qt.Shape[0] != cols || cols%SuperBlock != 0 {
		t.Fatalf("shape mismatch: bf16 row is %d wide, Q3_K row is %d, SuperBlock is %d", cols, qt.Shape[0], SuperBlock)
	}

	// Row 0 of each.
	refRow := rt.Raw[:cols*2]
	quantRowBytes := qt.Raw[:(cols/SuperBlock)*q3_kBlockBytes]

	want := make([]float32, cols)
	for i := 0; i < cols; i++ {
		bits := uint32(binary.LittleEndian.Uint16(refRow[i*2:])) << 16
		want[i] = math.Float32frombits(bits)
	}
	got := make([]float32, cols)
	DequantizeQ3_K(quantRowBytes, cols, got)

	corr := pearson(want, got)
	if corr < 0.95 {
		t.Fatalf("Q3_K row correlates %.4f with the bf16 reference, want near 1", corr)
	}
}

func pearson(a, b []float32) float64 {
	n := len(a)
	var sa, sb float64
	for i := 0; i < n; i++ {
		sa += float64(a[i])
		sb += float64(b[i])
	}
	ma, mb := sa/float64(n), sb/float64(n)
	var num, da, db float64
	for i := 0; i < n; i++ {
		x := float64(a[i]) - ma
		y := float64(b[i]) - mb
		num += x * y
		da += x * x
		db += y * y
	}
	if da == 0 || db == 0 {
		return 0
	}
	return num / math.Sqrt(da*db)
}
