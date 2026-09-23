package qwen35

import (
	"encoding/binary"
	"math"
	"math/rand"
	"os"
	"testing"

	"github.com/ThiraSoft/golem/nn"
	"github.com/ThiraSoft/golem/tensors"
)

func TestPrismGather(t *testing.T) {
	// Formula: hd = Inner/Rank, nk = SSMGroupCount, rep = Rank/nk
	// Gather[d + hd*(r + rep*k)] = d + hd*(k + nk*r)
	const (
		inner = 6144
		rank  = 48
		nk    = 16
		rep   = rank / nk    // 3
		hd    = inner / rank // 128
	)

	gather := make([]int32, inner)
	for k := 0; k < nk; k++ {
		for r := 0; r < rep; r++ {
			for d := 0; d < hd; d++ {
				dst := d + hd*(r+rep*k)
				src := d + hd*(k+nk*r)
				gather[dst] = int32(src)
			}
		}
	}

	// 1. Hand-computed positions:
	// Position A: k=0, r=0, d=0:
	//   dst = 0 + 128*(0 + 3*0) = 0
	//   src = 0 + 128*(0 + 16*0) = 0
	if gather[0] != 0 {
		t.Fatalf("gather[0] = %d, want 0", gather[0])
	}

	// Position B: k=1, r=0, d=5:
	//   dst = 5 + 128*(0 + 3*1) = 389
	//   src = 5 + 128*(1 + 16*0) = 133
	if gather[389] != 133 {
		t.Fatalf("gather[389] = %d, want 133", gather[389])
	}

	// Position C: k=5, r=2, d=42:
	//   dst = 42 + 128*(2 + 3*5) = 42 + 128*17 = 2218
	//   src = 42 + 128*(5 + 16*2) = 42 + 128*37 = 4778
	if gather[2218] != 4778 {
		t.Fatalf("gather[2218] = %d, want 4778", gather[2218])
	}

	// Verify that gather is a valid permutation of 0..6143
	seen := make([]bool, inner)
	for i, src := range gather {
		if src < 0 || int(src) >= inner {
			t.Fatalf("gather[%d] = %d out of range [0, %d)", i, src, inner)
		}
		if seen[src] {
			t.Fatalf("gather[%d] = %d is duplicated", i, src)
		}
		seen[src] = true
	}

	// 2. Synthetic 48x128 vector whose value encodes (head, d)
	vec := make([]float32, inner)
	for head := 0; head < rank; head++ {
		for d := 0; d < hd; d++ {
			vec[d+hd*head] = float32(head*1000 + d)
		}
	}

	gathered := make([]float32, inner)
	for i, src := range gather {
		gathered[i] = vec[src]
	}

	// Check the 3 hand-computed positions in gathered vector
	if gathered[0] != 0 {
		t.Fatalf("gathered[0] = %g, want 0", gathered[0])
	}
	if gathered[389] != 1005 {
		t.Fatalf("gathered[389] = %g, want 1005", gathered[389])
	}
	if gathered[2218] != 37042 {
		t.Fatalf("gathered[2218] = %g, want 37042", gathered[2218])
	}
}

func TestPrismBinding(t *testing.T) {
	const path = "/mnt/data/LLMs_models/prism-ml/Ternary-Bonsai-2-27B-gguf/Ternary-Bonsai-2-27B-PQ2_0.gguf"
	if _, err := os.Stat(path); err != nil {
		t.Skipf("%s is not present", path)
	}

	g, err := tensors.OpenGGUF(path)
	if err != nil {
		t.Fatal(err)
	}
	defer g.Close()

	cfg, err := LoadConfig(g, 1024)
	if err != nil {
		t.Fatal(err)
	}

	w, err := LoadWeights(g, cfg)
	if err != nil {
		t.Fatal(err)
	}

	if !w.Rotated {
		t.Fatal("expected Weights.Rotated to be true")
	}

	var preCount int
	checkMatrix := func(m *nn.Matrix, name string) {
		if m.Pre != nil && m.HadGroup == 1024 {
			preCount++
		}
	}

	for i := range w.Blocks {
		bw := &w.Blocks[i]
		bc := cfg.Blocks[i]
		prefix := "blk."
		if bc.Type == BlockFullAttn {
			checkMatrix(&bw.Q, prefix+"attn_q")
			checkMatrix(&bw.K, prefix+"attn_k")
			checkMatrix(&bw.V, prefix+"attn_v")
			checkMatrix(&bw.O, prefix+"attn_output")
		} else {
			checkMatrix(&bw.QKV, prefix+"attn_qkv")
			checkMatrix(&bw.AttnGate, prefix+"attn_gate")
			checkMatrix(&bw.SSMOut, prefix+"ssm_out")
			checkMatrix(&bw.SSMAlpha, prefix+"ssm_alpha")
			checkMatrix(&bw.SSMBeta, prefix+"ssm_beta")
		}
		checkMatrix(&bw.Gate, prefix+"ffn_gate")
		checkMatrix(&bw.Up, prefix+"ffn_up")
		checkMatrix(&bw.Down, prefix+"ffn_down")
	}
	checkMatrix(&w.OutputHead, "output.weight")

	if preCount != 401 {
		t.Fatalf("expected 401 matrices with Pre and HadGroup=1024, got %d", preCount)
	}

	if w.TokenEmbd.Pre == nil || w.TokenEmbd.HadGroup != 1024 {
		t.Fatal("expected TokenEmbd to have Pre and HadGroup=1024")
	}

	// Block 0 SSMOut has Gather of length 6144 that is a permutation
	gather := w.Blocks[0].SSMOut.Gather
	if len(gather) != 6144 {
		t.Fatalf("expected block 0 SSMOut.Gather length 6144, got %d", len(gather))
	}
	seen := make([]bool, 6144)
	for i, idx := range gather {
		if idx < 0 || int(idx) >= 6144 {
			t.Fatalf("gather[%d] = %d out of bounds [0, 6144)", i, idx)
		}
		if seen[idx] {
			t.Fatalf("gather[%d] = %d is a duplicate", i, idx)
		}
		seen[idx] = true
	}

	// Sign vector for 5120 starts with [-1, -1, -1, 1, -1, 1]
	wantSigns := []float32{-1, -1, -1, 1, -1, 1}
	if len(w.TokenEmbd.Pre) < len(wantSigns) {
		t.Fatalf("TokenEmbd Pre vector too short: %d", len(w.TokenEmbd.Pre))
	}
	for i, want := range wantSigns {
		if w.TokenEmbd.Pre[i] != want {
			t.Fatalf("sign[5120][%d] = %g, want %g", i, w.TokenEmbd.Pre[i], want)
		}
	}
}

func TestPrismAlgebra(t *testing.T) {
	// Algebra test that needs no file:
	// Random W (8x2048 float32), random signs, rotated weights W' = W D H
	// computed by applying PrepareGolem to each row's transpose appropriately:
	// y = W x = (W D H)(H D x).
	const (
		rows = 8
		cols = 2048
	)
	rng := rand.New(rand.NewSource(9999))

	W := make([][]float32, rows)
	for r := 0; r < rows; r++ {
		W[r] = make([]float32, cols)
		for c := 0; c < cols; c++ {
			W[r][c] = rng.Float32()*2 - 1
		}
	}

	signs := make([]float32, cols)
	for c := 0; c < cols; c++ {
		if rng.Intn(2) == 0 {
			signs[c] = -1.0
		} else {
			signs[c] = 1.0
		}
	}

	// Compute W' = W D H: for each row, PrepareGolem(rowCopy, signs, 1024)
	wPrimeData := make([]byte, rows*cols*4)
	for r := 0; r < rows; r++ {
		rowCopy := make([]float32, cols)
		copy(rowCopy, W[r])
		nn.PrepareGolem(rowCopy, signs, 1024)
		for c := 0; c < cols; c++ {
			bits := math.Float32bits(rowCopy[c])
			binary.LittleEndian.PutUint32(wPrimeData[(r*cols+c)*4:], bits)
		}
	}

	mRotated := nn.Matrix{
		Data:     wPrimeData,
		Quant:    nn.F32,
		Rows:     rows,
		Cols:     cols,
		Pre:      signs,
		HadGroup: 1024,
	}

	x := make([]float32, cols)
	for c := 0; c < cols; c++ {
		x[c] = rng.Float32()*2 - 1
	}

	// Ground truth y = W x
	wantY := make([]float32, rows)
	for r := 0; r < rows; r++ {
		wantY[r] = nn.DotF32(W[r], x)
	}

	batch := &nn.Batch{
		Size:  1,
		Width: cols,
		F:     [][]float32{x},
	}
	gotY := make([]float32, rows)
	mRotated.MatVec(batch, gotY)

	for r := 0; r < rows; r++ {
		diff := math.Abs(float64(gotY[r] - wantY[r]))
		rel := diff / math.Abs(float64(wantY[r]))
		if rel > 1e-4 {
			t.Fatalf("row %d: got %g, want %g (rel diff %g > 1e-4)", r, gotY[r], wantY[r], rel)
		}
	}

	// Same test with a Gather permutation:
	perm := rng.Perm(cols)
	gather := make([]int32, cols)
	for i, p := range perm {
		gather[i] = int32(p)
	}

	// With Gather, the activation handed to the product is x'[i] = x[gather[i]].
	// We want W'' H D x' = W x.
	// Letting z = P x where z[i] = x[gather[i]], we have W'' H D z = W x.
	// w_perm[i] = W[r][gather[i]], so that w_perm . z = W[r] . x.
	// Then w'' = PrepareGolem(w_perm, signs, 1024).
	wGatherData := make([]byte, rows*cols*4)
	for r := 0; r < rows; r++ {
		wPerm := make([]float32, cols)
		for i := 0; i < cols; i++ {
			wPerm[i] = W[r][gather[i]]
		}
		nn.PrepareGolem(wPerm, signs, 1024)
		for c := 0; c < cols; c++ {
			bits := math.Float32bits(wPerm[c])
			binary.LittleEndian.PutUint32(wGatherData[(r*cols+c)*4:], bits)
		}
	}

	mGather := nn.Matrix{
		Data:     wGatherData,
		Quant:    nn.F32,
		Rows:     rows,
		Cols:     cols,
		Pre:      signs,
		HadGroup: 1024,
		Gather:   gather,
	}

	gotYGather := make([]float32, rows)
	batchGather := &nn.Batch{
		Size:  1,
		Width: cols,
		F:     [][]float32{x},
	}
	mGather.MatVec(batchGather, gotYGather)

	for r := 0; r < rows; r++ {
		diff := math.Abs(float64(gotYGather[r] - wantY[r]))
		rel := diff / math.Abs(float64(wantY[r]))
		if rel > 1e-4 {
			t.Fatalf("gather row %d: got %g, want %g (rel diff %g > 1e-4)", r, gotYGather[r], wantY[r], rel)
		}
	}
}
