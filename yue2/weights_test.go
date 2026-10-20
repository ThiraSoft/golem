package yue2

import (
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/ThiraSoft/golem/nn"
	"github.com/ThiraSoft/golem/tensors"
)

func TestOpenBackboneQ8(t *testing.T) {
	path := filepath.Join(weightsDir(), "YuE2-3B-Q8_0.gguf")
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no weights file (%v): skipping backbone test", err)
	}

	b, err := OpenBackbone(path)
	if err != nil {
		t.Fatalf("OpenBackbone(%s): %v", path, err)
	}
	defer b.Close()

	if len(b.AR) != 28 {
		t.Errorf("len(AR) = %d, want 28", len(b.AR))
	}
	if len(b.NAR) != 28 {
		t.Errorf("len(NAR) = %d, want 28", len(b.NAR))
	}

	// AR[0].Q is Q8_0 2048x2048
	q := b.AR[0].Q
	if q.Quant != nn.Q8_0 {
		t.Errorf("AR[0].Q.Quant = %v, want Q8_0", q.Quant)
	}
	if q.Rows != 2048 || q.Cols != 2048 {
		t.Errorf("AR[0].Q shape = %dx%d, want 2048x2048", q.Rows, q.Cols)
	}

	// AR[0].K has 1024 rows (8 KV heads * 128 head dim)
	k := b.AR[0].K
	if k.Rows != 1024 {
		t.Errorf("AR[0].K.Rows = %d, want 1024", k.Rows)
	}
	if k.Cols != 2048 {
		t.Errorf("AR[0].K.Cols = %d, want 2048", k.Cols)
	}

	// Head rows and config
	if b.Head.Rows != 184704 {
		t.Errorf("Head.Rows = %d, want 184704", b.Head.Rows)
	}
	if b.Cfg.Vocab != 184704 {
		t.Errorf("Cfg.Vocab = %d, want 184704", b.Cfg.Vocab)
	}
	if b.Cfg.Context != 24576 {
		t.Errorf("Cfg.Context = %d, want 24576", b.Cfg.Context)
	}
	if b.Cfg.Dim != 2048 {
		t.Errorf("Cfg.Dim = %d, want 2048", b.Cfg.Dim)
	}
	if b.Cfg.FFN != 6144 {
		t.Errorf("Cfg.FFN = %d, want 6144", b.Cfg.FFN)
	}
	if b.Cfg.Heads != 16 {
		t.Errorf("Cfg.Heads = %d, want 16", b.Cfg.Heads)
	}
	if b.Cfg.KVHeads != 8 {
		t.Errorf("Cfg.KVHeads = %d, want 8", b.Cfg.KVHeads)
	}
	if b.Cfg.HeadDim != 128 {
		t.Errorf("Cfg.HeadDim = %d, want 128", b.Cfg.HeadDim)
	}
	if b.Cfg.Layers != 28 {
		t.Errorf("Cfg.Layers = %d, want 28", b.Cfg.Layers)
	}
	if b.Cfg.LatentDim != 64 {
		t.Errorf("Cfg.LatentDim = %d, want 64", b.Cfg.LatentDim)
	}
	if b.Cfg.RoPEBase != 1000000.0 {
		t.Errorf("Cfg.RoPEBase = %g, want 1000000.0", b.Cfg.RoPEBase)
	}
	if b.Cfg.Eps != 1e-6 {
		t.Errorf("Cfg.Eps = %g, want 1e-6", b.Cfg.Eps)
	}
	if b.Cfg.TimestepShift != 1.0 {
		t.Errorf("Cfg.TimestepShift = %g, want 1.0", b.Cfg.TimestepShift)
	}

	// Adapters and norms
	if len(b.Norm) != 2048 {
		t.Errorf("len(Norm) = %d, want 2048", len(b.Norm))
	}
	if b.Vae2LLM.Rows != 2048 || b.Vae2LLM.Cols != 64 {
		t.Errorf("Vae2LLM shape = %dx%d, want 2048x64", b.Vae2LLM.Rows, b.Vae2LLM.Cols)
	}
	if len(b.Vae2LLMBias) != 2048 {
		t.Errorf("len(Vae2LLMBias) = %d, want 2048", len(b.Vae2LLMBias))
	}
	if b.LLM2Vae.Rows != 64 || b.LLM2Vae.Cols != 2048 {
		t.Errorf("LLM2Vae shape = %dx%d, want 64x2048", b.LLM2Vae.Rows, b.LLM2Vae.Cols)
	}
	if len(b.LLM2VaeBias) != 64 {
		t.Errorf("len(LLM2VaeBias) = %d, want 64", len(b.LLM2VaeBias))
	}
	if b.Time0.Rows != 2048 || b.Time0.Cols != 256 {
		t.Errorf("Time0 shape = %dx%d, want 2048x256", b.Time0.Rows, b.Time0.Cols)
	}
	if len(b.Time0Bias) != 2048 {
		t.Errorf("len(Time0Bias) = %d, want 2048", len(b.Time0Bias))
	}
	if b.Time1.Rows != 2048 || b.Time1.Cols != 2048 {
		t.Errorf("Time1 shape = %dx%d, want 2048x2048", b.Time1.Rows, b.Time1.Cols)
	}
	if len(b.Time1Bias) != 2048 {
		t.Errorf("len(Time1Bias) = %d, want 2048", len(b.Time1Bias))
	}
}

func TestOpenBackboneBF16(t *testing.T) {
	path := filepath.Join(weightsDir(), "YuE2-3B-BF16.gguf")
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no weights file (%v): skipping backbone BF16 test", err)
	}

	b, err := OpenBackbone(path)
	if err != nil {
		t.Fatalf("OpenBackbone(%s): %v", path, err)
	}
	defer b.Close()

	if len(b.AR) != 28 {
		t.Errorf("len(AR) = %d, want 28", len(b.AR))
	}
	if len(b.NAR) != 28 {
		t.Errorf("len(NAR) = %d, want 28", len(b.NAR))
	}

	// AR[0].Q is BF16 2048x2048
	q := b.AR[0].Q
	if q.Quant != nn.BF16 {
		t.Errorf("AR[0].Q.Quant = %v, want BF16", q.Quant)
	}
	if q.Rows != 2048 || q.Cols != 2048 {
		t.Errorf("AR[0].Q shape = %dx%d, want 2048x2048", q.Rows, q.Cols)
	}

	// AR[0].K has 1024 rows
	k := b.AR[0].K
	if k.Rows != 1024 {
		t.Errorf("AR[0].K.Rows = %d, want 1024", k.Rows)
	}
	if k.Cols != 2048 {
		t.Errorf("AR[0].K.Cols = %d, want 2048", k.Cols)
	}

	if b.Head.Rows != 184704 {
		t.Errorf("Head.Rows = %d, want 184704", b.Head.Rows)
	}
	if b.Cfg.Vocab != 184704 {
		t.Errorf("Cfg.Vocab = %d, want 184704", b.Cfg.Vocab)
	}
	if b.Cfg.Context != 24576 {
		t.Errorf("Cfg.Context = %d, want 24576", b.Cfg.Context)
	}

	// Verify norms and biases come back widened to float32
	if len(b.Norm) != 2048 {
		t.Errorf("len(Norm) = %d, want 2048", len(b.Norm))
	}
	if len(b.AR[0].AttnNorm) != 2048 {
		t.Errorf("len(AR[0].AttnNorm) = %d, want 2048", len(b.AR[0].AttnNorm))
	}
	if len(b.AR[0].QNorm) != 128 {
		t.Errorf("len(AR[0].QNorm) = %d, want 128", len(b.AR[0].QNorm))
	}
}

func TestOpenVAE(t *testing.T) {
	path := filepath.Join(weightsDir(), "YuE2-Vae-F32.gguf")
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no weights file (%v): skipping VAE test", err)
	}

	vae, err := OpenVAE(path)
	if err != nil {
		t.Fatalf("OpenVAE(%s): %v", path, err)
	}

	// Verify 6 blocks with strides 6, 5, 4, 4, 2, 2
	wantStrides := []int{6, 5, 4, 4, 2, 2}
	if len(vae.Blocks) != len(wantStrides) {
		t.Fatalf("len(Blocks) = %d, want %d", len(vae.Blocks), len(wantStrides))
	}
	for i, want := range wantStrides {
		if got := vae.Blocks[i].Stride; got != want {
			t.Errorf("Blocks[%d].Stride = %d, want %d", i, got, want)
		}
	}

	// Verify channels: 2048 -> 1024 -> 512 -> 256 -> 128 -> 64 -> 64
	inChannels := []int{2048, 1024, 512, 256, 128, 64}
	outChannels := []int{1024, 512, 256, 128, 64, 64}
	for i := range vae.Blocks {
		if vae.Blocks[i].InChannels != inChannels[i] {
			t.Errorf("Blocks[%d].InChannels = %d, want %d", i, vae.Blocks[i].InChannels, inChannels[i])
		}
		if vae.Blocks[i].OutChannels != outChannels[i] {
			t.Errorf("Blocks[%d].OutChannels = %d, want %d", i, vae.Blocks[i].OutChannels, outChannels[i])
		}
	}

	// Conv1 shape: [2048][64][7] = 917504 floats
	wantConv1Len := 2048 * 64 * 7
	if len(vae.Conv1) != wantConv1Len {
		t.Errorf("len(Conv1) = %d, want %d", len(vae.Conv1), wantConv1Len)
	}
	if len(vae.Conv1Bias) != 2048 {
		t.Errorf("len(Conv1Bias) = %d, want 2048", len(vae.Conv1Bias))
	}

	// Verify one folded weight checked against a hand computation from weight_g/weight_v
	rawGGUF, err := tensors.OpenGGUF(path)
	if err != nil {
		t.Fatalf("OpenGGUF(%s): %v", path, err)
	}
	defer rawGGUF.Close()

	tg, err := rawGGUF.Get("decoder.conv1.weight_g")
	if err != nil {
		t.Fatalf("get weight_g: %v", err)
	}
	tv, err := rawGGUF.Get("decoder.conv1.weight_v")
	if err != nil {
		t.Fatalf("get weight_v: %v", err)
	}
	fg, err := tg.F32()
	if err != nil {
		t.Fatalf("weight_g F32: %v", err)
	}
	fv, err := tv.F32()
	if err != nil {
		t.Fatalf("weight_v F32: %v", err)
	}

	// Hand computation for d=0 (first output channel of conv1)
	dim0 := tv.Shape[2]              // 2048
	fan := tv.Shape[0] * tv.Shape[1] // 7 * 64 = 448
	if dim0 != 2048 || fan != 448 {
		t.Fatalf("unexpected conv1 shape: dim0=%d, fan=%d", dim0, fan)
	}

	gv := fg[0]
	var nsq float32
	for i := 0; i < fan; i++ {
		val := fv[i]
		nsq += val * val
	}
	s := gv / (float32(math.Sqrt(float64(nsq))) + 1e-12)
	wantWeight0 := fv[0] * s

	gotWeight0 := vae.Conv1[0]
	if math.Abs(float64(gotWeight0-wantWeight0)) > 1e-6 {
		t.Errorf("folded weight conv1[0] = %g, want %g (diff %g)", gotWeight0, wantWeight0, gotWeight0-wantWeight0)
	}

	// Also check snake parameters exp(alpha) and 1/exp(beta)
	tSnakeA, err := rawGGUF.Get("decoder.block.0.snake1.alpha")
	if err != nil {
		t.Fatalf("get snake alpha: %v", err)
	}
	tSnakeB, err := rawGGUF.Get("decoder.block.0.snake1.beta")
	if err != nil {
		t.Fatalf("get snake beta: %v", err)
	}
	fSnakeA, _ := tSnakeA.F32()
	fSnakeB, _ := tSnakeB.F32()

	wantExpA0 := float32(math.Exp(float64(fSnakeA[0])))
	wantInvExpB0 := 1.0 / float32(math.Exp(float64(fSnakeB[0])))

	gotExpA0 := vae.Blocks[0].SnakeAlpha[0]
	gotInvExpB0 := vae.Blocks[0].SnakeBeta[0]

	if math.Abs(float64(gotExpA0-wantExpA0)) > 1e-6 {
		t.Errorf("SnakeAlpha[0] = %g, want %g", gotExpA0, wantExpA0)
	}
	if math.Abs(float64(gotInvExpB0-wantInvExpB0)) > 1e-6 {
		t.Errorf("SnakeBeta[0] = %g, want %g", gotInvExpB0, wantInvExpB0)
	}
}
