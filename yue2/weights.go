package yue2

import (
	"fmt"
	"math"
	"os"

	"github.com/ThiraSoft/golem/nn"
	"github.com/ThiraSoft/golem/tensors"
)

// defaultWeightsDir is the default location of YuE2 GGUF files when GOLEM_YUE2_DIR is not set.
const defaultWeightsDir = "/mnt/data/LLMs_models/Serveurperso/YuE2-GGUF"

func weightsDir() string {
	if d := os.Getenv("GOLEM_YUE2_DIR"); d != "" {
		return d
	}
	return defaultWeightsDir
}

// Matrix describes a two-dimensional weight matrix pointing into mapped GGUF bytes.
type Matrix struct {
	Quant      nn.Quant
	Data       []byte
	Rows, Cols int
}

// Block contains the projections and layer norms for one transformer block.
type Block struct {
	AttnNorm, FFNNorm, QNorm, KNorm []float32
	Q, K, V, O, Gate, Up, Down      Matrix
}

// Backbone holds the AR and NAR transformer blocks, input embeddings, logit head,
// and cross-modal adapters (VAE to LLM, LLM to VAE, and timestep embedder MLPs).
// All matrix weights are views over the mapped GGUF file; norms and biases are float32.
type Backbone struct {
	Cfg                                            Config
	AR, NAR                                        []Block
	Embed, Head                                    Matrix
	Norm                                           []float32
	Vae2LLM, LLM2Vae, Time0, Time1                 Matrix
	Vae2LLMBias, LLM2VaeBias, Time0Bias, Time1Bias []float32
	file                                           *tensors.GGUF
}

// Close unmaps the underlying GGUF file. Matrix views become invalid after Close.
func (b *Backbone) Close() error {
	if b.file != nil {
		return b.file.Close()
	}
	return nil
}

// VAEResUnit holds weights and snake parameters for one residual unit in a VAE upsampling block.
type VAEResUnit struct {
	Dilation                int
	Snake1Alpha, Snake1Beta []float32
	// Conv1 is kernel size 7, stored in PyTorch order [OC][IC][K] = [C][C][7].
	Conv1, Conv1Bias        []float32
	Snake2Alpha, Snake2Beta []float32
	// Conv2 is kernel size 1, stored in PyTorch order [OC][IC][K] = [C][C][1].
	Conv2, Conv2Bias []float32
}

// VAEBlock holds an upsampling stage with a snake activation, transposed convolution,
// and 3 residual units.
type VAEBlock struct {
	InChannels, OutChannels int
	Stride, Kernel          int
	// SnakeAlpha and SnakeBeta are exp(alpha) and 1/exp(beta), length InChannels.
	SnakeAlpha, SnakeBeta []float32
	// ConvT is the transposed convolution weight in PyTorch order
	// [IC][OC][K] = [InChannels][OutChannels][Kernel], where Kernel = 2 * Stride.
	ConvT, ConvTBias []float32
	ResUnits         []VAEResUnit
}

// VAEWeights holds the full weights for the Oobleck audio VAE decoder.
// All convolutions have their weight norm folded (w = g * v / ||v||) at load time.
// Weights are stored in standard PyTorch row-major order:
// - Conv1d: [OC][IC][K]
// - ConvTranspose1d: [IC][OC][K]
// Note: yue2.cpp transposes ConvT to [IC, K*OC] specifically for ggml mul_mat.
// Here we preserve plain PyTorch order; later Vulkan kernels layout weights to suit.
// Snake parameters are precomputed as exp(alpha) and 1/exp(beta).
type VAEWeights struct {
	// Conv1 is the initial convolution from latent channels (64) to 2048, kernel 7.
	// Stored in PyTorch order [OC][IC][K] = [2048][64][7].
	Conv1, Conv1Bias []float32

	// Blocks are the 6 upsampling stages with strides 6, 5, 4, 4, 2, 2.
	// Channels: 2048 -> 1024 -> 512 -> 256 -> 128 -> 64 -> 64.
	Blocks []VAEBlock

	// FinalSnakeAlpha and FinalSnakeBeta are exp(alpha) and 1/exp(beta), length 64.
	FinalSnakeAlpha, FinalSnakeBeta []float32

	// Conv2 is the final projection from 64 channels to stereo audio (2 channels), kernel 7.
	// Stored in PyTorch order [OC][IC][K] = [2][64][7]. No bias.
	Conv2 []float32
}

func matrix(g *tensors.GGUF, name string) (Matrix, error) {
	t, ok := g.Tensors[name]
	if !ok {
		return Matrix{}, fmt.Errorf("tensor %q is absent", name)
	}
	if len(t.Shape) != 2 {
		return Matrix{}, fmt.Errorf("tensor %q has %d dimensions", name, len(t.Shape))
	}
	q, ok := nn.QuantOf(t.DType)
	if !ok {
		return Matrix{}, fmt.Errorf("tensor %q is %s, not a recognized weight format", name, t.DType)
	}
	return Matrix{
		Data:  t.Raw,
		Quant: q,
		Rows:  t.Shape[1],
		Cols:  t.Shape[0],
	}, nil
}

func floats(g *tensors.GGUF, name string) ([]float32, error) {
	t, ok := g.Tensors[name]
	if !ok {
		return nil, fmt.Errorf("tensor %q is absent", name)
	}
	return t.F32()
}

// OpenBackbone maps a YuE2 backbone GGUF file and binds all AR and NAR weights.
// The file is memory-mapped without copying weight buffers; norms and biases
// are widened to float32.
func OpenBackbone(path string) (*Backbone, error) {
	g, err := tensors.OpenGGUF(path)
	if err != nil {
		return nil, err
	}

	cfg, err := readConfig(g)
	if err != nil {
		g.Close()
		return nil, err
	}

	b := &Backbone{
		Cfg:  cfg,
		AR:   make([]Block, cfg.Layers),
		NAR:  make([]Block, cfg.Layers),
		file: g,
	}

	if b.Embed, err = matrix(g, "model.embed_tokens.weight"); err != nil {
		g.Close()
		return nil, err
	}
	if b.Head, err = matrix(g, "lm_head.weight"); err != nil {
		g.Close()
		return nil, err
	}
	if b.Norm, err = floats(g, "model.norm.weight"); err != nil {
		g.Close()
		return nil, err
	}
	if b.Vae2LLM, err = matrix(g, "vae2llm.weight"); err != nil {
		g.Close()
		return nil, err
	}
	if b.Vae2LLMBias, err = floats(g, "vae2llm.bias"); err != nil {
		g.Close()
		return nil, err
	}
	if b.LLM2Vae, err = matrix(g, "llm2vae.weight"); err != nil {
		g.Close()
		return nil, err
	}
	if b.LLM2VaeBias, err = floats(g, "llm2vae.bias"); err != nil {
		g.Close()
		return nil, err
	}
	if b.Time0, err = matrix(g, "time_embedder.mlp.0.weight"); err != nil {
		g.Close()
		return nil, err
	}
	if b.Time0Bias, err = floats(g, "time_embedder.mlp.0.bias"); err != nil {
		g.Close()
		return nil, err
	}
	if b.Time1, err = matrix(g, "time_embedder.mlp.2.weight"); err != nil {
		g.Close()
		return nil, err
	}
	if b.Time1Bias, err = floats(g, "time_embedder.mlp.2.bias"); err != nil {
		g.Close()
		return nil, err
	}

	for i := 0; i < cfg.Layers; i++ {
		// AR block
		ar := &b.AR[i]
		if ar.AttnNorm, err = floats(g, fmt.Sprintf("model.layers.%d.input_layernorm.weight", i)); err != nil {
			g.Close()
			return nil, err
		}
		if ar.FFNNorm, err = floats(g, fmt.Sprintf("model.layers.%d.post_attention_layernorm.weight", i)); err != nil {
			g.Close()
			return nil, err
		}
		if ar.QNorm, err = floats(g, fmt.Sprintf("model.layers.%d.self_attn.q_norm.weight", i)); err != nil {
			g.Close()
			return nil, err
		}
		if ar.KNorm, err = floats(g, fmt.Sprintf("model.layers.%d.self_attn.k_norm.weight", i)); err != nil {
			g.Close()
			return nil, err
		}
		if ar.Q, err = matrix(g, fmt.Sprintf("model.layers.%d.self_attn.q_proj.weight", i)); err != nil {
			g.Close()
			return nil, err
		}
		if ar.K, err = matrix(g, fmt.Sprintf("model.layers.%d.self_attn.k_proj.weight", i)); err != nil {
			g.Close()
			return nil, err
		}
		if ar.V, err = matrix(g, fmt.Sprintf("model.layers.%d.self_attn.v_proj.weight", i)); err != nil {
			g.Close()
			return nil, err
		}
		if ar.O, err = matrix(g, fmt.Sprintf("model.layers.%d.self_attn.o_proj.weight", i)); err != nil {
			g.Close()
			return nil, err
		}
		if ar.Gate, err = matrix(g, fmt.Sprintf("model.layers.%d.mlp.gate_proj.weight", i)); err != nil {
			g.Close()
			return nil, err
		}
		if ar.Up, err = matrix(g, fmt.Sprintf("model.layers.%d.mlp.up_proj.weight", i)); err != nil {
			g.Close()
			return nil, err
		}
		if ar.Down, err = matrix(g, fmt.Sprintf("model.layers.%d.mlp.down_proj.weight", i)); err != nil {
			g.Close()
			return nil, err
		}

		// NAR block
		nar := &b.NAR[i]
		if nar.AttnNorm, err = floats(g, fmt.Sprintf("model.layers.%d.nar_input_layernorm.weight", i)); err != nil {
			g.Close()
			return nil, err
		}
		if nar.FFNNorm, err = floats(g, fmt.Sprintf("model.layers.%d.nar_pre_mlp_layernorm.weight", i)); err != nil {
			g.Close()
			return nil, err
		}
		if nar.QNorm, err = floats(g, fmt.Sprintf("model.layers.%d.nar_self_attn.q_norm.weight", i)); err != nil {
			g.Close()
			return nil, err
		}
		if nar.KNorm, err = floats(g, fmt.Sprintf("model.layers.%d.nar_self_attn.k_norm.weight", i)); err != nil {
			g.Close()
			return nil, err
		}
		if nar.Q, err = matrix(g, fmt.Sprintf("model.layers.%d.nar_self_attn.q_proj.weight", i)); err != nil {
			g.Close()
			return nil, err
		}
		if nar.K, err = matrix(g, fmt.Sprintf("model.layers.%d.nar_self_attn.k_proj.weight", i)); err != nil {
			g.Close()
			return nil, err
		}
		if nar.V, err = matrix(g, fmt.Sprintf("model.layers.%d.nar_self_attn.v_proj.weight", i)); err != nil {
			g.Close()
			return nil, err
		}
		if nar.O, err = matrix(g, fmt.Sprintf("model.layers.%d.nar_self_attn.o_proj.weight", i)); err != nil {
			g.Close()
			return nil, err
		}
		if nar.Gate, err = matrix(g, fmt.Sprintf("model.layers.%d.nar_mlp.gate_proj.weight", i)); err != nil {
			g.Close()
			return nil, err
		}
		if nar.Up, err = matrix(g, fmt.Sprintf("model.layers.%d.nar_mlp.up_proj.weight", i)); err != nil {
			g.Close()
			return nil, err
		}
		if nar.Down, err = matrix(g, fmt.Sprintf("model.layers.%d.nar_mlp.down_proj.weight", i)); err != nil {
			g.Close()
			return nil, err
		}
	}

	return b, nil
}

func foldWeightNorm(g, v []float32, dim0, fan int) []float32 {
	out := make([]float32, len(v))
	for d := 0; d < dim0; d++ {
		gv := g[d]
		base := d * fan
		var nsq float32
		for i := 0; i < fan; i++ {
			val := v[base+i]
			nsq += val * val
		}
		s := gv / (float32(math.Sqrt(float64(nsq))) + 1e-12)
		for i := 0; i < fan; i++ {
			out[base+i] = v[base+i] * s
		}
	}
	return out
}

func loadFoldedConv(g *tensors.GGUF, pfx string) ([]float32, error) {
	tg, err := g.Get(pfx + ".weight_g")
	if err != nil {
		return nil, err
	}
	tv, err := g.Get(pfx + ".weight_v")
	if err != nil {
		return nil, err
	}
	fg, err := tg.F32()
	if err != nil {
		return nil, err
	}
	fv, err := tv.F32()
	if err != nil {
		return nil, err
	}
	if len(tv.Shape) < 2 {
		return nil, fmt.Errorf("%s.weight_v: unexpected shape %v", pfx, tv.Shape)
	}
	dim0 := tv.Shape[len(tv.Shape)-1]
	fan := len(fv) / dim0
	if len(fg) != dim0 {
		return nil, fmt.Errorf("%s: weight_g length %d != dim0 %d", pfx, len(fg), dim0)
	}
	return foldWeightNorm(fg, fv, dim0, fan), nil
}

func loadSnake(g *tensors.GGUF, pfx string) ([]float32, []float32, error) {
	tA, err := g.Get(pfx + ".alpha")
	if err != nil {
		return nil, nil, err
	}
	tB, err := g.Get(pfx + ".beta")
	if err != nil {
		return nil, nil, err
	}
	fA, err := tA.F32()
	if err != nil {
		return nil, nil, err
	}
	fB, err := tB.F32()
	if err != nil {
		return nil, nil, err
	}
	if len(fA) != len(fB) {
		return nil, nil, fmt.Errorf("snake %s: alpha length %d != beta length %d", pfx, len(fA), len(fB))
	}
	expAlpha := make([]float32, len(fA))
	invExpBeta := make([]float32, len(fB))
	for i := range fA {
		expAlpha[i] = float32(math.Exp(float64(fA[i])))
		invExpBeta[i] = 1.0 / float32(math.Exp(float64(fB[i])))
	}
	return expAlpha, invExpBeta, nil
}

// OpenVAE reads the Oobleck VAE decoder weights from a GGUF file.
// All weight_g/weight_v pairs are folded via weight normalization into plain
// float32 convolution weights; snake parameters are converted to exp(alpha) and 1/exp(beta).
func OpenVAE(path string) (*VAEWeights, error) {
	g, err := tensors.OpenGGUF(path)
	if err != nil {
		return nil, err
	}
	defer g.Close()

	vw := &VAEWeights{
		Blocks: make([]VAEBlock, 6),
	}

	var errFold error
	if vw.Conv1, errFold = loadFoldedConv(g, "decoder.conv1"); errFold != nil {
		return nil, errFold
	}
	if vw.Conv1Bias, errFold = floats(g, "decoder.conv1.bias"); errFold != nil {
		return nil, errFold
	}

	strides := []int{6, 5, 4, 4, 2, 2}
	inCh := []int{2048, 1024, 512, 256, 128, 64}
	outCh := []int{1024, 512, 256, 128, 64, 64}
	dilations := []int{1, 3, 9}

	for i := 0; i < 6; i++ {
		blk := &vw.Blocks[i]
		blk.Stride = strides[i]
		blk.Kernel = strides[i] * 2
		blk.InChannels = inCh[i]
		blk.OutChannels = outCh[i]
		blk.ResUnits = make([]VAEResUnit, 3)

		blkPfx := fmt.Sprintf("decoder.block.%d", i)
		if blk.SnakeAlpha, blk.SnakeBeta, errFold = loadSnake(g, blkPfx+".snake1"); errFold != nil {
			return nil, errFold
		}
		if blk.ConvT, errFold = loadFoldedConv(g, blkPfx+".conv_t1"); errFold != nil {
			return nil, errFold
		}
		if blk.ConvTBias, errFold = floats(g, blkPfx+".conv_t1.bias"); errFold != nil {
			return nil, errFold
		}

		for r := 0; r < 3; r++ {
			ru := &blk.ResUnits[r]
			ru.Dilation = dilations[r]
			rp := fmt.Sprintf("%s.res_unit%d", blkPfx, r+1)

			if ru.Snake1Alpha, ru.Snake1Beta, errFold = loadSnake(g, rp+".snake1"); errFold != nil {
				return nil, errFold
			}
			if ru.Conv1, errFold = loadFoldedConv(g, rp+".conv1"); errFold != nil {
				return nil, errFold
			}
			if ru.Conv1Bias, errFold = floats(g, rp+".conv1.bias"); errFold != nil {
				return nil, errFold
			}
			if ru.Snake2Alpha, ru.Snake2Beta, errFold = loadSnake(g, rp+".snake2"); errFold != nil {
				return nil, errFold
			}
			if ru.Conv2, errFold = loadFoldedConv(g, rp+".conv2"); errFold != nil {
				return nil, errFold
			}
			if ru.Conv2Bias, errFold = floats(g, rp+".conv2.bias"); errFold != nil {
				return nil, errFold
			}
		}
	}

	if vw.FinalSnakeAlpha, vw.FinalSnakeBeta, errFold = loadSnake(g, "decoder.snake1"); errFold != nil {
		return nil, errFold
	}
	if vw.Conv2, errFold = loadFoldedConv(g, "decoder.conv2"); errFold != nil {
		return nil, errFold
	}

	return vw, nil
}
