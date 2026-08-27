package qwen35

import (
	"fmt"

	"github.com/ThiraSoft/golem/sample"
	"github.com/ThiraSoft/golem/tensors"
)

type BlockType uint8

const (
	BlockSSM BlockType = iota
	BlockFullAttn
)

type BlockConfig struct {
	Index int
	Type  BlockType

	// Full Attention parameters (blocks where Type == BlockFullAttn)
	Heads    int // 24 heads for 27B
	KVHeads  int // 4 heads for 27B
	HeadDim  int // 256 for 27B
	RoPEBase float64
	RoPEDims int // 64 for 27B
	OutDim   int // 6144 for 27B

	// SSM / Mamba parameters (blocks where Type == BlockSSM)
	SSMConvKernel   int // 4
	SSMInnerSize    int // 6144
	SSMStateSize    int // 128
	SSMTimeStepRank int // 48
	SSMGroupCount   int // 16

	// Shared FFN
	FFN int // 17408
}

type Config struct {
	Arch       string
	Dim        int
	Vocab      int
	Eps        float32
	MaxContext int
	Blocks     []BlockConfig
	Sampling   sample.Params

	// NextN is how many of the trailing blocks are multi-token-prediction
	// blocks rather than part of the trunk. The checkpoint counts them in
	// block_count, and running one of them in the main pass would be a
	// sixty-fifth layer the model was never trained to have.
	NextN int
}

func perBlock(values []uint32, i int) int {
	if len(values) == 1 {
		return int(values[0])
	}
	return int(values[i])
}

func LoadConfig(g *tensors.GGUF, maxContext int) (*Config, error) {
	arch, err := g.String("general.architecture")
	if err != nil {
		return nil, err
	}
	if arch != "qwen35" && arch != "qwen3.8" {
		return nil, fmt.Errorf("architecture %q is not qwen35/qwen3.8", arch)
	}
	key := func(suffix string) string { return arch + "." + suffix }

	blockCount, err := g.Uint32(key("block_count"))
	if err != nil {
		return nil, err
	}
	dim, err := g.Uint32(key("embedding_length"))
	if err != nil {
		return nil, err
	}
	eps, err := g.Float32(key("attention.layer_norm_rms_epsilon"))
	if err != nil {
		return nil, err
	}
	heads, err := g.Uint32Slice(key("attention.head_count"))
	if err != nil {
		return nil, err
	}
	kvHeads, err := g.Uint32Slice(key("attention.head_count_kv"))
	if err != nil {
		return nil, err
	}
	ffn, err := g.Uint32Slice(key("feed_forward_length"))
	if err != nil {
		return nil, err
	}
	headDim, err := g.Uint32(key("attention.key_length"))
	if err != nil {
		return nil, err
	}
	ropeBase, err := g.Float32(key("rope.freq_base"))
	if err != nil {
		return nil, err
	}
	ropeDims := int(headDim)
	if v, err := g.Uint32(key("rope.dimension_count")); err == nil {
		ropeDims = int(v)
	}

	convKernel, _ := g.Uint32(key("ssm.conv_kernel"))
	stateSize, _ := g.Uint32(key("ssm.state_size"))
	groupCount, _ := g.Uint32(key("ssm.group_count"))
	timeStepRank, _ := g.Uint32(key("ssm.time_step_rank"))
	innerSize, _ := g.Uint32(key("ssm.inner_size"))

	nextN, _ := g.Uint32(key("nextn_predict_layers"))

	cfg := &Config{
		Arch:       arch,
		NextN:      int(nextN),
		Sampling:   loadSampling(g),
		Dim:        int(dim),
		Eps:        eps,
		MaxContext: maxContext,
	}

	embd, ok := g.Tensors["token_embd.weight"]
	if !ok || len(embd.Shape) != 2 {
		return nil, fmt.Errorf("token_embd.weight is absent or not two-dimensional")
	}
	cfg.Vocab = embd.Shape[1]

	cfg.Blocks = make([]BlockConfig, blockCount)
	for i := range cfg.Blocks {
		_, hasSSM := g.Tensors[fmt.Sprintf("blk.%d.ssm_out.weight", i)]
		bType := BlockSSM
		if !hasSSM {
			bType = BlockFullAttn
		}

		cfg.Blocks[i] = BlockConfig{
			Index:           i,
			Type:            bType,
			Heads:           perBlock(heads, i),
			KVHeads:         perBlock(kvHeads, i),
			FFN:             perBlock(ffn, i),
			HeadDim:         int(headDim),
			RoPEBase:        float64(ropeBase),
			RoPEDims:        ropeDims,
			OutDim:          int(innerSize),
			SSMConvKernel:   int(convKernel),
			SSMInnerSize:    int(innerSize),
			SSMStateSize:    int(stateSize),
			SSMTimeStepRank: int(timeStepRank),
			SSMGroupCount:   int(groupCount),
		}
	}
	return cfg, nil
}

func loadSampling(g *tensors.GGUF) sample.Params {
	p := sample.Defaults()
	if v, err := g.Float32("general.sampling.temp"); err == nil {
		p.Temperature = v
	}
	if v, err := g.Uint32("general.sampling.top_k"); err == nil {
		p.TopK = int(v)
	}
	if v, err := g.Float32("general.sampling.top_p"); err == nil {
		p.TopP = v
	}
	return p
}
