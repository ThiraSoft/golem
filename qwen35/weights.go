package qwen35

import (
	"fmt"

	"github.com/ThiraSoft/golem/nn"
	"github.com/ThiraSoft/golem/tensors"
)

type BlockWeights struct {
	AttnNorm []float32
	FFNNorm  []float32

	// Full Attention weights
	QNorm []float32
	KNorm []float32
	Q     nn.Matrix // [5120, 12288] (fused Q [6144] and Gate [6144])
	K     nn.Matrix // [5120, 1024]
	V     nn.Matrix // [5120, 1024]
	O     nn.Matrix // [6144, 5120]

	// Linear Attention (Gated Delta Net) weights
	QKV       nn.Matrix // [5120, 10240]
	AttnGate  nn.Matrix // [5120, 6144] (Gate Z)
	Conv1D    []float32 // [4, 10240]
	SSMA      []float32 // [48]
	SSMAlpha  nn.Matrix // [5120, 48]
	SSMBeta   nn.Matrix // [5120, 48]
	SSMDtBias []float32 // [48]
	SSMNorm   []float32 // [128]
	SSMOut    nn.Matrix // [6144, 5120] (Q5_K or Q4_K)

	// FFN weights (shared by both block types)
	Gate nn.Matrix // [5120, 17408]
	Up   nn.Matrix // [5120, 17408]
	Down nn.Matrix // [17408, 5120]
}

type MTPWeights struct {
	ENorm          []float32
	HNorm          []float32
	EHProj         nn.Matrix // [10240, 5120] Q8_0
	SharedHeadNorm []float32
	HasMTP         bool
}

type Weights struct {
	TokenEmbd  nn.Matrix
	OutputNorm []float32
	OutputHead nn.Matrix
	Blocks     []BlockWeights
	MTP        MTPWeights
}

func LoadWeights(g *tensors.GGUF, cfg *Config) (*Weights, error) {
	w := &Weights{
		Blocks: make([]BlockWeights, len(cfg.Blocks)),
	}

	embd, ok := g.Tensors["token_embd.weight"]
	if !ok {
		return nil, fmt.Errorf("token_embd.weight missing")
	}
	q, ok := nn.QuantOf(embd.DType)
	if !ok {
		return nil, fmt.Errorf("token_embd format %s unsupported", embd.DType)
	}
	w.TokenEmbd = nn.Matrix{
		Data:  embd.Raw,
		Quant: q,
		Rows:  cfg.Vocab,
		Cols:  cfg.Dim,
	}
	bindD4G(g, &w.TokenEmbd, "token_embd.weight")

	outNorm, ok := g.Tensors["output_norm.weight"]
	if !ok {
		return nil, fmt.Errorf("output_norm.weight missing")
	}
	var err error
	if w.OutputNorm, err = outNorm.F32(); err != nil {
		return nil, err
	}

	if outHead, ok := g.Tensors["output.weight"]; ok {
		hq, _ := nn.QuantOf(outHead.DType)
		w.OutputHead = nn.Matrix{
			Data:  outHead.Raw,
			Quant: hq,
			Rows:  cfg.Vocab,
			Cols:  cfg.Dim,
		}
		bindD4G(g, &w.OutputHead, "output.weight")
	} else {
		w.OutputHead = w.TokenEmbd
	}

	for i, bc := range cfg.Blocks {
		bw := &w.Blocks[i]
		prefix := fmt.Sprintf("blk.%d.", i)

		// Norms
		if t, ok := g.Tensors[prefix+"attn_norm.weight"]; ok {
			bw.AttnNorm, _ = t.F32()
		}
		if t, ok := g.Tensors[prefix+"post_attention_norm.weight"]; ok {
			bw.FFNNorm, _ = t.F32()
		}

		// FFN
		bindMatrix(g, prefix+"ffn_gate.weight", &bw.Gate, bc.FFN, cfg.Dim)
		bindMatrix(g, prefix+"ffn_up.weight", &bw.Up, bc.FFN, cfg.Dim)
		bindMatrix(g, prefix+"ffn_down.weight", &bw.Down, cfg.Dim, bc.FFN)

		if bc.Type == BlockFullAttn {
			// In Qwen 3.8, Q is fused with Gate: [5120, Heads*HeadDim*2]
			bindMatrix(g, prefix+"attn_q.weight", &bw.Q, bc.Heads*bc.HeadDim*2, cfg.Dim)
			bindMatrix(g, prefix+"attn_k.weight", &bw.K, bc.KVHeads*bc.HeadDim, cfg.Dim)
			bindMatrix(g, prefix+"attn_v.weight", &bw.V, bc.KVHeads*bc.HeadDim, cfg.Dim)
			bindMatrix(g, prefix+"attn_output.weight", &bw.O, cfg.Dim, bc.Heads*bc.HeadDim)
			if t, ok := g.Tensors[prefix+"attn_q_norm.weight"]; ok {
				bw.QNorm, _ = t.F32()
			}
			if t, ok := g.Tensors[prefix+"attn_k_norm.weight"]; ok {
				bw.KNorm, _ = t.F32()
			}
		} else {
			// Linear Attention (Gated Delta Net)
			bindMatrix(g, prefix+"attn_qkv.weight", &bw.QKV, 10240, cfg.Dim)
			bindMatrix(g, prefix+"attn_gate.weight", &bw.AttnGate, bc.SSMInnerSize, cfg.Dim)
			bindMatrix(g, prefix+"ssm_out.weight", &bw.SSMOut, cfg.Dim, bc.SSMInnerSize)
			bindMatrix(g, prefix+"ssm_alpha.weight", &bw.SSMAlpha, bc.SSMTimeStepRank, cfg.Dim)
			bindMatrix(g, prefix+"ssm_beta.weight", &bw.SSMBeta, bc.SSMTimeStepRank, cfg.Dim)

			if t, ok := g.Tensors[prefix+"ssm_conv1d.weight"]; ok {
				bw.Conv1D, _ = t.F32()
			}
			if t, ok := g.Tensors[prefix+"ssm_a"]; ok {
				bw.SSMA, _ = t.F32()
			}
			if t, ok := g.Tensors[prefix+"ssm_dt.bias"]; ok {
				bw.SSMDtBias, _ = t.F32()
			}
			if t, ok := g.Tensors[prefix+"ssm_norm.weight"]; ok {
				bw.SSMNorm, _ = t.F32()
			}
		}
	}

	// MTP block 64 weights
	if t, ok := g.Tensors["blk.64.nextn.enorm.weight"]; ok {
		w.MTP.ENorm, _ = t.F32()
		w.MTP.HasMTP = true
	}
	if t, ok := g.Tensors["blk.64.nextn.hnorm.weight"]; ok {
		w.MTP.HNorm, _ = t.F32()
	}
	if t, ok := g.Tensors["blk.64.nextn.shared_head_norm.weight"]; ok {
		w.MTP.SharedHeadNorm, _ = t.F32()
	}
	bindMatrix(g, "blk.64.nextn.eh_proj.weight", &w.MTP.EHProj, cfg.Dim, cfg.Dim*2)

	return w, nil
}

func bindMatrix(g *tensors.GGUF, name string, m *nn.Matrix, rows, cols int) {
	t, ok := g.Tensors[name]
	if !ok {
		return
	}
	q, _ := nn.QuantOf(t.DType)
	m.Data = t.Raw
	m.Quant = q
	m.Rows = rows
	m.Cols = cols
	bindD4G(g, m, name)
}

// bindD4G gives a D4G matrix the vector its activation must go through and the
// rotation that follows it, both from the file under a name nn.D4GVectorNames
// knows. A matrix in any other format has neither and this leaves it alone; the
// product does the transform itself, so nothing else in this package changes.
func bindD4G(g *tensors.GGUF, m *nn.Matrix, name string) {
	width, err := g.Uint32("golem.hadamard_group")
	if err != nil || width == 0 || !m.Quant.Golem() {
		return
	}
	for _, at := range nn.D4GVectorNames(name) {
		t, ok := g.Tensors[at]
		if !ok {
			continue
		}
		v, err := t.F32()
		if err != nil || len(v) != m.Cols {
			continue
		}
		m.Pre, m.HadGroup = v, int(width)
		return
	}
}
