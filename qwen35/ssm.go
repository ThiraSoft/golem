package qwen35

import (
	"math"

	"github.com/ThiraSoft/golem/nn"
)

// ForwardSSMToken computes one step of Qwen3.8-27B Linear Attention (Gated Delta Net).
//
// Channels breakdown in QKV [10240]:
// - Q : [0 : 2048]     -> 16 groups x 128 head_dim
// - K : [2048 : 4096]  -> 16 groups x 128 head_dim
// - V : [4096 : 10240] -> 48 groups x 128 head_dim
//
// State: 48 heads x 128 x 128 floats per SSM layer.
func ForwardSSMToken(
	cfg *Config, bc BlockConfig, bw *BlockWeights,
	cache *BlockCache, x []float32, out []float32, scratch *Scratch,
) {
	// 1. Projections:
	// qkv = bw.QKV * x -> [10240]
	// gate = bw.AttnGate * x -> [6144] (Gate Z)
	// alpha = bw.SSMAlpha * x -> [48]
	// beta = bw.SSMBeta * x -> [48]
	qkv := scratch.qkv[:10240]
	bw.QKV.MatVec(scratch.batchX, qkv)

	gateZ := scratch.gate[:bc.SSMInnerSize]
	bw.AttnGate.MatVec(scratch.batchX, gateZ)

	alpha := scratch.alpha[:bc.SSMTimeStepRank]
	beta := scratch.beta[:bc.SSMTimeStepRank]
	bw.SSMAlpha.MatVec(scratch.batchX, alpha)
	bw.SSMBeta.MatVec(scratch.batchX, beta)

	// 2. Activations for Alpha and Beta:
	// beta = sigmoid(beta)
	// decay = exp(softplus(alpha + dt_bias) * ssm_a)
	decay := scratch.decay[:bc.SSMTimeStepRank]
	for h := 0; h < bc.SSMTimeStepRank; h++ {
		beta[h] = float32(1.0 / (1.0 + math.Exp(float64(-beta[h]))))

		val := alpha[h] + bw.SSMDtBias[h]
		var softplusVal float32
		if val > 20.0 {
			softplusVal = val
		} else {
			softplusVal = float32(math.Log1p(math.Exp(float64(val))))
		}
		g := softplusVal * bw.SSMA[h] // ssm_a is negative
		decay[h] = float32(math.Exp(float64(g)))
	}

	// 3. Causal 1D Convolution over qkv [10240] with kernel 4:
	convOut := scratch.convOut[:10240]
	convDim := 10240
	kernel := bc.SSMConvKernel

	// The GGUF kernel is [conv_kernel, channels] with the kernel contiguous:
	// ggml reads c[tap + channel*n_taps] (ggml_compute_forward_ssm_conv_f32).
	// Indexing it the other way round mixes every channel's taps together.
	for c := 0; c < convDim; c++ {
		w := bw.Conv1D[c*kernel : (c+1)*kernel]
		sum := w[0]*cache.ConvState[0*convDim+c] +
			w[1]*cache.ConvState[1*convDim+c] +
			w[2]*cache.ConvState[2*convDim+c] +
			w[3]*qkv[c]
		// SiLU activation
		convOut[c] = sum / (1.0 + float32(math.Exp(float64(-sum))))
	}

	// Shift circular buffer
	copy(cache.ConvState[0*convDim:1*convDim], cache.ConvState[1*convDim:2*convDim])
	copy(cache.ConvState[1*convDim:2*convDim], cache.ConvState[2*convDim:3*convDim])
	copy(cache.ConvState[2*convDim:3*convDim], qkv)

	// 4. Split channels:
	// Q: [0 : 2048] (16 heads x 128)
	// K: [2048 : 4096] (16 heads x 128)
	// V: [4096 : 10240] (48 heads x 128)
	qChannels := convOut[0:2048]
	kChannels := convOut[2048:4096]
	vChannels := convOut[4096:10240]

	// 5. L2 Normalization on Q and K heads:
	numKHeads := bc.SSMGroupCount // 16
	stateSize := bc.SSMStateSize  // 128
	for i := 0; i < numKHeads; i++ {
		qHead := qChannels[i*stateSize : (i+1)*stateSize]
		var qSq float32
		for _, val := range qHead {
			qSq += val * val
		}
		invQ := float32(1.0 / math.Sqrt(float64(qSq+cfg.Eps)))
		for d := 0; d < stateSize; d++ {
			qHead[d] *= invQ
		}

		kHead := kChannels[i*stateSize : (i+1)*stateSize]
		var kSq float32
		for _, val := range kHead {
			kSq += val * val
		}
		invK := float32(1.0 / math.Sqrt(float64(kSq+cfg.Eps)))
		for d := 0; d < stateSize; d++ {
			kHead[d] *= invK
		}
	}

	// 6. Gated Delta Net recurrence for each of the 48 heads:
	numVHeads := bc.SSMTimeStepRank // 48
	ySSM := scratch.ySSM[:bc.SSMInnerSize]
	scale := float32(1.0 / math.Sqrt(float64(stateSize))) // 1 / sqrt(128)
	delta := make([]float32, stateSize)

	for h := 0; h < numVHeads; h++ {
		// Map head h to Q and K head (h % 16)
		kHeadIdx := h % numKHeads
		qHead := qChannels[kHeadIdx*stateSize : (kHeadIdx+1)*stateSize]
		kHead := kChannels[kHeadIdx*stateSize : (kHeadIdx+1)*stateSize]
		vHead := vChannels[h*stateSize : (h+1)*stateSize]
		yHead := ySSM[h*stateSize : (h+1)*stateSize]

		sBase := h * stateSize * stateSize
		dVal := decay[h]
		bVal := beta[h]

		// For each row j (0..127):
		for j := 0; j < stateSize; j++ {
			row := cache.SSMState[sBase+j*stateSize : sBase+(j+1)*stateSize]

			// 1. Decay state
			var sumK float32
			for i := 0; i < stateSize; i++ {
				hVal := row[i] * dVal
				row[i] = hVal
				sumK += hVal * kHead[i]
			}

			// 2. Compute delta
			delta[j] = (vHead[j] - sumK) * bVal
		}

		// 3. Update state with outer product and compute output with Q
		for j := 0; j < stateSize; j++ {
			row := cache.SSMState[sBase+j*stateSize : sBase+(j+1)*stateSize]
			d := delta[j]
			var sumQ float32
			for i := 0; i < stateSize; i++ {
				hVal := row[i] + d*kHead[i]
				row[i] = hVal
				sumQ += hVal * qHead[i]
			}
			yHead[j] = sumQ * scale
		}
	}

	// 7. Gated RMSNorm per 128 elements chunk + SiLU(Z):
	for h := 0; h < numVHeads; h++ {
		yHead := ySSM[h*stateSize : (h+1)*stateSize]
		nn.RMSNormPlain(yHead, bw.SSMNorm, cfg.Eps)
	}

	// Gate with SiLU(gateZ)
	for i := 0; i < bc.SSMInnerSize; i++ {
		zVal := gateZ[i]
		siluZ := zVal / (1.0 + float32(math.Exp(float64(-zVal))))
		ySSM[i] *= siluZ
	}

	// 8. Output Projection: out = bw.SSMOut * ySSM -> [Dim = 5120]
	load(scratch.batchYSSM, ySSM)
	bw.SSMOut.MatVec(scratch.batchYSSM, out)
}
