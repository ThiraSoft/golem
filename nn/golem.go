package nn

// The Golem scheme: what every one of golem's own formats shares, whatever
// codebook sits inside a block.
//
// A matrix is stored as A·(q ⊙ W): a per-column salience scale, then a Hadamard
// rotation. The activation meets the reciprocal on the way in — see
// PrepareGolem — so the product is unchanged and the rotation is undone nowhere
// but here, in Prepare and Unprepare. What varies between formats is only the
// codebook a block's weights are rounded onto; the vector, the rotation and the
// site a matrix reads are the same question for all of them, and this file is
// where that question is answered once.
//
// GolemBlock and GolemSubBlock name the block a step covers, and the step grid
// itself — golemSteps, GolemStep, GolemStepCode — is shared by every tier that
// has existed: two eight-bit codes a block, one per thirty-two weights, naming
// powers of two a sixteenth apart. A grid an eighth apart costs a whole point
// of perplexity against an fp16 step at the same granularity, which is more
// than the finer granularity wins back; a sixteenth apart is four percent over
// a range from 7.6e-6 to 0.48, wider than any weight of any model has asked
// for, and near enough the bottom of the curve to cost a tenth of what a
// coarser grid did.

import (
	"math"
	"strings"
)

// golemSteps is what the eight bits of a step code name.
var golemSteps [256]float32

// GolemStep expands a step code.
func GolemStep(code byte) float32 { return golemSteps[code] }

// GolemStepCode is the code nearest a step, in the ratio the codes are spaced
// by.
func GolemStepCode(v float32) byte {
	if !(v > 0) {
		return 0
	}
	c := math.Round(math.Log2(float64(v))*16 + 272)
	if c < 0 {
		c = 0
	}
	if c > 255 {
		c = 255
	}
	return byte(c)
}

func init() {
	for c := 0; c < 256; c++ {
		golemSteps[c] = float32(math.Exp2((float64(c) - 272) / 16))
	}
}

// golemSites says which activation a matrix reads, by the name of the tensor.
// Matrices sharing a site share the vector and the rotation, because they read
// the same activation — the three attention projections read the stream, the
// gate and the up read the feed forward's norm, and a mixture's two stacks read
// neither of those.
var golemSites = map[string]string{
	"attn_q": "qkv", "attn_k": "qkv", "attn_v": "qkv",
	"attn_output": "o",
	"ffn_gate":    "gateup", "ffn_up": "gateup",
	"ffn_down":         "down",
	"ffn_gate_up_exps": "gateup_exps",
	"ffn_down_exps":    "down_exps",
	// A hybrid's linear-attention block. Its four input projections read the
	// same normed stream a full attention's three do, and its output
	// projection stands where the attention's does — and no block is both
	// kinds, so the two sites are free to be the same two names.
	"attn_qkv": "qkv", "attn_gate": "qkv",
	"ssm_alpha": "qkv", "ssm_beta": "qkv",
	"ssm_out": "o",
}

// GolemSite names the activation a block's matrix reads, given the field of the
// tensor's name that says which matrix it is — attn_k of blk.7.attn_k.weight.
// A converter needs it to decide which site's statistics a matrix is quantized
// against, and it is the same table GolemVectorNames files the result under, so
// that the two answers cannot be given differently.
func GolemSite(matrix string) (string, bool) {
	site, ok := golemSites[matrix]
	return site, ok
}

// GolemVectorNames is where to look for the vector a matrix's activation must
// go through, most specific first. A converter writes one of these and a reader
// takes the first it finds, so that the two cannot drift apart: the naming is
// part of the format and lives here rather than in either of them.
//
// A block's matrix is filed under its site — blk.7.attn_k.weight reads
// blk.7.qkv.pre — and anything else under its own name. The tied head is both:
// output.pre when the converter had activations to measure it from, and its own
// name when it did not.
func GolemVectorNames(tensor string) []string {
	// The logit head, whether it is the table read the other way round or a
	// matrix of its own. Both read what the final norm made, so both are the
	// same site, and a model with an untied head has the two of them filed
	// under it. output.pre is what a converter calls that site.
	if tensor == "token_embd.weight" || tensor == "output.weight" {
		return []string{"output.pre", tensor + ".pre"}
	}
	if parts := strings.Split(tensor, "."); len(parts) == 4 && parts[0] == "blk" {
		if site, ok := golemSites[parts[2]]; ok {
			return []string{parts[0] + "." + parts[1] + "." + site + ".pre", tensor + ".pre"}
		}
	}
	return []string{tensor + ".pre"}
}
