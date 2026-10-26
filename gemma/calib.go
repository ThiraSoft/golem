package gemma

// A tap on the inputs of the weight matrices, for offline weight compression.
//
// The same hook qwen/calib.go is and for the same reason: the quality of a
// quantized weight depends on the activations that meet it, and those are only
// visible from inside a block. Nothing in inference reads this — Calib is nil
// unless a compression tool sets it, and the sites call it only when it is not.
//
// The sites are the names nn.GolemSite files the matrices under: "qkv" is the
// stream under attn_norm, "o" what the heads answered, "gateup" the stream under
// ffn_norm and "down" the gated product. A mixture block's shared expert reads
// the same two as a dense feed forward; its routed experts are not tapped, so
// their stacks are quantized blind. "head" is the final norm's output, block -1.
//
// Only the processor's path taps: a model running on the card never shows its
// activations to the host.
var Calib func(block int, site string, rows [][]float32)

func calib(block int, site string, rows [][]float32) {
	if Calib != nil {
		Calib(block, site, rows)
	}
}
