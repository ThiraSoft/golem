package qwen

// A tap on the inputs of the weight matrices, for offline weight compression.
//
// Nothing in inference reads this: Calib is nil unless a compression tool sets
// it, and the sites call it only when it is not. It exists because the quality
// of a quantized weight depends on the activations that meet it, and those are
// only visible from inside a block.

// Calib, when set, is called with every batch of activations just before it
// meets a weight matrix. site names the matrix the batch feeds: "qkv", "o",
// "gateup" or "down" inside a block, and "head" for the tied output head,
// whose block number is -1 because it belongs to no block.
var Calib func(block int, site string, rows [][]float32)

func calib(block int, site string, rows [][]float32) {
	if Calib != nil {
		Calib(block, site, rows)
	}
}
