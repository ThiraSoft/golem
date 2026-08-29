package qwen35

// A tap on the inputs of the weight matrices, for offline weight compression.
//
// The same hook qwen/calib.go is and for the same reason: the quality of a
// quantized weight depends on the activations that meet it, and those are only
// visible from inside a block. Nothing in inference reads this — Calib is nil
// unless a compression tool sets it, and the sites call it only when it is not.
//
// The sites are the same four names this model's two kinds of block share.
// "qkv" is what the attention norm produced, which a full attention reads with
// three projections and a delta net with four; "o" is what the mixer answered,
// which either reads with one; "gateup" and "down" are the feed forward's, and
// the feed forward is the same in both.
var Calib func(block int, site string, rows [][]float32)

func calib(block int, site string, rows ...[]float32) {
	if Calib != nil {
		Calib(block, site, rows)
	}
}
