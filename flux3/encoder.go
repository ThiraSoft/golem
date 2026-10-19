package flux3

import (
	"github.com/ThiraSoft/golem/krea2"
	"github.com/ThiraSoft/golem/vk"
)

// Taps are the layers FLUX 3 reads the text model's state on entry to.
// ComfyUI (Flux3Qwen3VLModel in comfy/text_encoders/flux.py) configures
// layer=[4, 8, 12, 16, 20, 24, 28, 32]. In comfy/text_encoders/llama.py,
// all_intermediate is appended before each layer executes (i in only_layers).
// Tap k is therefore the hidden state ENTERING layer k (a pre-hook on layer k),
// exactly as verified against fixture encoder/p0.hidden_l4 which has 0.0 max
// difference to cond[:, :2560]. Because tap 32 enters layer 32, layers 32..35
// never run.
var Taps = [8]int{4, 8, 12, 16, 20, 24, 28, 32}

// CondWidth is one token's conditioning: eight taps of the model's 2560 width.
const CondWidth = len(Taps) * 2560

// OpenEncoder loads Qwen3-VL-4B up to layer 32 with FLUX 3's taps.
func OpenEncoder(d *vk.Device, path string, host bool) (*krea2.Encoder, error) {
	return krea2.OpenEncoderTaps(d, path, host, Taps[:])
}
