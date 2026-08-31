package vk

import (
	"testing"

	"github.com/ThiraSoft/golem/nn"
)

// TestT3GDecodeMatchesCPUExactly is the contract a decoder is held to: it is a
// pure function of the bits it reads, so Go and the shader must agree to the
// last unit or the file means two things. A one-hot activation makes each
// product one weight, so a mismatch names the weight.
//
// This is unlike the encoder, which is held only to the same cost — see
// TestViterbiMatchesCPU and the reason in compress/README.md.
func TestT3GDecodeMatchesCPUExactly(t *testing.T) {
	testGolemDecodeMatchesCPUExactly(t, nn.T3G)
}
