package mimi

import (
	"testing"

	"github.com/ThiraSoft/golem/nn"
	"github.com/ThiraSoft/golem/pockettts/internal/reference"
	"github.com/ThiraSoft/golem/tensors"
)

// TestEncoderAgainstReference checks each stage of the encoder against what
// PyTorch computes on the same input: the SEANet stack, the transformer, and
// the fold into latents.
//
// The three are compared separately rather than only at the end, and that is
// what made the one mistake findable: the stack and the transformer were exact
// to a part in ten million while the latents were off by a factor of two on the
// first frame — which said, without ambiguity, that the fault was in the
// padding of the last layer and nowhere else.
func TestEncoderAgainstReference(t *testing.T) {
	f := reference.Load(t, "encoder")

	m, err := tensors.Open(reference.ModelPath(t, "french_24l"))
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()

	e, err := LoadEncoder(m, DefaultConfig)
	if err != nil {
		t.Fatal(err)
	}
	audio := f.Read(t, "audio")

	x, steps := e.input.Apply(audio, len(audio), e.input.NewState())
	for _, st := range e.stages {
		x = st.block.apply(x, steps, &blockState{
			s1: st.block.conv1.NewState(), s2: st.block.conv2.NewState(),
		})
		nn.ELU(x)
		x, steps = st.shrink.Apply(x, steps, st.shrink.NewState())
	}
	nn.ELU(x)
	x, steps = e.output.Apply(x, steps, e.output.NewState())
	reference.Compare(t, "encoder", x, f.Read(t, "encoder"), 5e-5)

	e.transformerSteps(x, steps)
	reference.Compare(t, "encoder_transformer", x, f.Read(t, "encoder_transformer"), 5e-5)

	latents, _ := e.down.Apply(x, steps, e.down.NewState())
	reference.Compare(t, "latents", latents, f.Read(t, "latents"), 5e-5)

	// And the whole thing through the public path, which must agree with the
	// stages it is made of.
	whole, err := e.Encode(audio)
	if err != nil {
		t.Fatal(err)
	}
	reference.Compare(t, "Encode", whole, f.Read(t, "latents"), 5e-5)
}

// A recording that does not fill a whole frame is refused rather than encoded
// short: the caller pads, because only the caller knows with what.
func TestEncodeRefusesAPartialFrame(t *testing.T) {
	e := &Encoder{Config: DefaultConfig}
	if _, err := e.Encode(make([]float32, SamplesPerFrame+1)); err == nil {
		t.Fatal("expected an error for a partial frame")
	}
}
