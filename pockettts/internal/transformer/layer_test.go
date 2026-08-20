package transformer

// Numerical parity of layer 0 against PyTorch.
//
// The method, inherited from the spike: no layer is deemed correct until its
// intermediate activations match the reference. Comparing only the final output
// would say that there is an error, never where it is.

import (
	"testing"

	"github.com/ThiraSoft/golem/nn"
	"github.com/ThiraSoft/golem/pockettts/internal/reference"
	"github.com/ThiraSoft/golem/tensors"
)

func open(t *testing.T) (*tensors.Model, reference.Fixtures, *Layer) {
	t.Helper()
	f := reference.Load(t, "layer0")
	m, err := tensors.Open(reference.ModelPath(t, f.Language))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { m.Close() })
	c, err := LoadLayer(m, "flow_lm.transformer.", 0, Geometry{
		DModel: f.DModel, NumHeads: f.NumHeads, DimFF: f.DimFF, MaxPeriod: f.MaxPeriod,
	})
	if err != nil {
		t.Fatal(err)
	}
	return m, f, c
}

// TestStepsPosition0 checks every intermediate activation of the first step.
func TestStepsPosition0(t *testing.T) {
	_, f, c := open(t)
	d, h, D := c.HeadDim, c.NumHeads, c.DModel

	input := f.Read(t, "input")
	x := append([]float32(nil), input[:D]...) // position 0

	// norm1
	n1 := append([]float32(nil), x...)
	c.Norm1.Apply(n1)
	reference.Compare(t, "norm1", n1, f.Read(t, "norm1")[:D], 1e-4)

	// in_proj (bfloat16: the reference is float32, the gap comes from the weights)
	proj := make([]float32, 3*D)
	c.InProj.Apply(n1, proj)
	reference.Compare(t, "in_proj", proj, f.Read(t, "in_proj")[:3*D], 5e-3)

	// RoPE at position 0: rotation of angle zero, q and k must be unchanged
	q := append([]float32(nil), proj[:h*d]...)
	k := append([]float32(nil), proj[h*d:2*h*d]...)
	for head := 0; head < h; head++ {
		nn.ApplyRoPE(q[head*d:(head+1)*d], 0, c.MaxPeriod)
		nn.ApplyRoPE(k[head*d:(head+1)*d], 0, c.MaxPeriod)
	}
	reference.Compare(t, "q_rope", q, f.Read(t, "q_rope")[:h*d], 5e-3)
	reference.Compare(t, "k_rope", k, f.Read(t, "k_rope")[:h*d], 5e-3)
	reference.Compare(t, "v", proj[2*h*d:], f.Read(t, "v")[:h*d], 5e-3)

	// full block
	cache := NewCache(f.SeqLen, h, d)
	out := append([]float32(nil), x...)
	c.Step(out, cache)
	reference.Compare(t, "output[0]", out, f.Read(t, "output")[:D], 5e-3)
}

// TestFullSequence replays the three positions in a row: what is at stake is
// the KV cache and the RoPE offset, invisible at the first step.
func TestFullSequence(t *testing.T) {
	_, f, c := open(t)
	D := c.DModel

	input := f.Read(t, "input")
	want := f.Read(t, "output")
	cache := NewCache(f.SeqLen, c.NumHeads, c.HeadDim)

	for p := 0; p < f.SeqLen; p++ {
		x := append([]float32(nil), input[p*D:(p+1)*D]...)
		c.Step(x, cache)
		reference.Compare(t, "output[position "+string(rune('0'+p))+"]", x, want[p*D:(p+1)*D], 5e-3)
	}
	if cache.Position != f.SeqLen {
		t.Errorf("cache at position %d, want %d", cache.Position, f.SeqLen)
	}
}
