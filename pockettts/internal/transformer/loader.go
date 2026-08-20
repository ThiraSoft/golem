package transformer

// A small reading helper: the modules of the flow net and of the decoder all
// follow the same naming convention — a prefix, then `.weight` and `.bias`.
// Saying it once saves thirty lines of error checking per module.

import (
	"fmt"

	"github.com/ThiraSoft/golem/nn"
	"github.com/ThiraSoft/golem/tensors"
)

// Loader is exported for the packages that assemble their own modules from the
// same naming conventions.
type Loader struct {
	M      *tensors.Model
	Prefix string
}

func (c Loader) Vector(name string) ([]float32, error) {
	t, err := c.M.Get(c.Prefix + name)
	if err != nil {
		return nil, err
	}
	return t.F32()
}

// Linear reads a projection and its bias if it has one.
func (c Loader) Linear(name string, inputs, outputs int) (nn.Linear, error) {
	t, err := c.M.Get(c.Prefix + name + ".weight")
	if err != nil {
		return nn.Linear{}, err
	}
	if len(t.Shape) != 2 || t.Shape[0] != outputs || t.Shape[1] != inputs {
		return nn.Linear{}, fmt.Errorf("%s: shape %v, want [%d %d]", name, t.Shape, outputs, inputs)
	}
	if t.DType != "BF16" {
		return nn.Linear{}, fmt.Errorf("%s: dtype %s", name, t.DType)
	}
	l := nn.Linear{Weights: t.Raw, Inputs: inputs, Outputs: outputs}
	if b, err := c.M.Get(c.Prefix + name + ".bias"); err == nil {
		if l.Bias, err = b.F32(); err != nil {
			return nn.Linear{}, err
		}
	}
	return l, nil
}

func (c Loader) Norm(name string, eps float32) (nn.LayerNorm, error) {
	gain, err := c.Vector(name + ".weight")
	if err != nil {
		return nn.LayerNorm{}, err
	}
	bias, err := c.Vector(name + ".bias")
	if err != nil {
		return nn.LayerNorm{}, err
	}
	return nn.LayerNorm{Gain: gain, Bias: bias, Eps: eps}, nil
}
