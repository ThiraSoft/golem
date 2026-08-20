package flowlm

// The flow net: what turns the transformer's output into an audio latent.
//
// The transformer does not predict the frame's latent, it predicts *the
// direction* leading from Gaussian noise to that latent. A small perceptron of
// six residual blocks, conditioned by the transformer's output and by two
// instants, gives this velocity field; integrating it over one step is enough —
// that is the point of the distillation the model was trained with.
//
// The conditioning does not enter through the input but through the
// normalizations: each block derives a shift, a scale and a gate from it, which
// it applies to its activations. Hence the profusion of small matrices and
// biases, where the transformer had none.

import (
	"fmt"
	"math"

	"github.com/ThiraSoft/golem/nn"
	"github.com/ThiraSoft/golem/pockettts/internal/transformer"
	"github.com/ThiraSoft/golem/tensors"
)

// TimeEncoder embeds a scalar instant into a conditioning vector.
type TimeEncoder struct {
	Freqs  []float32 // 128 frequencies, stored in the weights
	MLP0   nn.Linear // 256 -> 512
	MLP2   nn.Linear // 512 -> 512
	Norm   nn.RMSNorm
	buffer []float32
}

func (e *TimeEncoder) Apply(t float32, out []float32) {
	if e.buffer == nil {
		e.buffer = make([]float32, 2*len(e.Freqs))
	}
	half := len(e.Freqs)
	for i, f := range e.Freqs {
		angle := float64(t) * float64(f)
		e.buffer[i] = float32(math.Cos(angle))
		e.buffer[half+i] = float32(math.Sin(angle))
	}
	e.MLP0.Apply(e.buffer, out)
	nn.SiLU(out)
	tmp := make([]float32, len(out))
	e.MLP2.Apply(out, tmp)
	copy(out, tmp)
	e.Norm.Apply(out)
}

// ResidualBlock applies an MLP modulated by the conditioning.
type ResidualBlock struct {
	Modulation nn.Linear // 512 -> 1536: shift, scale, gate
	InLN       nn.LayerNorm
	MLP0, MLP2 nn.Linear
	mod, h, y  []float32
}

func (b *ResidualBlock) Apply(x, cond []float32) {
	n := len(x)
	if b.mod == nil {
		b.mod = make([]float32, 3*n)
		b.h = make([]float32, n)
		b.y = make([]float32, n)
	}
	copy(b.y, cond)
	nn.SiLU(b.y)
	b.Modulation.Apply(b.y, b.mod)
	shift, scale, gate := b.mod[:n], b.mod[n:2*n], b.mod[2*n:]

	copy(b.h, x)
	b.InLN.Apply(b.h)
	for i := range b.h {
		b.h[i] = b.h[i]*(1+scale[i]) + shift[i]
	}
	tmp := make([]float32, n)
	b.MLP0.Apply(b.h, tmp)
	nn.SiLU(tmp)
	b.MLP2.Apply(tmp, b.h)

	for i := range x {
		x[i] += gate[i] * b.h[i]
	}
}

// FinalLayer brings the internal width back down to the latent's dimension.
type FinalLayer struct {
	Modulation nn.Linear // 512 -> 1024: shift and scale
	Linear     nn.Linear // 512 -> 32
	mod, y     []float32
}

func (c *FinalLayer) Apply(x, cond, out []float32) {
	n := len(x)
	if c.mod == nil {
		c.mod = make([]float32, 2*n)
		c.y = make([]float32, n)
	}
	copy(c.y, cond)
	nn.SiLU(c.y)
	c.Modulation.Apply(c.y, c.mod)
	shift, scale := c.mod[:n], c.mod[n:]

	nn.NormalizeNoAffine(x, 1e-6)
	for i := range x {
		x[i] = x[i]*(1+scale[i]) + shift[i]
	}
	c.Linear.Apply(x, out)
}

// FlowNet assembles the whole thing.
type FlowNet struct {
	Time                [2]*TimeEncoder
	CondEmbed           nn.Linear // 1024 -> 512
	InputProj           nn.Linear // 32 -> 512
	Blocks              []*ResidualBlock
	Final               FinalLayer
	cond, t0, t1, state []float32
}

// Velocity returns the flow field at point x, between the instants s and t.
func (r *FlowNet) Velocity(condTransformer []float32, s, t float32, x, out []float32) {
	width := r.InputProj.Outputs
	if r.cond == nil {
		r.cond = make([]float32, width)
		r.t0 = make([]float32, width)
		r.t1 = make([]float32, width)
		r.state = make([]float32, width)
	}
	r.Time[0].Apply(s, r.t0)
	r.Time[1].Apply(t, r.t1)
	r.CondEmbed.Apply(condTransformer, r.cond)
	for i := range r.cond {
		r.cond[i] += (r.t0[i] + r.t1[i]) / 2
	}

	r.InputProj.Apply(x, r.state)
	for _, b := range r.Blocks {
		b.Apply(r.state, r.cond)
	}
	r.Final.Apply(r.state, r.cond, out)
}

// Latent integrates the flow from the starting noise. The model is distilled for
// a single step: `steps` is 1 in practice, and the loop exists only because the
// formulation allows it.
func (r *FlowNet) Latent(cond, noise []float32, steps int, out []float32) {
	copy(out, noise)
	velocity := make([]float32, len(out))
	for i := 0; i < steps; i++ {
		s := float32(i) / float32(steps)
		t := float32(i+1) / float32(steps)
		r.Velocity(cond, s, t, append([]float32(nil), out...), velocity)
		for j := range out {
			out[j] += velocity[j] / float32(steps)
		}
	}
}

// LoadFlowNet reads the weights of the flow net.
func LoadFlowNet(m *tensors.Model, cfg Config) (*FlowNet, error) {
	const (
		prefix = "flow_lm.flow_net."
		width  = 512
		blocks = 6
	)
	ld := transformer.Loader{M: m, Prefix: prefix}

	r := &FlowNet{}
	for i := 0; i < 2; i++ {
		p := fmt.Sprintf("time_embed.%d.", i)
		freqs, err := ld.Vector(p + "freqs")
		if err != nil {
			return nil, err
		}
		alpha, err := ld.Vector(p + "mlp.3.alpha")
		if err != nil {
			return nil, err
		}
		mlp0, err := ld.Linear(p+"mlp.0", 2*len(freqs), width)
		if err != nil {
			return nil, err
		}
		mlp2, err := ld.Linear(p+"mlp.2", width, width)
		if err != nil {
			return nil, err
		}
		r.Time[i] = &TimeEncoder{
			Freqs: freqs, MLP0: mlp0, MLP2: mlp2,
			Norm: nn.RMSNorm{Alpha: alpha, Eps: 1e-5},
		}
	}

	var err error
	if r.CondEmbed, err = ld.Linear("cond_embed", cfg.DModel, width); err != nil {
		return nil, err
	}
	if r.InputProj, err = ld.Linear("input_proj", cfg.LatentDim, width); err != nil {
		return nil, err
	}

	for i := 0; i < blocks; i++ {
		p := fmt.Sprintf("res_blocks.%d.", i)
		b := &ResidualBlock{}
		if b.Modulation, err = ld.Linear(p+"adaLN_modulation.1", width, 3*width); err != nil {
			return nil, err
		}
		if b.InLN, err = ld.Norm(p+"in_ln", 1e-6); err != nil {
			return nil, err
		}
		if b.MLP0, err = ld.Linear(p+"mlp.0", width, width); err != nil {
			return nil, err
		}
		if b.MLP2, err = ld.Linear(p+"mlp.2", width, width); err != nil {
			return nil, err
		}
		r.Blocks = append(r.Blocks, b)
	}

	if r.Final.Modulation, err = ld.Linear("final_layer.adaLN_modulation.1", width, 2*width); err != nil {
		return nil, err
	}
	if r.Final.Linear, err = ld.Linear("final_layer.linear", width, cfg.LatentDim); err != nil {
		return nil, err
	}
	return r, nil
}
