package mimi

// Mimi's quantiser, encoding side. The codec splits it in two: rvq_first holds
// the single semantic codebook, rvq_rest the thirty-one acoustic ones, each
// half with its own projection in and out of a 256-wide code space.
//
// The trap is in the storage. The file holds `embedding_sum` and
// `cluster_usage` — the numerator and denominator of a running average — and
// not the codebook. Reading the numerator as the codebook gives vectors a few
// hundred times too long, a nearest neighbour that is always the same entry,
// and a transcript of confident nonsense.

import (
	"fmt"
	"math"

	"github.com/ThiraSoft/golem/nn"
	"github.com/ThiraSoft/golem/tensors"
)

// codebookEpsilon floors the denominator, as the reference does before dividing.
const codebookEpsilon = 1e-5

type residualVQ struct {
	in        nn.Conv1d   // LatentDim -> 256, kernel 1
	out       nn.Conv1d   // 256 -> LatentDim, kernel 1
	codebooks [][]float32 // one per layer: 2048 x 256
}

type Quantizer struct {
	Codebooks int
	first     residualVQ
	rest      residualVQ
	dim       int // 256
}

func LoadQuantizer(m *tensors.Model, cfg Config) (*Quantizer, error) {
	q := &Quantizer{dim: 256}
	var err error
	if q.first, err = loadRVQ(m, cfg.Prefix+"quantizer.rvq_first", cfg.LatentDim, q.dim, 1); err != nil {
		return nil, err
	}
	if q.rest, err = loadRVQ(m, cfg.Prefix+"quantizer.rvq_rest", cfg.LatentDim, q.dim, 31); err != nil {
		return nil, err
	}
	q.Codebooks = len(q.first.codebooks) + len(q.rest.codebooks)
	return q, nil
}

func loadRVQ(m *tensors.Model, prefix string, latent, dim, layers int) (residualVQ, error) {
	var r residualVQ
	var err error
	if r.in, err = loadConv(m, prefix+".input_proj", latent, dim, 1, 1, 1); err != nil {
		return r, err
	}
	if r.out, err = loadConv(m, prefix+".output_proj", dim, latent, 1, 1, 1); err != nil {
		return r, err
	}
	for i := 0; i < layers; i++ {
		base := fmt.Sprintf("%s.vq.layers.%d._codebook.", prefix, i)
		sum, err := m.Get(base + "embedding_sum")
		if err != nil {
			return r, err
		}
		usage, err := m.Get(base + "cluster_usage")
		if err != nil {
			return r, err
		}
		s, err := sum.F32()
		if err != nil {
			return r, err
		}
		u, err := usage.F32()
		if err != nil {
			return r, err
		}
		if len(s) != len(u)*dim {
			return r, fmt.Errorf("%s: embedding_sum %d values for %d clusters of %d", base, len(s), len(u), dim)
		}
		book := make([]float32, len(s))
		for c := range u {
			d := u[c]
			if d < codebookEpsilon {
				d = codebookEpsilon
			}
			for j := 0; j < dim; j++ {
				book[c*dim+j] = s[c*dim+j] / d
			}
		}
		r.codebooks = append(r.codebooks, book)
	}
	return r, nil
}

// Encode projects the latent into the code space once per half, then walks the
// codebooks: each takes the nearest entry to what is left, and subtracts it.
func (q *Quantizer) Encode(latent []float32, codes []int) {
	k := 0
	for _, r := range []residualVQ{q.first, q.rest} {
		x, _ := r.in.Apply(latent, 1, r.in.NewState())
		residual := make([]float32, q.dim)
		copy(residual, x[:q.dim])
		for _, book := range r.codebooks {
			best, bestDist := 0, math.MaxFloat64
			for c := 0; c < len(book)/q.dim; c++ {
				var d float64
				row := book[c*q.dim : (c+1)*q.dim]
				for j, v := range residual {
					e := float64(v - row[j])
					d += e * e
				}
				if d < bestDist {
					best, bestDist = c, d
				}
			}
			codes[k] = best
			k++
			row := book[best*q.dim : (best+1)*q.dim]
			for j := range residual {
				residual[j] -= row[j]
			}
		}
	}
}
