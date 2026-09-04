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

	"github.com/ThiraSoft/golem/nn"
	"github.com/ThiraSoft/golem/tensors"
)

// codebookEpsilon floors the denominator, as the reference does before dividing.
const codebookEpsilon = 1e-5

type residualVQ struct {
	in        nn.Conv1d   // LatentDim -> 256, kernel 1
	out       nn.Conv1d   // 256 -> LatentDim, kernel 1
	codebooks [][]float32 // one per layer: 2048 x 256
	// norms[b][c] is the squared length of entry c of codebook b, computed
	// once at load. nearest() says what it is for.
	norms [][]float32
}

type Quantizer struct {
	Codebooks int
	first     residualVQ
	rest      residualVQ
	dim       int // 256
	entries   int // 2048, the codebook's width
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
	q.entries = len(q.first.norms[0])
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
		norms := make([]float32, len(u))
		for c := range norms {
			var n float32
			for _, v := range book[c*dim : (c+1)*dim] {
				n += v * v
			}
			norms[c] = n
		}
		r.codebooks = append(r.codebooks, book)
		r.norms = append(r.norms, norms)
	}
	return r, nil
}

// Encode projects the latent into the code space once per half, then walks the
// codebooks: each takes the nearest entry to what is left, and subtracts it.
//
// The scratch is allocated per call and not held on the Quantizer: one
// Quantizer serves every conversation a server holds at once, and a shared
// buffer would have two of them writing the same distances.
func (q *Quantizer) Encode(latent []float32, codes []int) {
	scores := make([]float32, q.entries)
	residual := make([]float32, q.dim)
	k := 0
	for _, r := range []residualVQ{q.first, q.rest} {
		x, _ := r.in.Apply(latent, 1, r.in.NewState())
		copy(residual, x[:q.dim])
		for b, book := range r.codebooks {
			best := nearest(book, r.norms[b], residual, q.dim, scores)
			codes[k] = best
			k++
			row := book[best*q.dim : (best+1)*q.dim]
			for j := range residual {
				residual[j] -= row[j]
			}
		}
	}
}

// nearest returns the entry closest to x, ranking them by ||c||^2 - 2 x.c
// instead of by ||x - c||^2. The two differ by ||x||^2, which is the same for
// every entry and so decides nothing — and what the shorter form buys is that
// x.c is a dot product: eight lanes at a time in nn.DotF32, and split across
// the pool. The subtraction inside the square is neither, and it was costing
// 22 ms a frame of an 80 ms budget, more than the whole sixteen-block trunk.
func nearest(book, norms, x []float32, dim int, scores []float32) int {
	entries := len(norms)
	nn.InParallel(entries, entries*dim, func(start, end int) {
		for c := start; c < end; c++ {
			scores[c] = norms[c] - 2*nn.DotF32(book[c*dim:(c+1)*dim], x)
		}
	})
	best, bestScore := 0, scores[0]
	for c := 1; c < entries; c++ {
		if scores[c] < bestScore {
			best, bestScore = c, scores[c]
		}
	}
	return best
}
