package transformer

// One layer of the flow_lm transformer, executed one timestep at a time.
//
// Generation is autoregressive at batch one: every audio frame adds a single
// position to the sequence. Writing the layer for one step — rather than for a
// [B, T, D] tensor that would then have to be sliced — removes all the index
// gymnastics, and the initial prompt fill is obtained by calling Step as many
// times as there are positions, with exactly the same arithmetic. The KV cache
// makes the two paths equivalent by construction.

import (
	"fmt"
	"math"

	"github.com/ThiraSoft/golem/nn"
	"github.com/ThiraSoft/golem/tensors"
)

// Layer holds the weights of one layer and its geometry.
type Layer struct {
	Norm1, Norm2 nn.LayerNorm
	InProj       nn.Linear // projects to Q, K and V concatenated
	OutProj      nn.Linear
	Linear1      nn.Linear
	Linear2      nn.Linear
	Scale1       []float32 // layer_scale: nil when the layer has none
	Scale2       []float32
	Context      int // number of visible past positions; 0 = all
	NumHeads     int
	HeadDim      int
	DModel       int
	MaxPeriod    float64
	scratchProj  []float32
	scratchAttn  []float32
	scratchFF    []float32
	scratchIn    []float32
	scratchOut   []float32
	batchAlloc   int
}

// Cache holds the keys and values already computed: without it, every new
// position would force all the previous ones to be run again.
type Cache struct {
	K, V     []float32 // capacity x NumHeads x HeadDim, laid out by position
	Position int       // number of positions already written
	capacity int
	numHeads int
	headDim  int
}

// NewCache allocates a cache for at most `capacity` positions.
func NewCache(capacity, numHeads, headDim int) *Cache {
	n := capacity * numHeads * headDim
	return &Cache{
		K: make([]float32, n), V: make([]float32, n),
		capacity: capacity, numHeads: numHeads, headDim: headDim,
	}
}

func (c *Cache) Reset() { c.Position = 0 }

// Step advances the layer by one position: x is modified in place and holds the
// output on return.
func (c *Layer) Step(x []float32, cache *Cache) { c.Block(x, 1, cache) }

// Block advances the layer by `batch` consecutive positions, laid end to end in
// x, modified in place.
//
// Processing several positions at once is not a convenience: it is what decides
// throughput. Every position re-reads the whole of the layer's matrices, and one
// position at a time the processor spends its time waiting on memory. The
// flow_lm has no choice — it generates one frame after another — but the audio
// transformer receives sixteen positions together, and the text prompt brings
// several dozen.
func (c *Layer) Block(x []float32, batch int, cache *Cache) {
	c.allocate(batch)

	D, d, h := c.DModel, c.HeadDim, c.NumHeads

	// --- self-attention block ---
	copy(c.scratchIn, x[:batch*D]) // the residual connection starts from the raw input
	copy(c.scratchOut, x[:batch*D])
	for l := 0; l < batch; l++ {
		c.Norm1.Apply(c.scratchOut[l*D : (l+1)*D])
	}
	c.InProj.ApplyBatch(c.scratchOut, c.scratchProj, batch)

	// The projections of the `batch` positions are independent; the attention is
	// not. So we apply the rotation and fill the cache for all of them, before
	// walking the positions in order.
	for l := 0; l < batch; l++ {
		proj := c.scratchProj[l*3*D : (l+1)*3*D]
		q, k, v := proj[0:h*d], proj[h*d:2*h*d], proj[2*h*d:]
		position := cache.Position + l
		for t := 0; t < h; t++ {
			nn.ApplyRoPE(q[t*d:(t+1)*d], position, c.MaxPeriod)
			nn.ApplyRoPE(k[t*d:(t+1)*d], position, c.MaxPeriod)
		}
		base := position * h * d
		copy(cache.K[base:base+h*d], k)
		copy(cache.V[base:base+h*d], v)
	}

	c.attention(batch, cache)
	cache.Position += batch

	c.OutProj.ApplyBatch(c.scratchAttn, c.scratchOut, batch)
	for l := 0; l < batch; l++ {
		applyScale(c.Scale1, c.scratchOut[l*D:(l+1)*D])
	}
	for i := range x[:batch*D] {
		x[i] = c.scratchIn[i] + c.scratchOut[i]
	}

	// --- feed-forward block ---
	copy(c.scratchIn, x[:batch*D])
	copy(c.scratchOut, x[:batch*D])
	for l := 0; l < batch; l++ {
		c.Norm2.Apply(c.scratchOut[l*D : (l+1)*D])
	}
	c.Linear1.ApplyBatch(c.scratchOut, c.scratchFF, batch)
	nn.GELU(c.scratchFF[:batch*c.Linear1.Outputs])
	c.Linear2.ApplyBatch(c.scratchFF, c.scratchOut, batch)
	for l := 0; l < batch; l++ {
		applyScale(c.Scale2, c.scratchOut[l*D:(l+1)*D])
	}
	for i := range x[:batch*D] {
		x[i] = c.scratchIn[i] + c.scratchOut[i]
	}
}

// attention computes the output of the `batch` positions whose keys and values
// have just been written into the cache.
//
// Every (position, head) pair is a task of its own, and they go to the pool in
// one section rather than one section of heads per position. The heads alone
// are eight tasks — one barrier per position, and a section the pool judges too
// small to split at all; sixteen positions of eight heads is a hundred and
// twenty-eight tasks in a single section, and enough work in it that splitting
// pays. Nothing else changes: the keys and values of the whole batch are
// written before any of this runs, so the position that reads up to itself is
// reading what is already there.
func (c *Layer) attention(batch int, cache *Cache) {
	D, d, h := c.DModel, c.HeadDim, c.NumHeads
	scale := float32(1 / math.Sqrt(float64(d)))

	// The longest window any of these positions looks back over, which is what
	// one worker's scores buffer has to hold.
	longest := cache.Position + batch
	if c.Context > 0 && c.Context < longest {
		longest = c.Context
	}

	nn.InParallel(batch*h, batch*h*longest*d*2, func(start, end int) {
		buffer := make([]float32, longest)
		for task := start; task < end; task++ {
			l, t := task/h, task%h

			// The visible positions run from `first` to the current one. The
			// flow_lm sees all of its past; the audio decoder is limited to a
			// window, which bounds the cost of attention on long utterances.
			n := cache.Position + l + 1
			first := 0
			if c.Context > 0 && n-c.Context > 0 {
				first = n - c.Context
			}
			scores := buffer[:n-first]

			qt := c.scratchProj[l*3*D+t*d : l*3*D+(t+1)*d]
			// Both halves go through the vector kernels rather than a Go loop.
			// The audio decoder attends over a window of two hundred and fifty
			// positions, sixteen times a frame in each of its two layers, and
			// written out termwise that was the single most expensive thing the
			// decoder did — more than every convolution together.
			for p := first; p < n; p++ {
				kp := cache.K[(p*h+t)*d : (p*h+t+1)*d]
				scores[p-first] = nn.DotF32(qt, kp) * scale
			}
			nn.SoftmaxInPlace(scores)

			head := c.scratchAttn[l*D+t*d : l*D+(t+1)*d]
			for i := range head {
				head[i] = 0
			}
			for p := first; p < n; p++ {
				vp := cache.V[(p*h+t)*d : (p*h+t+1)*d]
				nn.AxpyFull(head, vp, scores[p-first])
			}
		}
	})
}

// applyScale multiplies termwise, or does nothing if the layer has no scale.
// The audio decoder has one, the flow_lm does not.
func applyScale(scale, x []float32) {
	for i, e := range scale {
		x[i] *= e
	}
}

func (c *Layer) allocate(batch int) {
	if c.batchAlloc >= batch {
		return
	}
	D := c.DModel
	c.scratchProj = make([]float32, batch*3*D)
	c.scratchAttn = make([]float32, batch*D)
	c.scratchFF = make([]float32, batch*c.Linear1.Outputs)
	c.scratchIn = make([]float32, batch*D)
	c.scratchOut = make([]float32, batch*D)
	c.batchAlloc = batch
}

// Geometry describes a stack of identical layers.
type Geometry struct {
	DModel     int
	NumHeads   int
	DimFF      int
	NumLayers  int
	Context    int  // 0 = causal attention without limit
	LayerScale bool // whether the layers carry a scale factor
	MaxPeriod  float64
}

// LoadLayer reads a layer whose tensors are named `<prefix>layers.<index>.…`.
func LoadLayer(m *tensors.Model, basePrefix string, index int, g Geometry) (*Layer, error) {
	prefix := fmt.Sprintf("%slayers.%d.", basePrefix, index)
	dModel, numHeads, dimFF, maxPeriod := g.DModel, g.NumHeads, g.DimFF, g.MaxPeriod

	lin := func(name string, inputs, outputs int) (nn.Linear, error) {
		t, err := m.Get(prefix + name + ".weight")
		if err != nil {
			return nn.Linear{}, err
		}
		if len(t.Shape) != 2 || t.Shape[0] != outputs || t.Shape[1] != inputs {
			return nn.Linear{}, fmt.Errorf("%s: shape %v, want [%d %d]", name, t.Shape, outputs, inputs)
		}
		if t.DType != "BF16" {
			return nn.Linear{}, fmt.Errorf("%s: dtype %s, the kernel expects BF16", name, t.DType)
		}
		return nn.Linear{Weights: t.Raw, Inputs: inputs, Outputs: outputs}, nil
	}

	norm := func(name string) (nn.LayerNorm, error) {
		g, err := m.Get(prefix + name + ".weight")
		if err != nil {
			return nn.LayerNorm{}, err
		}
		b, err := m.Get(prefix + name + ".bias")
		if err != nil {
			return nn.LayerNorm{}, err
		}
		gain, err := g.F32()
		if err != nil {
			return nn.LayerNorm{}, err
		}
		bias, err := b.F32()
		if err != nil {
			return nn.LayerNorm{}, err
		}
		return nn.LayerNorm{Gain: gain, Bias: bias, Eps: 1e-5}, nil
	}

	c := &Layer{
		NumHeads: numHeads, HeadDim: dModel / numHeads,
		DModel: dModel, MaxPeriod: maxPeriod, Context: g.Context,
	}
	var err error
	if c.Norm1, err = norm("norm1"); err != nil {
		return nil, err
	}
	if c.Norm2, err = norm("norm2"); err != nil {
		return nil, err
	}
	if c.InProj, err = lin("self_attn.in_proj", dModel, 3*dModel); err != nil {
		return nil, err
	}
	if c.OutProj, err = lin("self_attn.out_proj", dModel, dModel); err != nil {
		return nil, err
	}
	if c.Linear1, err = lin("linear1", dModel, dimFF); err != nil {
		return nil, err
	}
	if c.Linear2, err = lin("linear2", dimFF, dModel); err != nil {
		return nil, err
	}
	if g.LayerScale {
		ld := Loader{M: m, Prefix: prefix}
		if c.Scale1, err = ld.Vector("layer_scale_1.scale"); err != nil {
			return nil, err
		}
		if c.Scale2, err = ld.Vector("layer_scale_2.scale"); err != nil {
			return nil, err
		}
	}
	return c, nil
}
