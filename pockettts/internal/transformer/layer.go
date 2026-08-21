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
	// x is the residual itself: nothing writes it until the section that adds
	// the block's output to it, and that section writes only the rows it read.
	// The norm works on a copy, which is what scratchOut is for.
	copy(c.scratchOut, x[:batch*D])
	for l := 0; l < batch; l++ {
		c.Norm1.Apply(c.scratchOut[l*D : (l+1)*D])
	}
	// The query, key and value projections, the rotation and the cache, in one
	// section. What follows a projection here is not one pass over its outputs
	// but three — rotate, copy the keys, copy the values — and on one core they
	// cost as much again as the product did.
	//
	// The section is cut into heads rather than into rows, because that is the
	// unit the rotation needs whole: slot s below is the s-th head of the query,
	// then of the key, then of the value, and its rows are its own. The
	// projections of the `batch` positions are independent; the attention that
	// follows is not, which is why every key and value is in the cache before
	// any score is computed.
	slots := 3 * h
	nn.InParallel(slots, batch*3*D*c.InProj.Inputs, func(first, last int) {
		c.InProj.ApplyRows(c.scratchOut, c.scratchProj, batch, first*d, last*d)
		for s := first; s < last; s++ {
			for l := 0; l < batch; l++ {
				proj := c.scratchProj[l*3*D : (l+1)*3*D]
				position := cache.Position + l
				head := proj[s*d : (s+1)*d]
				switch {
				case s < h: // a query head: rotated, and read where it lies
					nn.ApplyRoPE(head, position, c.MaxPeriod)
				case s < 2*h: // a key head: rotated, then kept
					nn.ApplyRoPE(head, position, c.MaxPeriod)
					base := position*h*d + (s-h)*d
					copy(cache.K[base:base+d], head)
				default: // a value head: kept as it is
					base := position*h*d + (s-2*h)*d
					copy(cache.V[base:base+d], head)
				}
			}
		}
	})

	c.attention(batch, cache)
	cache.Position += batch

	// The output projection, its scale and the residual in one section. The
	// three are one pass over the same rows, and the two that follow the
	// product used to be a pass on one core with the others waiting at the
	// barrier — a barrier the section had to reach first.
	c.finish(c.OutProj, c.scratchAttn, c.Scale1, x, batch, true)

	// --- feed-forward block ---
	// The residual and the norm's input are already copies of x: the section
	// that wrote x wrote them too, which is two passes over the width that no
	// longer happen while seven cores wait.
	for l := 0; l < batch; l++ {
		c.Norm2.Apply(c.scratchOut[l*D : (l+1)*D])
	}
	// The first projection activates what it computed before it lets go: GELU
	// over four thousand values is not worth a section of its own, and it was
	// paying for one.
	wide := c.Linear1.Outputs
	nn.InParallel(wide, batch*wide*(c.Linear1.Inputs+gelWork), func(start, end int) {
		c.Linear1.ApplyRows(c.scratchOut, c.scratchFF, batch, start, end)
		for l := 0; l < batch; l++ {
			nn.GELURange(c.scratchFF[l*wide:(l+1)*wide], start, end)
		}
	})
	c.finish(c.Linear2, c.scratchFF, c.Scale2, x, batch, false)
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
// gelWork is what one GELU costs beside the multiply-add the section's weight
// is counted in. It only decides whether a section is worth splitting.
const gelWork = 128

// finish computes a projection into the scratch, scales it and adds it to the
// residual, all inside one section. The three walk the same rows, so a worker
// that has computed rows [start, end) has everything it needs to finish them.
func (c *Layer) finish(l nn.Linear, in, scale, x []float32, batch int, again bool) {
	D := c.DModel
	nn.InParallel(D, batch*D*l.Inputs, func(start, end int) {
		l.ApplyRows(in, c.scratchProj, batch, start, end)
		for p := 0; p < batch; p++ {
			out := c.scratchProj[p*D : (p+1)*D]
			dst := x[p*D : (p+1)*D]
			for i := start; i < end; i++ {
				v := out[i]
				if scale != nil { // the layer may carry none
					v *= scale[i]
				}
				dst[i] += v
			}
			// The copy the next half's norm works on, made where the values
			// are still in this core's cache.
			if again {
				copy(c.scratchOut[p*D+start:p*D+end], dst[start:end])
			}
		}
	})
}

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
