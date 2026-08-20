package flowlm

// Loading a precomputed voice state.
//
// Cloning a voice requires encoding a reference WAV with the Mimi encoder — half
// the model, for work that happens only once per voice. The Python daemon
// already does it and caches the result on disk, in safetensors format: the K/V
// caches of the 24 layers, exactly as the transformer left them after listening
// to the reference. The Go engine reads them back and starts with the voice
// already in memory.

import (
	"encoding/binary"
	"fmt"
	"math"

	"github.com/ThiraSoft/golem/tensors"
)

// LoadVoice returns a state initialized from the file's caches, sized for
// `capacity` positions in total.
func (t *Transformer) LoadVoice(path string, capacity int) (*State, error) {
	v, err := tensors.Open(path)
	if err != nil {
		return nil, err
	}
	defer v.Close()

	h, d := t.Config.NumHeads, t.Config.DModel/t.Config.NumHeads
	state := t.NewState(capacity)

	for i := range t.Layers {
		prefix := fmt.Sprintf("transformer.layers.%d.self_attn/", i)

		offset, err := v.Get(prefix + "offset")
		if err != nil {
			return nil, err
		}
		position := int(binary.LittleEndian.Uint64(offset.Raw[:8]))

		cache, err := v.Get(prefix + "cache")
		if err != nil {
			return nil, err
		}
		// shape [2, 1, positions, heads, dim]: K then V, laid out exactly like
		// the Go cache, up to the slicing.
		if len(cache.Shape) != 5 || cache.Shape[0] != 2 || cache.Shape[3] != h || cache.Shape[4] != d {
			return nil, fmt.Errorf("%scache: unexpected shape %v", prefix, cache.Shape)
		}
		stored := cache.Shape[2]
		if position > stored {
			return nil, fmt.Errorf("%s: offset %d beyond the %d stored positions", prefix, position, stored)
		}
		if position > capacity {
			return nil, fmt.Errorf("voice of %d positions, capacity %d", position, capacity)
		}

		values, err := cache.F32()
		if err != nil {
			return nil, err
		}
		perBlock := stored * h * d
		c := state.Caches[i]
		copy(c.K, values[:position*h*d])
		copy(c.V, values[perBlock:perBlock+position*h*d])
		c.Position = position

		// The unwritten positions are NaN on the Python side; if they have
		// leaked into the useful part, attention will produce silence without
		// saying so.
		for _, x := range c.K[:position*h*d] {
			if math.IsNaN(float64(x)) {
				return nil, fmt.Errorf("%s: NaN within the %d useful positions", prefix, position)
			}
		}
	}
	return state, nil
}
