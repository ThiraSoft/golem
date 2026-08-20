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
	"bufio"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"

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

// SaveVoice writes the state as a voice file, in the layout LoadVoice reads and
// the Python daemon wrote: one K/V cache and one offset per layer, in
// safetensors.
//
// The caches are stored at exactly the length they hold. The reference writes
// them at their allocated capacity and leaves the rest NaN, which LoadVoice
// guards against; there is no reason to reproduce a hazard that costs disk
// space.
func (t *Transformer) SaveVoice(path string, state *State) error {
	h, d := t.Config.NumHeads, t.Config.DModel/t.Config.NumHeads
	position := state.Position()
	if position == 0 {
		return fmt.Errorf("nothing listened to: the state is empty")
	}

	type entry struct {
		DType   string `json:"dtype"`
		Shape   []int  `json:"shape"`
		Offsets [2]int `json:"data_offsets"`
	}
	header := map[string]entry{}
	var body []byte

	add := func(name string, shape []int, raw []byte, dtype string) {
		header[name] = entry{DType: dtype, Shape: shape, Offsets: [2]int{len(body), len(body) + len(raw)}}
		body = append(body, raw...)
	}

	for i := range t.Layers {
		prefix := fmt.Sprintf("transformer.layers.%d.self_attn/", i)
		c := state.Caches[i]
		n := position * h * d

		raw := make([]byte, 0, 8*n)
		for _, half := range [][]float32{c.K[:n], c.V[:n]} {
			for _, v := range half {
				raw = binary.LittleEndian.AppendUint32(raw, math.Float32bits(v))
			}
		}
		add(prefix+"cache", []int{2, 1, position, h, d}, raw, "F32")

		offset := binary.LittleEndian.AppendUint64(nil, uint64(position))
		add(prefix+"offset", []int{1}, offset, "I64")
	}

	meta, err := json.Marshal(header)
	if err != nil {
		return err
	}
	// The header is padded to a multiple of eight: the format asks for it, and
	// a reader that maps the file wants its floats aligned.
	for len(meta)%8 != 0 {
		meta = append(meta, ' ')
	}

	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	w := bufio.NewWriter(f)
	if err := binary.Write(w, binary.LittleEndian, uint64(len(meta))); err != nil {
		return err
	}
	if _, err := w.Write(meta); err != nil {
		return err
	}
	if _, err := w.Write(body); err != nil {
		return err
	}
	if err := w.Flush(); err != nil {
		return err
	}
	return f.Close()
}
