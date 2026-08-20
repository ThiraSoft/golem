package tensors

// GGUF: the container llama.cpp writes. A header, a table of typed metadata, a
// table of tensor descriptions, then the tensor data, aligned.
//
// Only reading is implemented, and nothing is copied: every tensor ends up as a
// view over the mapping.

import (
	"encoding/binary"
	"fmt"
	"math"
)

const ggufMagic = 0x46554747 // "GGUF", little-endian

// The thirteen metadata value types, in the order the format numbers them.
const (
	ggufUint8 uint32 = iota
	ggufInt8
	ggufUint16
	ggufInt16
	ggufUint32
	ggufInt32
	ggufFloat32
	ggufBool
	ggufString
	ggufArray
	ggufUint64
	ggufInt64
	ggufFloat64
)

// GGUF is an opened file. Close releases the mapping; the tensor views become
// invalid at that point.
type GGUF struct {
	Meta    map[string]any
	Tensors map[string]Tensor

	m          *mapping
	dataOffset int
}

// reader walks the header without bounds-checking every field by hand.
type reader struct {
	buf []byte
	pos int
	err error
}

func (r *reader) take(n int) []byte {
	if r.err != nil {
		return nil
	}
	if n < 0 || r.pos+n > len(r.buf) {
		r.err = fmt.Errorf("truncated at byte %d", r.pos)
		return nil
	}
	b := r.buf[r.pos : r.pos+n]
	r.pos += n
	return b
}

func (r *reader) u32() uint32 {
	b := r.take(4)
	if b == nil {
		return 0
	}
	return binary.LittleEndian.Uint32(b)
}

func (r *reader) u64() uint64 {
	b := r.take(8)
	if b == nil {
		return 0
	}
	return binary.LittleEndian.Uint64(b)
}

func (r *reader) str() string {
	n := r.u64()
	if r.err != nil {
		return ""
	}
	if n > uint64(len(r.buf)) {
		r.err = fmt.Errorf("string of %d bytes at %d", n, r.pos)
		return ""
	}
	return string(r.take(int(n)))
}

// value reads one metadata value of the given type.
func (r *reader) value(kind uint32) any {
	switch kind {
	case ggufUint8:
		b := r.take(1)
		if b == nil {
			return nil
		}
		return b[0]
	case ggufInt8:
		b := r.take(1)
		if b == nil {
			return nil
		}
		return int8(b[0])
	case ggufUint16:
		b := r.take(2)
		if b == nil {
			return nil
		}
		return binary.LittleEndian.Uint16(b)
	case ggufInt16:
		b := r.take(2)
		if b == nil {
			return nil
		}
		return int16(binary.LittleEndian.Uint16(b))
	case ggufUint32:
		return r.u32()
	case ggufInt32:
		return int32(r.u32())
	case ggufFloat32:
		return math.Float32frombits(r.u32())
	case ggufBool:
		b := r.take(1)
		if b == nil {
			return nil
		}
		return b[0] != 0
	case ggufString:
		return r.str()
	case ggufArray:
		kind := r.u32()
		n := r.u64()
		if r.err != nil {
			return nil
		}
		if n > uint64(len(r.buf)) {
			r.err = fmt.Errorf("array of %d values at %d", n, r.pos)
			return nil
		}
		out := make([]any, 0, n)
		for i := uint64(0); i < n; i++ {
			v := r.value(kind)
			if r.err != nil {
				return nil
			}
			out = append(out, v)
		}
		return out
	case ggufUint64:
		return r.u64()
	case ggufInt64:
		return int64(r.u64())
	case ggufFloat64:
		return math.Float64frombits(r.u64())
	}
	r.err = fmt.Errorf("unknown metadata type %d at byte %d", kind, r.pos)
	return nil
}

// OpenGGUF maps a file and reads its header. The tensor data is not touched.
func OpenGGUF(path string) (*GGUF, error) {
	m, err := mapFile(path)
	if err != nil {
		return nil, err
	}
	g := &GGUF{m: m, Meta: map[string]any{}, Tensors: map[string]Tensor{}}
	if err := g.readHeader(); err != nil {
		m.Close()
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	return g, nil
}

func (g *GGUF) readHeader() error {
	r := &reader{buf: g.m.data}
	if r.u32() != ggufMagic {
		return fmt.Errorf("not a GGUF file")
	}
	if v := r.u32(); v != 2 && v != 3 {
		return fmt.Errorf("GGUF version %d, only 2 and 3 are read", v)
	}
	tensorCount := r.u64()
	metaCount := r.u64()
	if r.err != nil {
		return r.err
	}
	for i := uint64(0); i < metaCount; i++ {
		key := r.str()
		kind := r.u32()
		v := r.value(kind)
		if r.err != nil {
			return fmt.Errorf("metadata %d (%q): %w", i, key, r.err)
		}
		g.Meta[key] = v
	}
	return g.readTensorTable(r, tensorCount)
}

// ggmlTypes maps the type numbers used in the tensor table onto the names the
// rest of golem uses. Only the types this repository actually reads are listed:
// an unknown one is an error rather than a silent misreading.
var ggmlTypes = map[uint32]string{
	0:  "F32",
	1:  "F16",
	2:  "Q4_0",
	8:  "Q8_0",
	14: "Q6_K",
	30: "BF16",
}

// blockGeometry gives, per type, how many weights sit in one block and how many
// bytes that block occupies.
var blockGeometry = map[string][2]int{
	"F32":  {1, 4},
	"F16":  {1, 2},
	"BF16": {1, 2},
	"Q4_0": {32, 18},   // one fp16 scale, then 32 nibbles
	"Q8_0": {32, 34},   // one fp16 scale, then 32 signed bytes
	"Q6_K": {256, 210}, // 128 low nibbles, 64 high pairs, 16 scales, one fp16
}

// rowBytes is the size on disk of one row of `n` weights of the given type.
func rowBytes(dtype string, n int) (int, error) {
	g, ok := blockGeometry[dtype]
	if !ok {
		return 0, fmt.Errorf("no block geometry for %s", dtype)
	}
	if n%g[0] != 0 {
		return 0, fmt.Errorf("a row of %d does not divide into blocks of %d (%s)", n, g[0], dtype)
	}
	return n / g[0] * g[1], nil
}

func (g *GGUF) readTensorTable(r *reader, count uint64) error {
	type entry struct {
		name   string
		shape  []int
		dtype  string
		offset uint64
		bytes  int
	}
	entries := make([]entry, 0, count)

	for i := uint64(0); i < count; i++ {
		name := r.str()
		dims := r.u32()
		if r.err != nil {
			return fmt.Errorf("tensor %d: %w", i, r.err)
		}
		if dims == 0 || dims > 4 {
			return fmt.Errorf("tensor %q has %d dimensions", name, dims)
		}
		shape := make([]int, dims)
		elements := 1
		for d := range shape {
			shape[d] = int(r.u64())
			elements *= shape[d]
		}
		kind := r.u32()
		offset := r.u64()
		if r.err != nil {
			return fmt.Errorf("tensor %q: %w", name, r.err)
		}
		dtype, ok := ggmlTypes[kind]
		if !ok {
			return fmt.Errorf("tensor %q: unsupported ggml type %d", name, kind)
		}
		// shape[0] is the row length; everything above it counts rows.
		size, err := rowBytes(dtype, shape[0])
		if err != nil {
			return fmt.Errorf("tensor %q: %w", name, err)
		}
		entries = append(entries, entry{name, shape, dtype, offset, size * (elements / shape[0])})
	}

	// The data section begins at the next multiple of the alignment.
	alignment := 32
	if v, err := g.Uint32("general.alignment"); err == nil {
		alignment = int(v)
	}
	if alignment <= 0 || alignment&(alignment-1) != 0 {
		return fmt.Errorf("alignment %d is not a power of two", alignment)
	}
	g.dataOffset = (r.pos + alignment - 1) &^ (alignment - 1)

	for _, e := range entries {
		start := g.dataOffset + int(e.offset)
		end := start + e.bytes
		if start < 0 || end > len(g.m.data) || end < start {
			return fmt.Errorf("tensor %q runs from %d to %d, past the end of the file", e.name, start, end)
		}
		g.Tensors[e.name] = Tensor{
			Shape: e.shape,
			DType: e.dtype,
			Raw:   g.m.data[start:end],
		}
	}
	return nil
}

// Get returns a tensor by name.
func (g *GGUF) Get(name string) (Tensor, error) {
	t, ok := g.Tensors[name]
	if !ok {
		return Tensor{}, fmt.Errorf("tensor %q is absent", name)
	}
	return t, nil
}

// Close releases the mapping. Tensor views must not be used afterwards.
func (g *GGUF) Close() error { return g.m.Close() }
