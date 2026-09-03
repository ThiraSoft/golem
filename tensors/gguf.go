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
	"sort"
	"strings"
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

// The tensor types golem writes for its own formats. The base is ASCII "glm"
// in the top three bytes and the bits a weight in the low one, so `67 6C 6D 03`
// in a hex dump names both the format and the tier to a reader with no tooling,
// and a sixth tier is 0x676C6D06 and needs no decision. ggml's own types run
// 0-39 and grow over time; nothing will ever be allocated up here.
//
// This is only the second of three layers. A type number lives in somebody
// else's enum and cannot be a safe discriminator on its own — see
// checkGolemFile, which is what actually decides that a file is a golem file.
const (
	golemTypeBase uint32 = 0x676C6D00 // "glm\0"
	golemT3G      uint32 = golemTypeBase | 3
	golemT4G      uint32 = golemTypeBase | 4
	golemT5G      uint32 = golemTypeBase | 5
)

// GolemFormat is the value of the `golem.format` key: the codec family and the
// layout version. It is what makes a file a golem file — not the tensor type,
// which is per tensor and borrowed. A reader that meets a private tensor type
// without this key refuses the file rather than guessing which era wrote it.
//
// Bump the version when the bytes of a block change meaning. The name changes
// when the codec family does.
const GolemFormat = "trellis/1"

// ggmlTypes maps the type numbers used in the tensor table onto the names the
// rest of golem uses. Only the types this repository actually reads are listed:
// an unknown one is an error rather than a silent misreading.
var ggmlTypes = map[uint32]string{
	0:  "F32",
	1:  "F16",
	2:  "Q4_0",
	3:  "Q4_1",
	8:  "Q8_0",
	10: "Q2_K",
	11: "Q3_K",
	12: "Q4_K",
	13: "Q5_K",
	14: "Q6_K",
	30: "BF16",
	// golem's own, which llama.cpp will not recognise and is not meant to.
	golemT3G: "T3G",
	golemT4G: "T4G",
	golemT5G: "T5G",
	// 1000-1004 are retired and never reused. They were, in turn, the lattice,
	// Lloyd, the wide lattice, T4G and T5G, and then 1000-1002 were the three
	// trellis tiers. The collision that ended that numbering is the reason for
	// everything above: 1000 meant D4G, 26 bytes per 64 weights, and then T3G,
	// 52 per 128 — the same 3.25 bits a weight, so a lattice-era file passed
	// the type lookup, passed the row-size check, and decoded through the
	// trellis hash with its rotation silently read as absent. An integer in
	// another project's enum can collide with its own past as easily as with
	// its owner's future.
}

// The geometry this build implements. It is a second copy of nn's constants
// because tensors cannot import nn — nn's own tests read GGUFs — and
// TestGolemBlockGeometryMatchesNN is the join that holds the two equal.
const (
	golemSeq        = 128 // weights coded as one trellis path
	golemState      = 12  // bits of state the path carries
	golemScaleBlock = 64  // weights under one step code, a byte each
)

// golemTierBits is the bits a weight each private type codes at, which is by
// construction the low byte of its type number.
var golemTierBits = map[string]int{"T3G": 3, "T4G": 4, "T5G": 5}

// golemBlockBytes is what a block of seq weights occupies at k bits each with
// a state of the given width: one step code per scale block, then the path,
// which is the state once and k bits for every weight after the first.
func golemBlockBytes(seq, state, k int) int {
	return seq/golemScaleBlock + ((seq-1)*k+state+7)/8
}

// checkGolemFile decides whether a file carrying golem's private tensor types
// is a file this build can read. It is the third layer, and the one that does
// the deciding: the type numbers say which tier a tensor is, and this says
// whether the file means what those numbers mean here.
//
// It refuses four things a type number and a row size cannot catch:
//
//   - a `golem.d4.*` key, which only the lattice-era converter wrote. That era
//     is numerically indistinguishable from the trellis at three bits, so the
//     retired key is the only evidence left;
//   - a private tensor type with no `golem.format`, which is a file written
//     before the key existed. A missing key cannot be read as agreement;
//   - a `golem.format` this build does not implement;
//   - a declared geometry that disagrees with what this build codes, or that
//     names a body tier no tensor in the file uses.
func (g *GGUF) checkGolemFile() error {
	for key := range g.Meta {
		if strings.HasPrefix(key, "golem.d4.") {
			return fmt.Errorf("gguf: carries %q, which the lattice-era converter wrote; this file predates the trellis format and has to be rebuilt with golemquant", key)
		}
	}
	tiers := map[string]bool{}
	for _, t := range g.Tensors {
		if _, ok := golemTierBits[t.DType]; ok {
			tiers[t.DType] = true
		}
	}
	if len(tiers) == 0 {
		return nil
	}

	format, err := g.String("golem.format")
	if err != nil {
		return fmt.Errorf("gguf: golem's own tensor types with no golem.format key; the file predates the key and has to be rebuilt with golemquant")
	}
	if format != GolemFormat {
		return fmt.Errorf("gguf: golem.format is %q and this build reads %q", format, GolemFormat)
	}

	seq, err := g.Uint32("golem.trellis.seq")
	if err != nil {
		return fmt.Errorf("gguf: %s file with no golem.trellis.seq: %w", format, err)
	}
	if int(seq) != golemSeq {
		return fmt.Errorf("gguf: the file codes %d weights as one path and this build codes %d", seq, golemSeq)
	}
	state, err := g.Uint32("golem.trellis.state")
	if err != nil {
		return fmt.Errorf("gguf: %s file with no golem.trellis.state: %w", format, err)
	}
	if int(state) != golemState {
		return fmt.Errorf("gguf: the file carries %d bits of trellis state and this build carries %d", state, golemState)
	}

	bits, err := g.Uint32("golem.trellis.bits")
	if err != nil {
		return fmt.Errorf("gguf: %s file with no golem.trellis.bits to say which tier the body is: %w", format, err)
	}
	bodySeen := false
	names := make([]string, 0, len(tiers))
	for name := range tiers {
		names = append(names, name)
	}
	sort.Strings(names)
	for _, name := range names {
		k := golemTierBits[name]
		if k == int(bits) {
			bodySeen = true
		}
		// The declared geometry has to imply the block size the tensor type
		// claims. Both sides are already pinned above, so this only fires when
		// blockGeometry drifts from the arithmetic that produced it — which is
		// the failure that reads every row at the wrong offset.
		if want, got := blockGeometry[name][1], golemBlockBytes(int(seq), int(state), k); want != got {
			return fmt.Errorf("gguf: %s reads %d bytes a block and a %d-weight path of %d bits with %d of state is %d", name, want, seq, k, state, got)
		}
	}
	if !bodySeen {
		return fmt.Errorf("gguf: golem.trellis.bits says the body is %d bits and no tensor in the file is that tier (%s)", bits, strings.Join(names, ", "))
	}
	return nil
}

// blockGeometry gives, per type, how many weights sit in one block and how many
// bytes that block occupies.
var blockGeometry = map[string][2]int{
	"F32":  {1, 4},
	"F16":  {1, 2},
	"BF16": {1, 2},
	"Q4_0": {32, 18},   // one fp16 scale, then 32 nibbles
	"Q4_1": {32, 20},   // an fp16 scale and an fp16 minimum, then 32 nibbles
	"Q8_0": {32, 34},   // one fp16 scale, then 32 signed bytes
	"Q2_K": {256, 84},  // 16 packed scale-and-minimum bytes, two-bit quants, 2 fp16
	"Q3_K": {256, 110}, // hmask, two-bit quants, twelve packed scales, one fp16
	"Q4_K": {256, 144}, // 2 fp16 (d, dmin) + 12 scales + 128 nibbles
	"Q5_K": {256, 176}, // 2 fp16 (d, dmin) + 12 scales + 32 high bits + 128 nibbles
	"Q6_K": {256, 210}, // 128 low nibbles, 64 high pairs, 16 scales, one fp16
	"T3G":  {128, 52},  // two step codes, then a 393-bit path in 400
	"T4G":  {128, 67},  // two step codes, then a 520-bit trellis path
	"T5G":  {128, 83},  // the same at five bits a weight, 648 of them
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
			// The two ways a number can be golem's and still unreadable are
			// worth naming, because "unsupported ggml type 1000" sends the
			// reader to look for a missing decoder rather than to rebuild.
			if kind >= 1000 && kind <= 1004 {
				return fmt.Errorf("tensor %q: ggml type %d is one of golem's retired numbers; this file predates the move to %#x and has to be rebuilt with golemquant", name, kind, golemTypeBase)
			}
			if kind>>8 == golemTypeBase>>8 {
				return fmt.Errorf("tensor %q: %#x is a golem type at %d bits a weight, which this build does not implement", name, kind, kind&0xFF)
			}
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
			Shape:  e.shape,
			DType:  e.dtype,
			Raw:    g.m.data[start:end],
			Offset: start,
		}
	}
	return g.checkGolemFile()
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
