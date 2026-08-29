package tensors

// Writing a GGUF, which until now golem only read.
//
// A converter needs it: quantizing to a format of a different width changes
// every tensor's size, so the file cannot be patched in place the way a
// simulation can. The metadata is carried over value for value from the file it
// came from — a checkpoint's vocabulary, its rope base, its chat template — and
// only the tensors are new.

import (
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"os"
	"sort"
)

// OutTensor is one tensor to write: its bytes already in the target format.
type OutTensor struct {
	Name  string
	Shape []int // row length first, as GGUF orders them
	DType string
	Data  []byte
}

// WriteGGUF writes the metadata and tensors to path, aligned as GGUF v3 wants.
// meta is written in the order its keys sort, so the same input gives the same
// file.
func WriteGGUF(path string, meta map[string]any, tensors []OutTensor) (err error) {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer func() {
		if cerr := f.Close(); err == nil {
			err = cerr
		}
	}()

	w := &writer{}
	w.u32(ggufMagic)
	w.u32(3)
	w.u64(uint64(len(tensors)))
	w.u64(uint64(len(meta)))

	keys := make([]string, 0, len(meta))
	for k := range meta {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	for _, k := range keys {
		w.str(k)
		if err := w.value(meta[k]); err != nil {
			return fmt.Errorf("metadata %q: %w", k, err)
		}
	}

	const alignment = 32
	// The tensor table carries offsets into the data section, so the data
	// layout is decided before the table is written.
	offsets := make([]uint64, len(tensors))
	var at uint64
	for i, t := range tensors {
		offsets[i] = at
		at += uint64(len(t.Data))
		at = (at + alignment - 1) &^ (alignment - 1)
	}
	for i, t := range tensors {
		w.str(t.Name)
		w.u32(uint32(len(t.Shape)))
		for _, d := range t.Shape {
			w.u64(uint64(d))
		}
		kind, ok := typeNumbers[t.DType]
		if !ok {
			return fmt.Errorf("tensor %q: no type number for %s", t.Name, t.DType)
		}
		w.u32(kind)
		w.u64(offsets[i])
	}
	if w.err != nil {
		return w.err
	}

	pad := (len(w.buf) + alignment - 1) &^ (alignment - 1)
	w.buf = append(w.buf, make([]byte, pad-len(w.buf))...)
	if _, err := f.Write(w.buf); err != nil {
		return err
	}
	for i, t := range tensors {
		if _, err := f.Write(t.Data); err != nil {
			return err
		}
		next := uint64(len(t.Data))
		next = (next + alignment - 1) &^ (alignment - 1)
		if gap := int(next) - len(t.Data); gap > 0 {
			if _, err := f.Write(make([]byte, gap)); err != nil {
				return err
			}
		}
		_ = i
	}
	return nil
}

// typeNumbers is ggmlTypes read the other way round.
var typeNumbers = func() map[string]uint32 {
	m := make(map[string]uint32, len(ggmlTypes))
	for n, name := range ggmlTypes {
		m[name] = n
	}
	return m
}()

type writer struct {
	buf []byte
	err error
}

func (w *writer) u32(v uint32) { w.buf = binary.LittleEndian.AppendUint32(w.buf, v) }
func (w *writer) u64(v uint64) { w.buf = binary.LittleEndian.AppendUint64(w.buf, v) }

func (w *writer) str(s string) {
	w.u64(uint64(len(s)))
	w.buf = append(w.buf, s...)
}

// value writes one metadata value, tagged with the type its Go form implies.
// The reader's types and these are the same set, so a file read and written
// back comes out identical.
func (w *writer) value(v any) error {
	switch x := v.(type) {
	case uint8:
		w.u32(ggufUint8)
		w.buf = append(w.buf, x)
	case int8:
		w.u32(ggufInt8)
		w.buf = append(w.buf, byte(x))
	case uint16:
		w.u32(ggufUint16)
		w.buf = binary.LittleEndian.AppendUint16(w.buf, x)
	case int16:
		w.u32(ggufInt16)
		w.buf = binary.LittleEndian.AppendUint16(w.buf, uint16(x))
	case uint32:
		w.u32(ggufUint32)
		w.u32(x)
	case int32:
		w.u32(ggufInt32)
		w.u32(uint32(x))
	case float32:
		w.u32(ggufFloat32)
		w.u32(math.Float32bits(x))
	case bool:
		w.u32(ggufBool)
		b := byte(0)
		if x {
			b = 1
		}
		w.buf = append(w.buf, b)
	case string:
		w.u32(ggufString)
		w.str(x)
	case uint64:
		w.u32(ggufUint64)
		w.u64(x)
	case int64:
		w.u32(ggufInt64)
		w.u64(uint64(x))
	case float64:
		w.u32(ggufFloat64)
		w.u64(math.Float64bits(x))
	case []any:
		w.u32(ggufArray)
		if len(x) == 0 {
			// An empty array still has to declare what it is empty of, and the
			// reader threw that away. Nothing golem reads writes one.
			return fmt.Errorf("an empty array has no element type to write")
		}
		kind, err := typeOf(x[0])
		if err != nil {
			return err
		}
		w.u32(kind)
		w.u64(uint64(len(x)))
		for _, e := range x {
			// The element type is written once, so the elements go in bare.
			sub := &writer{}
			if err := sub.value(e); err != nil {
				return err
			}
			w.buf = append(w.buf, sub.buf[4:]...)
		}
	default:
		return fmt.Errorf("no GGUF type for %T", v)
	}
	return nil
}

func typeOf(v any) (uint32, error) {
	probe := &writer{}
	if err := probe.value(v); err != nil {
		return 0, err
	}
	return binary.LittleEndian.Uint32(probe.buf), nil
}

var _ io.Writer = (*os.File)(nil)
