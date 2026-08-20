package tensors

// Reading a safetensors file: a JSON header describing every tensor (dtype,
// shape, bounds within the blob), followed by the raw data. The file is mapped
// rather than read: there is no reason to copy 672 MB, and the kernel will only
// fault in the pages actually touched.

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"syscall"
)

// Tensor describes one tensor of the file and points at its bytes, without
// copying them.
type Tensor struct {
	Name  string
	DType string
	Shape []int
	Raw   []byte
}

// Elems is the number of scalars in the tensor.
func (t Tensor) Elems() int {
	n := 1
	for _, d := range t.Shape {
		n *= d
	}
	return n
}

// F32 converts the tensor to float32. The weights are bfloat16: exactly the
// high 16 bits of a float32, so the conversion is a shift — no loss, no
// rounding.
func (t Tensor) F32() ([]float32, error) {
	out := make([]float32, t.Elems())
	switch t.DType {
	case "BF16":
		for i := range out {
			bits := uint32(binary.LittleEndian.Uint16(t.Raw[i*2:])) << 16
			out[i] = math.Float32frombits(bits)
		}
	case "F32":
		for i := range out {
			out[i] = math.Float32frombits(binary.LittleEndian.Uint32(t.Raw[i*4:]))
		}
	default:
		return nil, fmt.Errorf("unsupported dtype %q", t.DType)
	}
	return out, nil
}

// Model is an open safetensors file, indexed by tensor name.
type Model struct {
	tensors map[string]Tensor
	mapped  []byte
	file    *os.File
}

// header is the JSON description of one tensor in the file header.
type header struct {
	DType   string `json:"dtype"`
	Shape   []int  `json:"shape"`
	Offsets [2]int `json:"data_offsets"`
}

// Open maps the file into memory and reads its header.
func Open(path string) (*Model, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	info, err := f.Stat()
	if err != nil {
		f.Close()
		return nil, err
	}
	data, err := syscall.Mmap(int(f.Fd()), 0, int(info.Size()),
		syscall.PROT_READ, syscall.MAP_SHARED)
	if err != nil {
		f.Close()
		return nil, fmt.Errorf("mmap: %w", err)
	}

	size := binary.LittleEndian.Uint64(data[:8])
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(data[8:8+size], &raw); err != nil {
		syscall.Munmap(data)
		f.Close()
		return nil, fmt.Errorf("unreadable header: %w", err)
	}

	start := 8 + int(size) // the data follows the header immediately
	m := &Model{tensors: make(map[string]Tensor, len(raw)), mapped: data, file: f}
	for name, value := range raw {
		if name == "__metadata__" {
			continue
		}
		var h header
		if err := json.Unmarshal(value, &h); err != nil {
			continue
		}
		m.tensors[name] = Tensor{
			Name:  name,
			DType: h.DType,
			Shape: h.Shape,
			Raw:   data[start+h.Offsets[0] : start+h.Offsets[1]],
		}
	}
	return m, nil
}

// Get returns a tensor by name.
func (m *Model) Get(name string) (Tensor, error) {
	t, ok := m.tensors[name]
	if !ok {
		return Tensor{}, fmt.Errorf("missing tensor %q", name)
	}
	return t, nil
}

// Names returns every tensor name.
func (m *Model) Names() []string {
	names := make([]string, 0, len(m.tensors))
	for name := range m.tensors {
		names = append(names, name)
	}
	return names
}

// Bytes returns the total size of the tensor data.
func (m *Model) Bytes() int {
	n := 0
	for _, t := range m.tensors {
		n += len(t.Raw)
	}
	return n
}

func (m *Model) Close() error {
	syscall.Munmap(m.mapped)
	return m.file.Close()
}
