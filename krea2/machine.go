package krea2

// What the three networks share on their way to the card: reading a
// checkpoint's tensors, gathering the small ones into the parameter buffer,
// and handing the large ones to vk.K2 one buffer each.

import (
	"encoding/binary"
	"fmt"
	"math"

	"github.com/ThiraSoft/golem/tensors"
	"github.com/ThiraSoft/golem/vk"
)

// e4m3 is what an fp8 e4m3fn byte stands for.
func e4m3(b byte) float32 {
	sign := float32(1)
	if b&0x80 != 0 {
		sign = -1
	}
	e, m := int(b>>3&15), float64(b&7)
	if e == 0 {
		return sign * float32(m/8*math.Exp2(-6))
	}
	return sign * float32((1+m/8)*math.Exp2(float64(e-7)))
}

// floats reads a tensor of any float type the checkpoints hold as float32.
func floats(t tensors.Tensor) ([]float32, error) {
	switch t.DType {
	case "F8_E4M3":
		out := make([]float32, len(t.Raw))
		for i, b := range t.Raw {
			out[i] = e4m3(b)
		}
		return out, nil
	case "F16":
		out := make([]float32, len(t.Raw)/2)
		for i := range out {
			out[i] = halfToFloat(binary.LittleEndian.Uint16(t.Raw[2*i:]))
		}
		return out, nil
	}
	return t.F32()
}

// params gathers the small tensors: norms, biases, modulation vectors. Each
// lands at an offset in floats, which is what a kernel is told.
type params struct {
	data []float32
}

func (p *params) add(v []float32) uint32 {
	at := len(p.data)
	p.data = append(p.data, v...)
	// Keep every vector on a sixteen-float boundary; nothing needs it, but a
	// misplaced offset then shows as garbage rather than as a shifted copy.
	for len(p.data)%16 != 0 {
		p.data = append(p.data, 0)
	}
	return uint32(at)
}

// checkpoint is an open safetensors file with a name prefix and the tensors
// that were asked for, so that a missing or misshapen one says which.
type checkpoint struct {
	m    *tensors.Model
	path string
}

func openCheckpoint(path string) (*checkpoint, error) {
	m, err := tensors.Open(path)
	if err != nil {
		return nil, fmt.Errorf("krea2: %s: %w", path, err)
	}
	return &checkpoint{m: m, path: path}, nil
}

func (c *checkpoint) get(name string, dtype string, shape ...int) (tensors.Tensor, error) {
	t, err := c.m.Get(name)
	if err != nil {
		return t, fmt.Errorf("krea2: %s: %w", c.path, err)
	}
	if dtype != "" && t.DType != dtype {
		return t, fmt.Errorf("krea2: %s: %s is %s, want %s", c.path, name, t.DType, dtype)
	}
	if len(shape) > 0 {
		ok := len(shape) == len(t.Shape)
		for i := 0; ok && i < len(shape); i++ {
			ok = shape[i] == t.Shape[i]
		}
		if !ok {
			return t, fmt.Errorf("krea2: %s: %s has shape %v, want %v", c.path, name, t.Shape, shape)
		}
	}
	return t, nil
}

func (c *checkpoint) floats(name string, n int) ([]float32, error) {
	t, err := c.get(name, "")
	if err != nil {
		return nil, err
	}
	if t.Elems() != n {
		return nil, fmt.Errorf("krea2: %s: %s has %d values, want %d", c.path, name, t.Elems(), n)
	}
	return floats(t)
}

func (c *checkpoint) scalar(name string) (float32, error) {
	v, err := c.floats(name, 1)
	if err != nil {
		return 0, err
	}
	return v[0], nil
}

// fp8 uploads an fp8 matrix of rows × cols.
func (c *checkpoint) fp8(k *vk.K2, name string, rows, cols int) (int, error) {
	t, err := c.get(name, "F8_E4M3", rows, cols)
	if err != nil {
		return 0, err
	}
	return k.AddWeights(t.Raw)
}

func (c *checkpoint) close() { c.m.Close() }

func halfToFloat(h uint16) float32 {
	sign := uint32(h&0x8000) << 16
	exp := uint32(h >> 10 & 0x1f)
	mant := uint32(h & 0x3ff)
	switch {
	case exp == 0:
		v := float32(mant) * float32(math.Exp2(-24))
		if sign != 0 {
			v = -v
		}
		return v
	case exp == 31:
		return math.Float32frombits(sign | 0x7f800000 | mant<<13)
	}
	return math.Float32frombits(sign | (exp+112)<<23 | mant<<13)
}

// floatToHalf rounds to the nearest fp16, ties to even.
func floatToHalf(f float32) uint16 {
	b := math.Float32bits(f)
	sign := uint16(b>>16) & 0x8000
	exp := int(b>>23&0xff) - 127 + 15
	mant := b & 0x7fffff
	switch {
	case exp >= 31:
		return sign | 0x7c00
	case exp <= 0:
		if exp < -10 {
			return sign
		}
		mant |= 0x800000
		shift := uint(14 - exp)
		half := mant >> shift
		rem := mant & (1<<shift - 1)
		if rem > 1<<(shift-1) || (rem == 1<<(shift-1) && half&1 == 1) {
			half++
		}
		return sign | uint16(half)
	}
	half := uint32(exp)<<10 | mant>>13
	rem := mant & 0x1fff
	if rem > 0x1000 || (rem == 0x1000 && half&1 == 1) {
		half++
	}
	return sign | uint16(half)
}

// arena hands out regions of the card's arena, in floats, each on a
// sixty-four-float boundary.
type arena struct{ n int }

func (a *arena) take(n int) uint32 {
	at := a.n
	a.n += (n + 63) / 64 * 64
	return uint32(at)
}
