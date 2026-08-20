package tensors

// Typed access to the metadata. The GGUF format stores a value in whichever
// integer width fits, and a key that is a scalar in one model is an array in
// another — `head_count_kv` is a single value in Gemma E2B and forty-eight of
// them in the 12B. These accessors absorb both, so that no engine has to.

import "fmt"

func (g *GGUF) raw(key string) (any, error) {
	v, ok := g.Meta[key]
	if !ok {
		return nil, fmt.Errorf("metadata %q is absent", key)
	}
	return v, nil
}

func (g *GGUF) String(key string) (string, error) {
	v, err := g.raw(key)
	if err != nil {
		return "", err
	}
	s, ok := v.(string)
	if !ok {
		return "", fmt.Errorf("metadata %q is %T, not a string", key, v)
	}
	return s, nil
}

func (g *GGUF) Bool(key string) (bool, error) {
	v, err := g.raw(key)
	if err != nil {
		return false, err
	}
	b, ok := v.(bool)
	if !ok {
		return false, fmt.Errorf("metadata %q is %T, not a bool", key, v)
	}
	return b, nil
}

func (g *GGUF) Float32(key string) (float32, error) {
	v, err := g.raw(key)
	if err != nil {
		return 0, err
	}
	switch n := v.(type) {
	case float32:
		return n, nil
	case float64:
		return float32(n), nil
	}
	return 0, fmt.Errorf("metadata %q is %T, not a float", key, v)
}

// asUint32 widens any of the integer types the format may have used.
func asUint32(key string, v any) (uint32, error) {
	switch n := v.(type) {
	case uint8:
		return uint32(n), nil
	case uint16:
		return uint32(n), nil
	case uint32:
		return n, nil
	case uint64:
		return uint32(n), nil
	case int8:
		return uint32(n), nil
	case int16:
		return uint32(n), nil
	case int32:
		return uint32(n), nil
	case int64:
		return uint32(n), nil
	}
	return 0, fmt.Errorf("metadata %q is %T, not an integer", key, v)
}

func (g *GGUF) Uint32(key string) (uint32, error) {
	v, err := g.raw(key)
	if err != nil {
		return 0, err
	}
	return asUint32(key, v)
}

// Uint32Slice reads an array, or a scalar as an array of one.
func (g *GGUF) Uint32Slice(key string) ([]uint32, error) {
	v, err := g.raw(key)
	if err != nil {
		return nil, err
	}
	items, ok := v.([]any)
	if !ok {
		n, err := asUint32(key, v)
		if err != nil {
			return nil, err
		}
		return []uint32{n}, nil
	}
	out := make([]uint32, len(items))
	for i, item := range items {
		if out[i], err = asUint32(key, item); err != nil {
			return nil, err
		}
	}
	return out, nil
}

func (g *GGUF) BoolSlice(key string) ([]bool, error) {
	v, err := g.raw(key)
	if err != nil {
		return nil, err
	}
	items, ok := v.([]any)
	if !ok {
		b, ok := v.(bool)
		if !ok {
			return nil, fmt.Errorf("metadata %q is %T, not a bool", key, v)
		}
		return []bool{b}, nil
	}
	out := make([]bool, len(items))
	for i, item := range items {
		b, ok := item.(bool)
		if !ok {
			return nil, fmt.Errorf("metadata %q holds a %T, not a bool", key, item)
		}
		out[i] = b
	}
	return out, nil
}

func (g *GGUF) Strings(key string) ([]string, error) {
	v, err := g.raw(key)
	if err != nil {
		return nil, err
	}
	items, ok := v.([]any)
	if !ok {
		return nil, fmt.Errorf("metadata %q is %T, not an array", key, v)
	}
	out := make([]string, len(items))
	for i, item := range items {
		s, ok := item.(string)
		if !ok {
			return nil, fmt.Errorf("metadata %q holds a %T, not a string", key, item)
		}
		out[i] = s
	}
	return out, nil
}
