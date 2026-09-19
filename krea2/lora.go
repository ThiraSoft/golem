package krea2

// A LoRA: for some of the DiT's products W, a change s·B·A of rank r, B
// out × r and A r × in, which the mobile front applies with ComfyUI's
// LoraLoaderModelOnly at a strength s from 0 to 2.
//
// ComfyUI folds the change into the weights, two ways according to where a
// module sits. A module it holds on the card is patched once: the fp8
// weight widened to fp16, plus s·(B·A) computed in float32, rounded back to
// fp8 stochastically (comfy.float.stochastic_rounding, seeded with the
// weight's name). A change that small is mostly under an fp8 step, so the
// rounding keeps it on average and adds a noise larger, weight by weight,
// than the change. A module it left in system memory is patched in bf16 at
// every call, as it is cast, which is the arithmetic to 0.06% of a DiT call.
// Which modules are which depends on the card's free memory at the time:
// recorded here, 210 of the 256 changed weights were patched as cast.
//
// golem keeps the fp8 weights as they are and adds s·B·(A·x) to each
// product's answer, which is what both ways stand for and what ComfyUI's
// bypass loader computes. A and B are small beside W (rank 32 is half a
// percent of a block), so the card holds both, and another LoRA is a few
// hundred megabytes to upload and another strength nothing, where folding
// would read the DiT's twelve gigabytes again.

import (
	"encoding/binary"
	"fmt"
	"math"
	"path/filepath"
	"strings"

	"github.com/ThiraSoft/golem/tensors"
	"github.com/ThiraSoft/golem/vk"
)

// LoRA is a LoRA read into memory, its matrices as fp16.
type LoRA struct {
	Name   string
	path   string
	deltas map[string]*loraDelta // by the DiT weight's name, "blocks.0.attn.wq"
}

// loraDelta is one product's change, its rank padded to a multiple of 32
// with zeros, which is what the products' kernel steps through.
type loraDelta struct {
	out, in, rank int
	a, b          []uint16 // A rank × in, B out × rank, fp16
	alpha         float32  // alpha / rank when the file gives an alpha, else 1
}

// loraRankStep is what a rank is padded to.
const loraRankStep = 32

// OpenLoRA reads a LoRA of the DiT: PEFT's lora_A and lora_B, or kohya's
// lora_down and lora_up, with or without an alpha, named after the DiT's
// weights with or without ComfyUI's "diffusion_model." in front. fp16, bf16
// or float32. Anything else in the file is refused by name rather than
// skipped: a LoRA half applied looks like a LoRA that does little.
func OpenLoRA(path string) (*LoRA, error) {
	m, err := tensors.Open(path)
	if err != nil {
		return nil, fmt.Errorf("krea2: LoRA %s: %w", path, err)
	}
	defer m.Close()
	l := &LoRA{Name: strings.TrimSuffix(filepath.Base(path), ".safetensors"), path: path, deltas: map[string]*loraDelta{}}
	type parts struct{ a, b, alpha *tensors.Tensor }
	found := map[string]*parts{}
	for _, name := range m.Names() {
		t, err := m.Get(name)
		if err != nil {
			return nil, err
		}
		key := strings.TrimPrefix(name, "diffusion_model.")
		var target string
		var slot **tensors.Tensor
		p := func(suffix string) bool {
			if !strings.HasSuffix(key, suffix) {
				return false
			}
			target = strings.TrimSuffix(key, suffix)
			if found[target] == nil {
				found[target] = &parts{}
			}
			return true
		}
		switch {
		case p(".lora_A.weight"), p(".lora_down.weight"):
			slot = &found[target].a
		case p(".lora_B.weight"), p(".lora_up.weight"):
			slot = &found[target].b
		case p(".alpha"):
			slot = &found[target].alpha
		default:
			return nil, fmt.Errorf("krea2: LoRA %s: %s is not a part of a LoRA this reads", path, name)
		}
		if *slot != nil {
			return nil, fmt.Errorf("krea2: LoRA %s: %s given twice", path, name)
		}
		tt := t
		*slot = &tt
	}
	for target, p := range found {
		if p.a == nil || p.b == nil {
			return nil, fmt.Errorf("krea2: LoRA %s: %s has one of its two matrices", path, target)
		}
		if len(p.a.Shape) != 2 || len(p.b.Shape) != 2 || p.a.Shape[0] != p.b.Shape[1] {
			return nil, fmt.Errorf("krea2: LoRA %s: %s is %v by %v", path, target, p.b.Shape, p.a.Shape)
		}
		d := &loraDelta{out: p.b.Shape[0], in: p.a.Shape[1], alpha: 1}
		r := p.a.Shape[0]
		d.rank = (r + loraRankStep - 1) / loraRankStep * loraRankStep
		a, err := halves(*p.a)
		if err != nil {
			return nil, fmt.Errorf("krea2: LoRA %s: %s: %w", path, target, err)
		}
		b, err := halves(*p.b)
		if err != nil {
			return nil, fmt.Errorf("krea2: LoRA %s: %s: %w", path, target, err)
		}
		// A's rows are the rank, and padding them is zeros at the end; B's
		// columns are, so each of its rows is padded.
		d.a = make([]uint16, d.rank*d.in)
		copy(d.a, a)
		d.b = make([]uint16, d.out*d.rank)
		for o := 0; o < d.out; o++ {
			copy(d.b[o*d.rank:o*d.rank+r], b[o*r:(o+1)*r])
		}
		if p.alpha != nil {
			v, err := floats(*p.alpha)
			if err != nil || len(v) != 1 {
				return nil, fmt.Errorf("krea2: LoRA %s: %s.alpha is not one number", path, target)
			}
			d.alpha = v[0] / float32(r)
		}
		l.deltas[target] = d
	}
	if len(l.deltas) == 0 {
		return nil, fmt.Errorf("krea2: LoRA %s changes nothing", path)
	}
	return l, nil
}

// Rank is the largest rank the LoRA has.
func (l *LoRA) Rank() int {
	r := 0
	for _, d := range l.deltas {
		r = max(r, d.rank)
	}
	return r
}

// Bytes is what the LoRA holds on the card.
func (l *LoRA) Bytes() int {
	n := 0
	for _, d := range l.deltas {
		n += 2 * (len(d.a) + len(d.b))
	}
	return n
}

// halves reads a tensor as fp16. bf16 and float32 are rounded to it: a LoRA's
// values sit well inside fp16's range, and ComfyUI computes the change from
// them in fp16 or bf16 as well.
func halves(t tensors.Tensor) ([]uint16, error) {
	switch t.DType {
	case "F16":
		out := make([]uint16, t.Elems())
		for i := range out {
			out[i] = binary.LittleEndian.Uint16(t.Raw[2*i:])
		}
		return out, nil
	case "BF16", "F32":
		v, err := t.F32()
		if err != nil {
			return nil, err
		}
		out := make([]uint16, len(v))
		for i, f := range v {
			if math.IsInf(float64(f), 0) || math.IsNaN(float64(f)) || math.Abs(float64(f)) > 65504 {
				return nil, fmt.Errorf("a value of %g, past fp16", f)
			}
			out[i] = floatToHalf(f)
		}
		return out, nil
	}
	return nil, fmt.Errorf("%s, not fp16, bf16 or float32", t.DType)
}

func halfBytes(h []uint16) []byte {
	out := make([]byte, 2*len(h))
	for i, v := range h {
		binary.LittleEndian.PutUint16(out[2*i:], v)
	}
	return out
}

// On the card. A product with a change runs as one: the kernel goes on
// from W's columns to B's, against t = strength · A·x rather than x, in the
// same sums (shaders/krea2_mm.comp). t is a product of its own before it.
// The products that read the same input (a block's four attention products,
// its two MLP products going up) share it, their A's stacked, one row of the
// stack a row of t. Every B is in one buffer, alpha folded in, so that the
// strength is t's alone and another strength uploads nothing.

// loraMaxGroup bounds a stack: four products of rank 256.
const loraMaxGroup = 4 * 256

// loraOn is one product's change on the card.
type loraOn struct {
	a       int    // the stack of A's this product's is in
	bAt     uint32 // its B's first byte in the B's buffer
	in      uint32 // A's columns
	rows    uint32 // the stack's rows: the stride of t
	at      uint32 // this product's first row in the stack
	rank    uint32
	stacked bool // made by loraA before the stack's products, not by the product
}

// loraGroups is the stacks of a block whose weights are named from pre.
func loraGroups(pre string) [][]string {
	return [][]string{
		{pre + "attn.wq", pre + "attn.wk", pre + "attn.wv", pre + "attn.gate"},
		{pre + "mlp.gate", pre + "mlp.up"},
	}
}

// SetLoRA puts a LoRA on the DiT at a strength, or takes it off when l is nil
// or the strength 0, as ComfyUI's loader does. The same LoRA at another
// strength uploads nothing.
func (m *DiT) SetLoRA(l *LoRA, strength float32) error {
	if strength == 0 {
		l = nil
	}
	if l == m.loraOf && (l == nil || strength == m.loraS) {
		return nil
	}
	m.forget()
	if l != nil && l == m.loraOf {
		m.loraS = strength
		return nil
	}
	m.dropLoRA()
	if l == nil {
		return nil
	}
	for name, d := range l.deltas {
		p, ok := m.products[name]
		if !ok {
			return fmt.Errorf("krea2: LoRA %s changes %s, which is not one of the DiT's products", l.Name, name)
		}
		if d.out != p.rows || d.in != p.cols {
			return fmt.Errorf("krea2: LoRA %s changes %s by %d × %d, and it is %d × %d", l.Name, name, d.out, d.in, p.rows, p.cols)
		}
	}
	var bs []uint16
	stack := func(names []string, stacked bool) error {
		var a []uint16
		var ons []*loraOn
		for _, name := range names {
			d := l.deltas[name]
			if d == nil {
				continue
			}
			lo := &loraOn{bAt: uint32(2 * len(bs)), in: uint32(d.in), at: uint32(len(a) / d.in), rank: uint32(d.rank), stacked: stacked}
			if d.alpha == 1 {
				bs = append(bs, d.b...)
			} else {
				for _, h := range d.b {
					bs = append(bs, floatToHalf(d.alpha*halfToFloat(h)))
				}
			}
			a = append(a, d.a...)
			m.lora[m.products[name].w] = lo
			ons = append(ons, lo)
		}
		if len(ons) == 0 {
			return nil
		}
		rows := uint32(len(a)) / ons[0].in
		if rows > loraMaxGroup {
			return fmt.Errorf("krea2: LoRA %s stacks %d rows on %s, past the %d the DiT keeps room for", l.Name, rows, names[0], loraMaxGroup)
		}
		w, err := m.k.AddWeights(halfBytes(a))
		if err != nil {
			return err
		}
		m.loraBufs = append(m.loraBufs, w)
		for _, lo := range ons {
			lo.a, lo.rows = w, rows
		}
		return nil
	}
	grouped := map[string]bool{}
	for _, g := range m.groups {
		if err := stack(g, true); err != nil {
			m.dropLoRA()
			return err
		}
		for _, name := range g {
			grouped[name] = true
		}
	}
	for name := range l.deltas {
		if !grouped[name] {
			if err := stack([]string{name}, false); err != nil {
				m.dropLoRA()
				return err
			}
		}
	}
	w, err := m.k.AddWeights(halfBytes(bs))
	if err != nil {
		m.dropLoRA()
		return err
	}
	m.loraBufs = append(m.loraBufs, w)
	m.k.SetLoRA(w)
	m.loraOf, m.loraS = l, strength
	return nil
}

func (m *DiT) dropLoRA() {
	m.k.FreeWeights(m.loraBufs...)
	m.loraBufs, m.lora, m.loraOf, m.loraS = nil, map[int]*loraOn{}, nil, 0
}

// LoRA is the LoRA on the DiT and its strength.
func (m *DiT) LoRA() (*LoRA, float32) { return m.loraOf, m.loraS }

// loraA records t = strength · A·x for the stack the first of these products
// with a change is in.
func (m *DiT) loraA(r *vk.Recorder, cols, x uint32, ws ...int) {
	for _, w := range ws {
		if lo := m.lora[w]; lo != nil {
			m.k.MM(r, lo.a, false, vk.K2MM{Outputs: lo.rows, Inputs: lo.in, Cols: cols, X: x, XStride: lo.in,
				Y: m.loraT, YStride: lo.rows, Bias: vk.K2None, Scale: m.loraS, Mode: vk.K2Store})
			return
		}
	}
}

// withLoRA fills in product w's change, making its t first when no stack
// did.
func (m *DiT) withLoRA(r *vk.Recorder, w int, mm vk.K2MM) vk.K2MM {
	lo := m.lora[w]
	if lo == nil {
		return mm
	}
	if !lo.stacked {
		m.loraA(r, mm.Cols, mm.X, w)
	}
	mm.LoRAAt, mm.LoRARank, mm.LoRAT, mm.LoRATStride = lo.bAt, lo.rank, m.loraT+lo.at, lo.rows
	return mm
}
