package vk

// Each Krea 2 kernel against a plain Go version of what it does, on random
// numbers at the shapes the networks use. The card rounds the operands of
// its products to fp16 on purpose, so the Go version rounds them too, and
// what is left is the order of the sums. The attention also rounds its
// probabilities, which the Go version does not, and is held to a few parts
// in a thousand.

import (
	"encoding/binary"
	"math"
	"math/rand"
	"testing"
)

// e4m3 is the float an fp8 e4m3fn byte stands for.
func e4m3(b byte) float32 {
	sign := float32(1)
	if b&0x80 != 0 {
		sign = -1
	}
	e, m := int(b>>3&15), float64(b&7)
	if e == 0 {
		return sign * float32(m/8*math.Pow(2, -6))
	}
	return sign * float32((1+m/8)*math.Pow(2, float64(e-7)))
}

func randomFP8(r *rand.Rand, n int) []byte {
	out := make([]byte, n)
	for i := range out {
		b := byte(r.Intn(256))
		for b&0x7F == 0x7F { // NaN
			b = byte(r.Intn(256))
		}
		// Keep the weights the size real ones are, under a few units.
		if b>>3&15 > 9 {
			b &^= 0x40
		}
		out[i] = b
	}
	return out
}

func randomFloats(r *rand.Rand, n int, scale float32) []float32 {
	out := make([]float32, n)
	for i := range out {
		out[i] = float32(r.NormFloat64()) * scale
	}
	return out
}

func k2Compare(t *testing.T, name string, got, want []float32, tol float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: %d values, want %d", name, len(got), len(want))
	}
	var norm float64
	for _, v := range want {
		norm += float64(v) * float64(v)
	}
	scale := math.Sqrt(norm/float64(len(want))) + 1e-12
	worst, at := 0.0, 0
	for i := range got {
		g := math.Abs(float64(got[i] - want[i]))
		if g > worst || math.IsNaN(g) {
			worst, at = g, i
			if math.IsNaN(g) {
				break
			}
		}
	}
	if !(worst/scale <= tol) {
		t.Fatalf("%s: gap %.3g at %d (got %g, want %g), %.4f%% of the scale", name, worst, at, got[at], want[at], worst/scale*100)
	}
	t.Logf("%s: max gap %.3g, %.4f%% of the scale", name, worst, worst/scale*100)
}

func newK2(t *testing.T, arena int, params []float32) *K2 {
	t.Helper()
	d := open(t)
	t.Cleanup(d.Close)
	if !d.Coopmat() {
		t.Skip("no cooperative matrices")
	}
	k, err := NewK2(d, arena, params)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(k.Close)
	return k
}

func run(t *testing.T, k *K2, record func(r *Recorder)) {
	t.Helper()
	p, err := k.Compile(record)
	if err != nil {
		t.Fatal(err)
	}
	defer p.Close()
	if err := p.Run(); err != nil {
		t.Fatal(err)
	}
}

func TestK2ProductFP8(t *testing.T) {
	r := rand.New(rand.NewSource(1))
	for _, shape := range [][3]int{{6144, 64, 300}, {300, 2560, 7}, {1024, 6144, 3}, {12, 32, 129}} {
		outs, ins, cols := shape[0], shape[1], shape[2]
		w := randomFP8(r, outs*ins)
		x := randomFloats(r, cols*ins, 1)
		bias := randomFloats(r, outs, 1)
		gate := randomFloats(r, outs, 1)
		base := randomFloats(r, cols*outs, 1)
		const scale = 0.37
		for mode := uint32(0); mode < 3; mode++ {
			params := append(append([]float32{}, bias...), gate...)
			k := newK2(t, cols*ins+cols*outs+outs, params)
			wb, err := k.AddWeights(w)
			if err != nil {
				t.Fatal(err)
			}
			yAt, gAt := cols*ins, cols*ins+cols*outs
			k.Write(0, x)
			k.Write(yAt, base)
			k.Write(gAt, gate)
			run(t, k, func(rec *Recorder) {
				k.MM(rec, wb, true, K2MM{Outputs: uint32(outs), Inputs: uint32(ins), Cols: uint32(cols),
					X: 0, XStride: uint32(ins), Y: uint32(yAt), YStride: uint32(outs), Bias: 0, Scale: scale,
					Mode: mode, GateA: uint32(gAt), GateB: uint32(outs)})
			})
			got, err := k.Read(yAt, cols*outs)
			if err != nil {
				t.Fatal(err)
			}
			want := make([]float32, cols*outs)
			for c := 0; c < cols; c++ {
				for o := 0; o < outs; o++ {
					var s float64
					for i := 0; i < ins; i++ {
						s += float64(e4m3(w[o*ins+i])) * float64(f16ToF32(f32ToF16(x[c*ins+i])))
					}
					v := float32(s)*scale + bias[o]
					switch mode {
					case K2Store:
						want[c*outs+o] = v
					case K2Add:
						want[c*outs+o] = base[c*outs+o] + v
					default:
						want[c*outs+o] = base[c*outs+o] + 2*gate[o]*v
					}
				}
			}
			k2Compare(t, "mm", got, want, 5e-5)
		}
	}
}

func TestK2ProductFP16(t *testing.T) {
	r := rand.New(rand.NewSource(2))
	outs, ins, cols := 384, 96, 200
	wf := randomFloats(r, outs*ins, 0.1)
	w := make([]byte, 2*len(wf))
	for i, v := range wf {
		binary.LittleEndian.PutUint16(w[2*i:], f32ToF16(v))
	}
	x := randomFloats(r, cols*ins, 1)
	k := newK2(t, cols*ins+cols*outs, []float32{0})
	wb, _ := k.AddWeights(w)
	k.Write(0, x)
	run(t, k, func(rec *Recorder) {
		k.MM(rec, wb, false, K2MM{Outputs: uint32(outs), Inputs: uint32(ins), Cols: uint32(cols),
			XStride: uint32(ins), Y: uint32(cols * ins), YStride: uint32(outs), Bias: K2None, Scale: 1})
	})
	got, _ := k.Read(cols*ins, cols*outs)
	want := make([]float32, cols*outs)
	for c := 0; c < cols; c++ {
		for o := 0; o < outs; o++ {
			var s float64
			for i := 0; i < ins; i++ {
				s += float64(f16ToF32(f32ToF16(wf[o*ins+i]))) * float64(f16ToF32(f32ToF16(x[c*ins+i])))
			}
			want[c*outs+o] = float32(s)
		}
	}
	k2Compare(t, "mm16", got, want, 1e-5)
}

func TestK2Norm(t *testing.T) {
	r := rand.New(rand.NewSource(3))
	rows, n := 37, 6144
	x := randomFloats(r, rows*n, 3)
	w := randomFloats(r, n, 0.2)
	sB := randomFloats(r, n, 0.1)
	hB := randomFloats(r, n, 0.1)
	sA := randomFloats(r, n, 0.1)
	hA := randomFloats(r, n, 0.1)
	params := append(append(append([]float32{}, w...), sB...), hB...)
	k := newK2(t, 2*rows*n+2*n, params)
	k.Write(0, x)
	k.Write(2*rows*n, sA)
	k.Write(2*rows*n+n, hA)
	run(t, k, func(rec *Recorder) {
		k.Norm(rec, K2Norm{Src: 0, SrcStride: uint32(n), Dst: uint32(rows * n), DstStride: uint32(n), N: uint32(n), Rows: uint32(rows),
			Weight: 0, OnePlus: 1, Eps: 1e-5, ScaleA: uint32(2 * rows * n), ScaleB: uint32(n), ShiftA: uint32(2*rows*n + n), ShiftB: uint32(2 * n)})
	})
	got, _ := k.Read(rows*n, rows*n)
	want := make([]float32, rows*n)
	for r := 0; r < rows; r++ {
		var ss float64
		for i := 0; i < n; i++ {
			ss += float64(x[r*n+i]) * float64(x[r*n+i])
		}
		inv := 1 / math.Sqrt(ss/float64(n)+1e-5)
		for i := 0; i < n; i++ {
			v := float32(float64(x[r*n+i])*inv) * (1 + w[i])
			want[r*n+i] = v*(1+sA[i]+sB[i]) + hA[i] + hB[i]
		}
	}
	k2Compare(t, "norm", got, want, 1e-5)
}

func TestK2Rope(t *testing.T) {
	r := rand.New(rand.NewSource(4))
	tokens, heads, dim := 50, 3, 128
	stride := heads*dim + 17
	x := randomFloats(r, tokens*stride, 1)
	table := make([]float32, tokens*dim)
	for tk := 0; tk < tokens; tk++ {
		for i := 0; i < dim/2; i++ {
			a := float64(tk) * math.Pow(1000, -float64(i)/float64(dim/2))
			table[tk*dim+i] = float32(math.Cos(a))
			table[tk*dim+dim/2+i] = float32(math.Sin(a))
		}
	}
	for halves := uint32(0); halves < 2; halves++ {
		k := newK2(t, tokens*stride+tokens*dim, nil)
		k.Write(0, x)
		k.Write(tokens*stride, table)
		run(t, k, func(rec *Recorder) {
			k.Rope(rec, K2Rope{X: 0, Stride: uint32(stride), Heads: uint32(heads), Dim: uint32(dim), Tokens: uint32(tokens), Table: uint32(tokens * stride), Halves: halves})
		})
		got, _ := k.Read(0, tokens*stride)
		want := append([]float32{}, x...)
		for tk := 0; tk < tokens; tk++ {
			for h := 0; h < heads; h++ {
				at := tk*stride + h*dim
				for i := 0; i < dim/2; i++ {
					i0, i1 := at+2*i, at+2*i+1
					if halves == 1 {
						i0, i1 = at+i, at+i+dim/2
					}
					c, s := table[tk*dim+i], table[tk*dim+dim/2+i]
					want[i0] = c*x[i0] - s*x[i1]
					want[i1] = s*x[i0] + c*x[i1]
				}
			}
		}
		k2Compare(t, "rope", got, want, 1e-6)
	}
}

func TestK2Act(t *testing.T) {
	r := rand.New(rand.NewSource(5))
	rows, n := 9, 1000
	x := randomFloats(r, rows*n, 3)
	y := randomFloats(r, rows*n, 3)
	sig := func(v float64) float64 { return 1 / (1 + math.Exp(-v)) }
	ops := map[uint32]func(a, b float64) float64{
		K2Gelu:    func(a, b float64) float64 { return 0.5 * a * (1 + math.Tanh(0.7978845608028654*(a+0.044715*a*a*a))) },
		K2SwiGLU:  func(a, b float64) float64 { return a * sig(a) * b },
		K2Sigmoid: func(a, b float64) float64 { return a * sig(b) },
		K2Sum:     func(a, b float64) float64 { return a + b },
		K2Copy:    func(a, b float64) float64 { return b },
		K2SiLU:    func(a, b float64) float64 { return a * sig(a) },
	}
	for op, f := range ops {
		k := newK2(t, 2*rows*n, nil)
		k.Write(0, x)
		k.Write(rows*n, y)
		run(t, k, func(rec *Recorder) {
			k.Act(rec, K2Act{X: 0, XStride: uint32(n), Y: uint32(rows * n), YStride: uint32(n), N: uint32(n), Rows: uint32(rows), Op: op})
		})
		got, _ := k.Read(0, rows*n)
		want := make([]float32, rows*n)
		for i := range want {
			want[i] = float32(f(float64(x[i]), float64(y[i])))
		}
		k2Compare(t, "act", got, want, 1e-5)
	}
}

// attention is the plain version: every query of every head of every
// sequence against the keys it may see.
func attention(a K2Attn, mem []float32) []float32 {
	out := append([]float32{}, mem...)
	hd := a.HeadDim
	for s := 0; s < a.Seqs; s++ {
		for h := 0; h < a.Heads; h++ {
			kh := h / int(a.Group)
			for q := 0; q < int(a.Queries); q++ {
				qAt := int(a.Q) + s*int(a.QSeq) + q*int(a.QStride) + h*hd
				last := int(a.Keys)
				if a.Causal != 0 {
					last = q + 1
				}
				scores := make([]float64, last)
				peak := math.Inf(-1)
				for kk := 0; kk < last; kk++ {
					kAt := int(a.K) + s*int(a.KSeq) + kk*int(a.KStride) + kh*hd
					var d float64
					for i := 0; i < hd; i++ {
						d += float64(mem[qAt+i]) * float64(mem[kAt+i])
					}
					scores[kk] = d * float64(a.Scale)
					peak = math.Max(peak, scores[kk])
				}
				var sum float64
				for kk := range scores {
					scores[kk] = math.Exp(scores[kk] - peak)
					sum += scores[kk]
				}
				oAt := int(a.O) + s*int(a.OSeq) + q*int(a.OStride) + h*hd
				for i := 0; i < hd; i++ {
					var v float64
					for kk := range scores {
						v += scores[kk] * float64(mem[int(a.V)+s*int(a.VSeq)+kk*int(a.VStride)+kh*hd+i])
					}
					out[oAt+i] = float32(v / sum)
				}
			}
		}
	}
	return out
}

func TestK2Attention(t *testing.T) {
	r := rand.New(rand.NewSource(6))
	cases := []struct {
		name                         string
		heads, group, hd, seqs, len_ int
		causal                       uint32
	}{
		{"dit", 8, 4, 128, 1, 263, 0},
		{"encoder", 8, 4, 128, 1, 41, 1},
		{"fusion", 4, 1, 128, 20, 12, 0},
		{"vae", 1, 1, 384, 1, 150, 0},
	}
	for _, c := range cases {
		qs := c.heads * c.hd
		ks := c.heads / c.group * c.hd
		n := c.len_ * c.seqs
		q, kAt, vAt, oAt := 0, n*qs, n*qs+n*ks, n*qs+2*n*ks
		mem := randomFloats(r, oAt+n*qs, 1)
		a := K2Attn{Q: uint32(q), QStride: uint32(qs), QSeq: uint32(c.len_ * qs),
			K: uint32(kAt), KStride: uint32(ks), KSeq: uint32(c.len_ * ks),
			V: uint32(vAt), VStride: uint32(ks), VSeq: uint32(c.len_ * ks),
			O: uint32(oAt), OStride: uint32(qs), OSeq: uint32(c.len_ * qs),
			Queries: uint32(c.len_), Keys: uint32(c.len_), Group: uint32(c.group), Causal: c.causal,
			Scale: float32(1 / math.Sqrt(float64(c.hd))), Heads: c.heads, Seqs: c.seqs, HeadDim: c.hd}
		k := newK2(t, len(mem), nil)
		k.Write(0, mem)
		run(t, k, func(rec *Recorder) { k.Attn(rec, a) })
		got, _ := k.Read(oAt, n*qs)
		want := attention(a, mem)[oAt:]
		k2Compare(t, c.name, got, want, 5e-3)
	}
}

func TestK2Conv(t *testing.T) {
	r := rand.New(rand.NewSource(7))
	for _, c := range []struct{ outs, ins, h, w, taps, up int }{{96, 16, 20, 33, 9, 0}, {3, 96, 17, 16, 9, 0}, {384, 192, 9, 10, 1, 0}, {192, 384, 12, 14, 9, 1}} {
		K := c.ins * c.taps
		kStride := (K + 31) / 32 * 32
		wf := randomFloats(r, c.outs*K, 0.1)
		w := make([]byte, 2*c.outs*kStride)
		for o := 0; o < c.outs; o++ {
			for i := 0; i < K; i++ {
				binary.LittleEndian.PutUint16(w[2*(o*kStride+i):], f32ToF16(wf[o*K+i]))
			}
		}
		px := c.h * c.w
		ih, iw := c.h>>c.up, c.w>>c.up
		x := randomFloats(r, c.ins*ih*iw, 1)
		res := randomFloats(r, c.outs*px, 1)
		bias := randomFloats(r, c.outs, 1)
		k := newK2(t, c.ins*px+2*c.outs*px, bias)
		wb, _ := k.AddWeights(w)
		k.Write(0, x)
		k.Write(c.ins*px+c.outs*px, res)
		run(t, k, func(rec *Recorder) {
			k.Conv(rec, wb, K2Conv{Outputs: uint32(c.outs), Inputs: uint32(c.ins), H: uint32(c.h), W: uint32(c.w), Taps: uint32(c.taps),
				KStride: uint32(kStride), X: 0, Y: uint32(c.ins * px), Bias: 0, Residual: uint32(c.ins*px + c.outs*px), Up: uint32(c.up)})
		})
		got, _ := k.Read(c.ins*px, c.outs*px)
		want := make([]float32, c.outs*px)
		for o := 0; o < c.outs; o++ {
			for y := 0; y < c.h; y++ {
				for xx := 0; xx < c.w; xx++ {
					s := float64(bias[o])
					for i := 0; i < c.ins; i++ {
						for tap := 0; tap < c.taps; tap++ {
							yy, xs := y, xx
							if c.taps == 9 {
								yy, xs = y+tap/3-1, xx+tap%3-1
							}
							if yy < 0 || yy >= c.h || xs < 0 || xs >= c.w {
								continue
							}
							s += float64(f16ToF32(f32ToF16(wf[o*K+i*c.taps+tap]))) * float64(f16ToF32(f32ToF16(x[i*ih*iw+(yy>>c.up)*iw+xs>>c.up])))
						}
					}
					want[o*px+y*c.w+xx] = float32(s) + res[o*px+y*c.w+xx]
				}
			}
		}
		k2Compare(t, "conv", got, want, 5e-5)
	}
}

func TestK2Pix(t *testing.T) {
	r := rand.New(rand.NewSource(8))
	C, H, W := 96, 13, 11
	px := H * W
	x := randomFloats(r, C*px, 2)
	res := randomFloats(r, C*px, 1)
	gamma := randomFloats(r, C, 1)
	k := newK2(t, 8*C*px, gamma)
	k.Write(0, x)
	k.Write(7*C*px, res)
	run(t, k, func(rec *Recorder) {
		k.Pix(rec, K2Pix{Src: 0, Dst: uint32(C * px), C: uint32(C), H: uint32(H), W: uint32(W), Gamma: 0, SiLU: 1, Op: K2ChanNorm})
		k.Pix(rec, K2Pix{Src: 0, Dst: uint32(2 * C * px), C: uint32(C), H: uint32(H), W: uint32(W), Op: K2Up2})
		k.Pix(rec, K2Pix{Src: 0, Dst: uint32(6 * C * px), C: uint32(C), H: uint32(H), W: uint32(W), Op: K2ToTokens})
		k.Pix(rec, K2Pix{Src: uint32(6 * C * px), Dst: uint32(C * px * 7 / 1), C: uint32(C), H: uint32(H), W: uint32(W), Op: K2ToPlanes, Residual: uint32(7 * C * px)})
	})
	got, _ := k.Read(0, 8*C*px)
	norm := make([]float32, C*px)
	for p := 0; p < px; p++ {
		var ss float64
		for c := 0; c < C; c++ {
			ss += float64(x[c*px+p]) * float64(x[c*px+p])
		}
		f := math.Sqrt(float64(C)) / math.Sqrt(ss)
		for c := 0; c < C; c++ {
			v := float64(x[c*px+p]) * f * float64(gamma[c])
			norm[c*px+p] = float32(v / (1 + math.Exp(-v)))
		}
	}
	k2Compare(t, "chnorm", got[C*px:2*C*px], norm, 1e-5)
	up := make([]float32, 4*C*px)
	for c := 0; c < C; c++ {
		for y := 0; y < 2*H; y++ {
			for xx := 0; xx < 2*W; xx++ {
				up[c*4*px+y*2*W+xx] = x[c*px+(y/2)*W+xx/2]
			}
		}
	}
	k2Compare(t, "up2", got[2*C*px:6*C*px], up, 0)
	back := make([]float32, C*px)
	for i := range back {
		back[i] = x[i] + res[i]
	}
	k2Compare(t, "tokens and back", got[7*C*px:8*C*px], back, 0)
}

func f32ToF16(f float32) uint16 {
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

func f16ToF32(h uint16) float32 {
	sign := uint32(h&0x8000) << 16
	exp := uint32(h >> 10 & 0x1f)
	mant := uint32(h & 0x3ff)
	switch {
	case exp == 0:
		return math.Float32frombits(sign) + float32(mant)*float32(math.Pow(2, -24))*sgn(sign)
	case exp == 31:
		return math.Float32frombits(sign | 0x7f800000 | mant<<13)
	}
	return math.Float32frombits(sign | (exp+112)<<23 | mant<<13)
}

func sgn(s uint32) float32 {
	if s != 0 {
		return -1
	}
	return 1
}

// A product with a LoRA on it: the kernel goes on from W to B against t, a
// few columns further along a wider t, B at an offset in its buffer, in each
// of the three modes.
func TestK2ProductLoRA(t *testing.T) {
	r := rand.New(rand.NewSource(3))
	outs, ins, cols, rank := 300, 64, 130, 64
	tRows, tAt := 96, 32 // t is 96 rows a column; this product's are 32 to 96
	w := randomFP8(r, outs*ins)
	x := randomFloats(r, cols*ins, 1)
	tv := randomFloats(r, cols*tRows, 1)
	bf := randomFloats(r, outs*rank, 0.05)
	const pad = 24 // halves before B in its buffer, 48 bytes
	bb := make([]byte, 2*(pad+len(bf)))
	for i, v := range bf {
		binary.LittleEndian.PutUint16(bb[2*(pad+i):], f32ToF16(v))
	}
	bias := randomFloats(r, outs, 1)
	gate := randomFloats(r, outs, 1)
	base := randomFloats(r, cols*outs, 1)
	const scale = 0.37
	for mode := uint32(0); mode < 3; mode++ {
		params := append(append([]float32{}, bias...), gate...)
		yAt := cols * ins
		tIn := yAt + cols*outs
		gAt := tIn + cols*tRows
		k := newK2(t, gAt+outs, params)
		wb, err := k.AddWeights(w)
		if err != nil {
			t.Fatal(err)
		}
		lb, err := k.AddWeights(bb)
		if err != nil {
			t.Fatal(err)
		}
		k.SetLoRA(lb)
		k.Write(0, x)
		k.Write(yAt, base)
		k.Write(tIn, tv)
		k.Write(gAt, gate)
		run(t, k, func(rec *Recorder) {
			k.MM(rec, wb, true, K2MM{Outputs: uint32(outs), Inputs: uint32(ins), Cols: uint32(cols),
				X: 0, XStride: uint32(ins), Y: uint32(yAt), YStride: uint32(outs), Bias: 0, Scale: scale,
				Mode: mode, GateA: uint32(gAt), GateB: uint32(outs),
				LoRAAt: 2 * pad, LoRARank: uint32(rank), LoRAT: uint32(tIn + tAt), LoRATStride: uint32(tRows)})
		})
		got, err := k.Read(yAt, cols*outs)
		if err != nil {
			t.Fatal(err)
		}
		want := make([]float32, cols*outs)
		for c := 0; c < cols; c++ {
			for o := 0; o < outs; o++ {
				var s float64
				for i := 0; i < ins; i++ {
					s += float64(e4m3(w[o*ins+i])) * float64(f16ToF32(f32ToF16(x[c*ins+i])))
				}
				for j := 0; j < rank; j++ {
					s += float64(f16ToF32(f32ToF16(bf[o*rank+j]))) * float64(f16ToF32(f32ToF16(tv[c*tRows+tAt+j])))
				}
				v := float32(s)*scale + bias[o]
				switch mode {
				case K2Store:
					want[c*outs+o] = v
				case K2Add:
					want[c*outs+o] = base[c*outs+o] + v
				default:
					want[c*outs+o] = base[c*outs+o] + 2*gate[o]*v
				}
			}
		}
		k2Compare(t, "mm with a LoRA", got, want, 5e-5)
	}
}
