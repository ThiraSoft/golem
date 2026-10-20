package vk

// Each Krea 2 kernel against a plain Go version of what it does, on random
// numbers at the shapes the networks use. The card rounds the operands of
// its products to fp16 on purpose, so the Go version rounds them too, and
// what is left is the order of the sums. The attention also rounds its
// probabilities, which the Go version does not, and is held to a few parts
// in a thousand.

import (
	"encoding/binary"
	"fmt"
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

// TestK2ProductSwiGLU: a SwiGLU's stacked weight in runs (SwiGLURow), X in
// fp16, silu(gate)·up out as fp16; with fp16 weights in tiles of 128 and of
// 256, and fp8 ones.
func TestK2ProductSwiGLU(t *testing.T) {
	r := rand.New(rand.NewSource(9))
	for _, c := range []struct {
		n, ins, cols int
		fp8          bool
	}{{352, 96, 200, false}, {9216, 32, 2000, false}, {352, 96, 200, true}} {
		outs := 2 * c.n
		wf := randomFloats(r, outs*c.ins, 0.1)
		wq := make([]float32, len(wf)) // what the card sees of each weight
		var w []byte
		if c.fp8 {
			w8 := randomFP8(r, outs*c.ins)
			for i := range wq {
				wq[i] = e4m3(w8[i])
			}
			w = make([]byte, len(w8))
			for o := 0; o < outs; o++ {
				copy(w[SwiGLURow(o, c.n)*c.ins:], w8[o*c.ins:(o+1)*c.ins])
			}
		} else {
			w = make([]byte, 2*len(wf))
			for o := 0; o < outs; o++ {
				at := SwiGLURow(o, c.n) * c.ins
				for i := 0; i < c.ins; i++ {
					h := f32ToF16(wf[o*c.ins+i])
					wq[o*c.ins+i] = f16ToF32(h)
					binary.LittleEndian.PutUint16(w[2*(at+i):], h)
				}
			}
		}
		x := randomFloats(r, c.cols*c.ins, 1)
		xh := make([]float32, len(x)/2) // x as fp16, two to a float
		for i := 0; i < len(x); i += 2 {
			lo, hi := uint32(f32ToF16(x[i])), uint32(f32ToF16(x[i+1]))
			xh[i/2] = math.Float32frombits(lo | hi<<16)
		}
		const scale = 0.8
		yAt := len(xh)
		k := newK2(t, yAt+c.cols*c.n/2, []float32{0})
		wb, _ := k.AddWeights(w)
		k.Write(0, xh)
		run(t, k, func(rec *Recorder) {
			k.MM(rec, wb, c.fp8, K2MM{Outputs: uint32(outs), Inputs: uint32(c.ins), Cols: uint32(c.cols),
				XStride: uint32(c.ins), Y: uint32(2 * yAt), YStride: uint32(c.n), Bias: K2None, Scale: scale,
				Mode: K2SwiGLUHalves, XHalves: 1})
		})
		raw, _ := k.Read(yAt, c.cols*c.n/2)
		got := make([]float32, c.cols*c.n)
		for i, v := range raw {
			b := math.Float32bits(v)
			got[2*i], got[2*i+1] = f16ToF32(uint16(b)), f16ToF32(uint16(b>>16))
		}
		want := make([]float32, c.cols*c.n)
		for col := 0; col < c.cols; col++ {
			for g := 0; g < c.n; g++ {
				var a, u float64
				for i := 0; i < c.ins; i++ {
					xi := float64(f16ToF32(f32ToF16(x[col*c.ins+i])))
					a += float64(wq[g*c.ins+i]) * xi
					u += float64(wq[(c.n+g)*c.ins+i]) * xi
				}
				a, u = a*scale, u*scale
				want[col*c.n+g] = f16ToF32(f32ToF16(float32(a / (1 + math.Exp(-a)) * u)))
			}
		}
		k2Compare(t, fmt.Sprintf("swiglu %d x %d x %d, fp8 %v", outs, c.ins, c.cols, c.fp8), got, want, 1e-2) // an fp16 step of the largest either side
	}
}

func TestK2Flux3QKRope(t *testing.T) {
	r := rand.New(rand.NewSource(10))
	const heads, tokens, stride = 3, 37, 3*128 + 32
	q := randomFloats(r, tokens*stride, 2)
	kk := randomFloats(r, tokens*stride, 2)
	table := randomFloats(r, tokens*128, 1)
	qw := randomFloats(r, 128, 0.5)
	kw := randomFloats(r, 128, 0.5)
	k := newK2(t, 2*tokens*stride+tokens*128, append(append([]float32{}, qw...), kw...))
	k.Write(0, q)
	k.Write(tokens*stride, kk)
	k.Write(2*tokens*stride, table)
	run(t, k, func(rec *Recorder) {
		k.Flux3QKRope(rec, K2Flux3QKRope{Q: 0, K: tokens * stride, Stride: stride, Heads: heads, Tokens: tokens,
			Table: 2 * tokens * stride, QWeight: 0, KWeight: 128, Eps: 1e-6})
	})
	got, _ := k.Read(0, 2*tokens*stride)
	want := append(append([]float32{}, q...), kk...)
	for which, w := range [][]float32{qw, kw} {
		x := want[which*tokens*stride:]
		for tok := 0; tok < tokens; tok++ {
			for h := 0; h < heads; h++ {
				v := x[tok*stride+h*128:][:128]
				var ss float64
				for _, e := range v {
					ss += float64(e) * float64(e)
				}
				inv := 1 / math.Sqrt(ss/128+1e-6)
				for i := 0; i < 64; i++ {
					x0 := float64(v[2*i]) * inv * float64(w[2*i])
					x1 := float64(v[2*i+1]) * inv * float64(w[2*i+1])
					c, s := float64(table[tok*128+i]), float64(table[tok*128+64+i])
					v[2*i], v[2*i+1] = float32(c*x0-s*x1), float32(s*x0+c*x1)
				}
			}
		}
	}
	k2Compare(t, "qkrope", got, want, 1e-5)
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

func TestK2CopyStrided(t *testing.T) {
	x := make([]float32, 64)
	for i := range x {
		x[i] = float32(i)
	}
	k := newK2(t, 64, nil)
	k.Write(0, x)
	run(t, k, func(rec *Recorder) { k.CopyStrided(rec, 40, 5, 1, 10, 3, 3) })
	got, _ := k.Read(0, 64)
	want := append([]float32(nil), x...)
	for i, v := range []float32{1, 2, 3, -1, -1, 11, 12, 13, -1, -1, 21, 22, 23} {
		if v >= 0 {
			want[40+i] = v
		}
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("float %d is %g, want %g", i, got[i], want[i])
		}
	}
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
		amp                          float32
	}{
		{"dit", 8, 4, 128, 1, 263, 0, 1},
		{"encoder", 8, 4, 128, 1, 41, 1, 1},
		{"fusion", 4, 1, 128, 20, 12, 0, 1},
		{"vae", 1, 1, 384, 1, 150, 0, 1},
		// What shaders/krea2_flash.comp takes: long, with and without
		// groups, two sequences, ragged ends, and peaked scores, which fp16
		// keys round to 1.39% of the scale whichever kernel runs.
		{"long", 4, 1, 128, 1, 2100, 0, 1},
		{"long gqa", 8, 4, 128, 1, 1100, 0, 1},
		{"long seqs", 2, 1, 128, 2, 700, 0, 1},
		{"long peaked", 2, 1, 128, 1, 1500, 0, 3},
	}
	for _, c := range cases {
		qs := c.heads * c.hd
		ks := c.heads / c.group * c.hd
		n := c.len_ * c.seqs
		q, kAt, vAt, oAt := 0, n*qs, n*qs+n*ks, n*qs+2*n*ks
		mem := randomFloats(r, oAt+n*qs, c.amp)
		a := K2Attn{Q: uint32(q), QStride: uint32(qs), QSeq: uint32(c.len_ * qs),
			K: uint32(kAt), KStride: uint32(ks), KSeq: uint32(c.len_ * ks),
			V: uint32(vAt), VStride: uint32(ks), VSeq: uint32(c.len_ * ks),
			O: uint32(oAt), OStride: uint32(qs), OSeq: uint32(c.len_ * qs),
			Queries: uint32(c.len_), Keys: uint32(c.len_), Group: uint32(c.group), Causal: c.causal,
			Scale: float32(1 / math.Sqrt(float64(c.hd))), Heads: c.heads, Seqs: c.seqs, HeadDim: c.hd}
		want := attention(a, mem)[oAt:]
		tol := 5e-3
		if c.amp > 1 {
			tol = 2e-2
		}
		// Reading the arena, and reading the fp16 copy ReserveAttention
		// makes room for.
		for _, reserve := range []bool{false, true} {
			k := newK2(t, len(mem), nil)
			name := c.name
			if reserve {
				k.ReserveAttention(c.len_, c.heads/c.group)
				name += ", fp16 copy"
			}
			k.Write(0, mem)
			run(t, k, func(rec *Recorder) { k.Attn(rec, a) })
			got, _ := k.Read(oAt, n*qs)
			k2Compare(t, name, got, want, tol)
		}
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

func TestK2Flux3(t *testing.T) {
	r := rand.New(rand.NewSource(13))
	rows, n := 4, 3072
	x := randomFloats(r, rows*n, 1)
	scale := randomFloats(r, n, 0.5)
	shift := randomFloats(r, n, 0.5)

	// Test 1: Flux3Norm float32
	k := newK2(t, 2*rows*n+2*n, nil)
	k.Write(0, x)
	k.Write(2*rows*n, scale)
	k.Write(2*rows*n+n, shift)
	run(t, k, func(rec *Recorder) {
		k.Flux3Norm(rec, K2Flux3Norm{
			Src: 0, SrcStride: uint32(n),
			Dst: uint32(rows * n), DstStride: uint32(n),
			N: uint32(n), Rows: uint32(rows),
			Scale: uint32(2 * rows * n), Shift: uint32(2*rows*n + n),
			Eps: 1e-6, Halves: 0,
		})
	})
	got, _ := k.Read(rows*n, rows*n)
	want := make([]float32, rows*n)
	for row := 0; row < rows; row++ {
		var sum float64
		for i := 0; i < n; i++ {
			sum += float64(x[row*n+i])
		}
		mean := sum / float64(n)
		var varSum float64
		for i := 0; i < n; i++ {
			diff := float64(x[row*n+i]) - mean
			varSum += diff * diff
		}
		inv := 1.0 / math.Sqrt(varSum/float64(n)+1e-6)
		for i := 0; i < n; i++ {
			d := (float64(x[row*n+i]) - mean) * inv
			want[row*n+i] = float32(d*(1.0+float64(scale[i])) + float64(shift[i]))
		}
	}
	k2Compare(t, "flux3_norm f32", got, want, 1e-5)

	// Test 1b: Flux3Norm fp16 halves
	kHalves := newK2(t, 2*rows*n+2*n, nil)
	kHalves.Write(0, x)
	kHalves.Write(2*rows*n, scale)
	kHalves.Write(2*rows*n+n, shift)
	run(t, kHalves, func(rec *Recorder) {
		kHalves.Flux3Norm(rec, K2Flux3Norm{
			Src: 0, SrcStride: uint32(n),
			Dst: uint32(2 * rows * n), DstStride: uint32(n),
			N: uint32(n), Rows: uint32(rows),
			Scale: uint32(2 * rows * n), Shift: uint32(2*rows*n + n),
			Eps: 1e-6, Halves: 1,
		})
	})
	gotH, _ := kHalves.Read(rows*n, rows*n)
	gotF := make([]float32, rows*n)
	for i := 0; i < rows*n; i += 2 {
		bits := math.Float32bits(gotH[i/2])
		gotF[i] = f16ToF32(uint16(bits & 0xffff))
		gotF[i+1] = f16ToF32(uint16(bits >> 16))
	}
	k2Compare(t, "flux3_norm fp16", gotF, want, 2e-3)

	// Test 2: Flux3Gate
	gate := randomFloats(r, n, 0.5)
	out := randomFloats(r, rows*n, 1)
	xBase := randomFloats(r, rows*n, 1)
	kGate := newK2(t, 2*rows*n+n, nil)
	kGate.Write(0, xBase)
	kGate.Write(rows*n, out)
	kGate.Write(2*rows*n, gate)
	run(t, kGate, func(rec *Recorder) {
		kGate.Flux3Gate(rec, K2Flux3Gate{
			X: 0, XStride: uint32(n),
			Out: uint32(rows * n), OutStride: uint32(n),
			Gate: uint32(2 * rows * n),
			N:    uint32(n), Rows: uint32(rows),
		})
	})
	gotGate, _ := kGate.Read(0, rows*n)
	wantGate := make([]float32, rows*n)
	for row := 0; row < rows; row++ {
		for i := 0; i < n; i++ {
			wantGate[row*n+i] = xBase[row*n+i] + out[row*n+i]*gate[i]
		}
	}
	k2Compare(t, "flux3_gate", gotGate, wantGate, 1e-5)
}

func TestK2Yue2Axpy(t *testing.T) {
	r := rand.New(rand.NewSource(42))
	n := 4096
	a := randomFloats(r, n, 2)
	b := randomFloats(r, n, 2)
	s := float32(-0.015625)

	// Test single destination: dst = a + s * b
	arena := 3 * n
	k := newK2(t, arena, nil)
	k.Write(0, a)
	k.Write(n, b)
	run(t, k, func(rec *Recorder) {
		k.Yue2Axpy(rec, K2Yue2Axpy{
			N:   uint32(n),
			Dst: uint32(2 * n),
			A:   0,
			B:   uint32(n),
			S:   s,
		})
	})
	got, _ := k.Read(2*n, n)
	want := make([]float32, n)
	for i := range want {
		want[i] = a[i] + s*b[i]
	}
	k2Compare(t, "yue2_axpy_single", got, want, 1e-6)

	// Test dual destination: dst = dst2 = a + s * b
	arenaDual := 4 * n
	k2 := newK2(t, arenaDual, nil)
	k2.Write(0, a)
	k2.Write(n, b)
	run(t, k2, func(rec *Recorder) {
		k2.Yue2Axpy(rec, K2Yue2Axpy{
			N:       uint32(n),
			Dst:     uint32(2 * n),
			A:       0,
			B:       uint32(n),
			S:       s,
			Dst2:    uint32(3 * n),
			HasDst2: 1,
		})
	})
	got1, _ := k2.Read(2*n, n)
	got2, _ := k2.Read(3*n, n)
	k2Compare(t, "yue2_axpy_dual_dst1", got1, want, 1e-6)
	k2Compare(t, "yue2_axpy_dual_dst2", got2, want, 1e-6)
}
