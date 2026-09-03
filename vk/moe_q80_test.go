package vk

// The routed kernels against a Q8_0 expert stack, held to the Q4_0 ones.
//
// A mixture in Q8_0 is why this exists: eight and a half bits a weight against
// four and a half puts the 26B A4B's expert pool at twenty-four gigabytes,
// which is more than the sixteen a card can address here — the first checkpoint
// this engine reads whose pool does not fit anywhere but the file.
//
// The two kernels are compared on the same floats, quantized each way. They
// cannot agree exactly: Q4_0 keeps sixteen levels a block and Q8_0 two hundred
// and fifty-six, so what is measured is that the finer one answers the coarser
// one's answer to within the coarser one's own error. A wrong stride, a wrong
// scale offset or a wrong expert does not land inside that; it lands in noise.

import (
	"encoding/binary"
	"math"
	"math/rand"
	"testing"
	"unsafe"

	"github.com/ThiraSoft/golem/nn"
)

// encodeQ4_0 is llama.cpp's, block by block: the value furthest from nought
// sets a scale that puts it at minus eight, and every weight is the nearest of
// the sixteen levels above that.
func encodeQ4_0(w []float32) []byte {
	nb := len(w) / nn.QuantBlock
	out := make([]byte, nb*18)
	for b := 0; b < nb; b++ {
		blk := w[b*nn.QuantBlock : (b+1)*nn.QuantBlock]
		amax, most := float32(0), float32(0)
		for _, v := range blk {
			if a := float32(math.Abs(float64(v))); a > amax {
				amax, most = a, v
			}
		}
		d := most / -8
		id := float32(0)
		if d != 0 {
			id = 1 / d
		}
		binary.LittleEndian.PutUint16(out[b*18:], nn.FloatToHalf(d))
		for i := 0; i < nn.QuantBlock/2; i++ {
			lo := byte(min(15, int(blk[i]*id+8.5)))
			hi := byte(min(15, int(blk[i+nn.QuantBlock/2]*id+8.5)))
			out[b*18+2+i] = lo | hi<<4
		}
	}
	return out
}

// encodeQ8_0 is the same shape with two hundred and fifty-six levels and no
// offset, which is why its kernel needs no correction term.
func encodeQ8_0(w []float32) []byte {
	nb := len(w) / nn.QuantBlock
	out := make([]byte, nb*34)
	for b := 0; b < nb; b++ {
		blk := w[b*nn.QuantBlock : (b+1)*nn.QuantBlock]
		amax := float32(0)
		for _, v := range blk {
			if a := float32(math.Abs(float64(v))); a > amax {
				amax = a
			}
		}
		d := amax / 127
		id := float32(0)
		if d != 0 {
			id = 1 / d
		}
		binary.LittleEndian.PutUint16(out[b*34:], nn.FloatToHalf(d))
		for i, v := range blk {
			out[b*34+2+i] = byte(int8(max(-128, min(127, int(math.Round(float64(v*id)))))))
		}
	}
	return out
}

// TestQ80ExpertsAnswerWhatQ40Does runs one column through the fused gate and up
// in both forms and compares what it wrote — the activation the second
// projection reads, dequantized.
func TestQ80ExpertsAnswerWhatQ40Does(t *testing.T) {
	d := open(t)
	defer d.Close()

	const dim, ffn, experts = 64, 64, 4
	const pick = 2 // an expert that is not the first, so a wrong stride shows
	rng := rand.New(rand.NewSource(19))
	weights := make([]float32, experts*2*ffn*dim)
	for i := range weights {
		weights[i] = rng.Float32()*0.4 - 0.2
	}
	x := make([]float32, dim)
	for i := range x {
		x[i] = rng.Float32()*2 - 1
	}
	q, scales, widened := q80Column(x)

	gelu, err := d.Upload(asBytes(nn.GELUTableData()))
	if err != nil {
		t.Fatal(err)
	}
	defer gelu.Close()
	xq, err := d.Upload(asBytesUint32(q))
	if err != nil {
		t.Fatal(err)
	}
	defer xq.Close()
	xs, err := d.Upload(asBytes(scales))
	if err != nil {
		t.Fatal(err)
	}
	defer xs.Close()
	ids, err := d.Upload(asBytesInt32([]int32{pick}))
	if err != nil {
		t.Fatal(err)
	}
	defer ids.Close()

	run := func(spirv []byte, packed []byte, stride int) []float32 {
		t.Helper()
		w, err := d.Upload(splitFor(packed, experts*2*ffn, dim, stride))
		if err != nil {
			t.Fatal(err)
		}
		defer w.Close()
		mid := ffn / nn.QuantBlock
		aq, err := d.Readback(ffn, bufferUsageStorage)
		if err != nil {
			t.Fatal(err)
		}
		defer aq.Close()
		as, err := d.Readback(2*mid*4, bufferUsageStorage)
		if err != nil {
			t.Fatal(err)
		}
		defer as.Close()
		pipe, err := d.NewPipeline(spirv, 7, uint32(unsafe.Sizeof(moePush{})))
		if err != nil {
			t.Fatal(err)
		}
		defer pipe.Close()
		set, err := pipe.NewSet([]*Buffer{w, xq, xs, ids, gelu, aq, as})
		if err != nil {
			t.Fatal(err)
		}
		push := moePush{dim: dim, ffn: ffn, used: 1, act: uint32(GELU), cap: 1}
		if err := set.Dispatch(uint32(ffn/nn.QuantBlock), unsafe.Pointer(&push)); err != nil {
			t.Fatal(err)
		}
		out := make([]float32, ffn)
		bytes := aq.Bytes()
		sc := as.Floats()
		for i := range out {
			out[i] = float32(int8(bytes[i])) * sc[i/nn.QuantBlock]
		}
		return out
	}

	coarse := run(moeGateUpSPIRV, encodeQ4_0(weights), 18)
	fine := run(moeGateUpQ80SPIRV, encodeQ8_0(weights), 34)

	// The truth: the same product over the float weights and the activation the
	// card actually sees, which is the Q8_0 one widened. Comparing the two
	// kernels to each other says nothing on its own — the gate is a GELU and
	// x·sigma(x) is steep near nought, so two quantizations of the same matrix
	// differ by more than either differs from the truth in the tail. Comparing
	// each to the truth says which is the finer, which is the claim.
	want := expertReference(weights, widened, pick, dim, ffn)

	off := func(got []float32) float64 {
		var num, den float64
		for i := range want {
			diff := float64(got[i] - want[i])
			num += diff * diff
			den += float64(want[i]) * float64(want[i])
		}
		return math.Sqrt(num / den)
	}
	q4, q8 := off(coarse), off(fine)
	t.Logf("against the float answer: Q4_0 is %.4f off, Q8_0 is %.4f", q4, q8)
	if q8 >= q4 {
		t.Fatalf("the Q8_0 experts are %.4f from the truth and the Q4_0 ones %.4f — the finer form must be the nearer", q8, q4)
	}
	// Eight bits a weight over sixty-four inputs, twice, through a gate: a
	// twentieth is loose for that and tight against anything structural, which
	// lands in noise rather than near.
	if q8 > 0.05 {
		t.Fatalf("the Q8_0 experts sit %.4f from the float answer, which is not a quantization error", q8)
	}
}

// expertReference is the fused gate and up in Go: one expert's two halves
// against the activation, gated by ggml's tabulated GELU.
func expertReference(weights, x []float32, expert, dim, ffn int) []float32 {
	table := nn.GELUTableData()
	base := expert * 2 * ffn * dim
	out := make([]float32, ffn)
	for r := 0; r < ffn; r++ {
		var gate, up float32
		for i := 0; i < dim; i++ {
			gate += weights[base+r*dim+i] * x[i]
			up += weights[base+(ffn+r)*dim+i] * x[i]
		}
		a := gate
		switch {
		case gate <= -10:
			a = 0
		case gate >= 10:
		default:
			a = table[nn.FloatToHalf(gate)]
		}
		out[r] = a * up
	}
	return out
}

// splitFor lays a packed stack out the way the routed kernels read it, by the
// block width its format uses.
func splitFor(packed []byte, rows, cols, stride int) []byte {
	if stride == 34 {
		return splitQ8_0(packed, rows, cols)
	}
	return splitQ4_0(packed, rows, cols)
}

// TestQ80DownAnswersWhatQ40Does is the other half, and it was missing: the
// second projection was written by analogy with the first and only ever
// compiled. It reads a different buffer, a different scale layout and a
// different activation, so analogy is not evidence.
func TestQ80DownAnswersWhatQ40Does(t *testing.T) {
	d := open(t)
	defer d.Close()

	const dim, ffn, experts = 64, 64, 8
	rng := rand.New(rand.NewSource(31))
	weights := make([]float32, experts*dim*ffn)
	for i := range weights {
		weights[i] = rng.Float32()*0.4 - 0.2
	}
	// One column routing to eight experts, each with its own weight.
	ids := []int32{5, 1, 7, 0, 3, 6, 2, 4}
	cw := make([]float32, len(ids))
	for i := range cw {
		cw[i] = rng.Float32()
	}
	// The activation, one row per pair of a column and an expert.
	act := make([]float32, len(ids)*ffn)
	for i := range act {
		act[i] = rng.Float32()*2 - 1
	}
	aq := make([]uint32, 0, len(ids)*ffn/4)
	as := make([]float32, 0, 2*len(ids)*ffn/nn.QuantBlock)
	widened := make([]float32, 0, len(act))
	for k := range ids {
		q, s, w := q80Column(act[k*ffn : (k+1)*ffn])
		aq = append(aq, q...)
		widened = append(widened, w...)
		as = append(as, s...)
	}
	// The kernel wants every pair's scales together, then every pair's
	// corrections: p.col*2*used*nb + k*nb + b, with the corrections a used*nb
	// further on. q80Column hands them back one pair at a time, scales then
	// corrections, so they are dealt out here.
	nb := ffn / nn.QuantBlock
	laid := make([]float32, 2*len(ids)*nb)
	for k := range ids {
		copy(laid[k*nb:], as[k*2*nb:k*2*nb+nb])
		copy(laid[len(ids)*nb+k*nb:], as[k*2*nb+nb:k*2*nb+2*nb])
	}

	host := func(b []byte) *Buffer {
		buf, err := d.Upload(b)
		if err != nil {
			t.Fatal(err)
		}
		return buf
	}
	aqBuf := host(asBytesUint32(aq))
	defer aqBuf.Close()
	asBuf := host(asBytes(laid))
	defer asBuf.Close()
	idBuf := host(asBytesInt32(ids))
	defer idBuf.Close()
	cwBuf := host(asBytes(cw))
	defer cwBuf.Close()

	run := func(spirv []byte, packed []byte, stride int) []float32 {
		t.Helper()
		w := host(splitFor(packed, experts*dim, ffn, stride))
		defer w.Close()
		y, err := d.Readback(dim*4, bufferUsageStorage)
		if err != nil {
			t.Fatal(err)
		}
		defer y.Close()
		pipe, err := d.NewPipeline(spirv, 6, uint32(unsafe.Sizeof(moePush{})))
		if err != nil {
			t.Fatal(err)
		}
		defer pipe.Close()
		set, err := pipe.NewSet([]*Buffer{w, aqBuf, asBuf, idBuf, cwBuf, y})
		if err != nil {
			t.Fatal(err)
		}
		push := moePush{dim: dim, ffn: ffn, used: uint32(len(ids)), cap: 1}
		if err := set.Dispatch(uint32(dim/downOuts), unsafe.Pointer(&push)); err != nil {
			t.Fatal(err)
		}
		return append([]float32(nil), y.Floats()[:dim]...)
	}

	coarse := run(moeDownSPIRV, encodeQ4_0(weights), 18)
	fine := run(moeDownQ80SPIRV, encodeQ8_0(weights), 34)

	// The truth: the same eight products over the float weights and the
	// activation the card sees, summed with the routing weights.
	want := make([]float32, dim)
	for k, e := range ids {
		base := int(e) * dim * ffn
		for r := 0; r < dim; r++ {
			var s float32
			for i := 0; i < ffn; i++ {
				s += weights[base+r*ffn+i] * widened[k*ffn+i]
			}
			want[r] += cw[k] * s
		}
	}
	off := func(got []float32) float64 {
		var num, den float64
		for i := range want {
			diff := float64(got[i] - want[i])
			num += diff * diff
			den += float64(want[i]) * float64(want[i])
		}
		return math.Sqrt(num / den)
	}
	q4, q8 := off(coarse), off(fine)
	t.Logf("against the float answer: Q4_0 is %.4f off, Q8_0 is %.4f", q4, q8)
	if q8 >= q4 {
		t.Fatalf("the Q8_0 down projection is %.4f from the truth and the Q4_0 one %.4f — the finer form must be the nearer", q8, q4)
	}
	if q8 > 0.05 {
		t.Fatalf("the Q8_0 down projection sits %.4f from the float answer, which is not a quantization error", q8)
	}
}
