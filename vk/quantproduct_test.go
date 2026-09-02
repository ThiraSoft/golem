package vk

// Every weight format vk/quantproduct.go offers, against nn's own reader of the
// same bytes and the same activation.
//
// One test for five formats, because that is what the file is for: a caller
// asks for a pipeline by type, lays the matrix out by type, and binds four
// buffers. If that door is right for Q4_0 it is right for the other four, and
// if it is wrong for one of them the failure names which.
//
// The matrices are random bytes in each format's own structure rather than a
// quantizer's output. That is enough here and it is not enough everywhere: what
// random bytes cannot catch is a *scale* unpacked from the wrong bits, which a
// quantizer's narrow range of scales would hide and a real checkpoint would
// not — vk/q4k_test.go is that test, on a real tensor, for the two formats
// whose six-bit scales have a history of being read the wrong way round. What
// this one catches is the rest: a block's offset, a nibble order, a correction
// term with the wrong sign.

import (
	"encoding/binary"
	"math"
	"math/rand"
	"testing"
	"unsafe"

	"github.com/ThiraSoft/golem/nn"
)

// q80Column is one column of activation in the form every kernel here reads:
// thirty-two magnitudes to a block as packed signed bytes, then the block's
// scales, then eight times each block's sum. The eight is Q4_0's recentring,
// which the other formats divide back out — shaders/matvec.comp says where.
func q80Column(x []float32) (q []uint32, scales, widened []float32) {
	nb := len(x) / nn.QuantBlock
	q = make([]uint32, nb*8)
	scales = make([]float32, 2*nb)
	widened = make([]float32, len(x))
	for b := 0; b < nb; b++ {
		block := x[b*32 : b*32+32]
		peak := float32(0)
		for _, v := range block {
			if a := float32(math.Abs(float64(v))); a > peak {
				peak = a
			}
		}
		if peak == 0 {
			continue
		}
		scale := nn.Widen(nn.Narrow(peak / 127))
		total := 0
		for j, v := range block {
			e := int(math.Round(float64(v) * float64(127/peak)))
			e = min(max(e, -128), 127)
			total += e
			q[b*8+j/4] |= uint32(uint8(int8(e))) << uint(8*(j%4))
			widened[b*32+j] = float32(e) * scale
		}
		scales[b] = scale
		scales[nb+b] = 8 * scale * float32(total)
	}
	return q, scales, widened
}

func TestQuantProductMatchesReference(t *testing.T) {
	d, err := Open()
	if err != nil {
		t.Skip(err)
	}
	defer d.Close()

	const rows, cols = 48, 512
	for _, f := range []struct {
		q     nn.Quant
		bytes int     // what one row of `cols` takes in the file
		tol   float32 // relative, against a dot of five hundred terms
	}{
		{nn.Q4_0, cols / nn.QuantBlock * 18, 1e-3},
		{nn.Q4_1, cols / nn.QuantBlock * 20, 1e-3},
		{nn.Q4_K, cols / nn.SuperBlock * 144, 1e-3},
		// Q5_K is the one packing here that computes rather than moves:
		// splitQ5_K multiplies d by each sub-block's scale and stores the
		// product as an fp16, where the reference keeps both factors and
		// multiplies in float32. That is one more rounding a block, and it is
		// the format's own cost rather than the kernel's.
		{nn.Q5_K, cols / nn.SuperBlock * 176, 1e-3},
		{nn.Q6_K, cols / nn.SuperBlock * 210, 1e-3},
	} {
		t.Run(f.q.String(), func(t *testing.T) {
			rng := rand.New(rand.NewSource(int64(f.q)))
			data := make([]byte, rows*f.bytes)
			rng.Read(data)
			// The magnitudes are random, but the scales are not left to chance:
			// a random fp16 is as likely to be an infinity as anything else,
			// and one of those makes every answer a NaN and says nothing.
			m := nn.Matrix{Data: data, Quant: f.q, Rows: rows, Cols: cols}
			sane(t, m, rng)

			x := make([]float32, cols)
			for i := range x {
				x[i] = rng.Float32()*2 - 1
			}
			// The reference reads the *quantized* activation, not the floats
			// behind it. Rounding the activation to eight bits is what the
			// card does to every projection here and it is not what is under
			// test; leaving it in the comparison drowns the thing that is,
			// because a K-quant's random six-bit scales make a dot of five
			// hundred terms cancel to a few per cent of its own magnitude.
			q, scales, xq := q80Column(x)
			want := make([]float32, rows)
			// The size of the terms, not of the answer. A row of random
			// six-bit scales cancels to a hundredth of what it adds up, and a
			// tolerance measured against the answer would then be asking the
			// card for more digits than the sum has.
			mag := make([]float32, rows)
			row := make([]float32, cols)
			for r := 0; r < rows; r++ {
				m.Row(r, row)
				var sum, size float32
				for i, v := range row {
					sum += v * xq[i]
					size += float32(math.Abs(float64(v * xq[i])))
				}
				want[r], mag[r] = sum, size
			}

			got := quantProductOnCard(t, d, m, q, scales, m.Rows)
			// What is left is the order of the sum and the fp16 roundings each
			// packing carries.
			for r := range want {
				if diff := float32(math.Abs(float64(got[r] - want[r]))); diff > f.tol*mag[r] {
					t.Fatalf("row %d: card %g, reference %g", r, got[r], want[r])
				}
			}
		})
	}
}

// sane replaces every fp16 scale in a random matrix with a small finite one, so
// that the test measures a product and not an infinity.
func sane(tb testing.TB, m nn.Matrix, rng *rand.Rand) {
	tb.Helper()
	put := func(at int) {
		binary.LittleEndian.PutUint16(m.Data[at:], nn.Narrow(rng.Float32()*0.05))
	}
	nb, nsb := m.Cols/nn.QuantBlock, m.Cols/nn.SuperBlock
	for r := 0; r < m.Rows; r++ {
		base := r * m.RowBytes()
		switch m.Quant {
		case nn.Q4_0:
			for b := 0; b < nb; b++ {
				put(base + b*18)
			}
		case nn.Q4_1:
			for b := 0; b < nb; b++ {
				put(base + b*20)
				put(base + b*20 + 2)
			}
		case nn.Q4_K:
			for sb := 0; sb < nsb; sb++ {
				put(base + sb*144)
				put(base + sb*144 + 2)
			}
		case nn.Q5_K:
			for sb := 0; sb < nsb; sb++ {
				put(base + sb*176)
				put(base + sb*176 + 2)
			}
		case nn.Q6_K:
			for sb := 0; sb < nsb; sb++ {
				put(base + sb*210 + 208)
			}
		}
	}
}

// quantProductOnCard is the door itself: lay the matrix out for its format, ask
// for the pipeline that reads it, bind four buffers, dispatch.
func quantProductOnCard(tb testing.TB, d *Device, m nn.Matrix, q []uint32, scales []float32, rows int) []float32 {
	tb.Helper()
	layout, err := quantLayout(m.Quant, m.Data, m.Rows, m.Cols)
	if err != nil {
		tb.Fatal(err)
	}
	products := newQuantProducts(d, d.Coopmat())
	defer products.Close()
	pipe, err := products.get(m.Quant)
	if err != nil {
		tb.Fatal(err)
	}

	w, err := d.Upload(layout)
	if err != nil {
		tb.Fatal(err)
	}
	defer w.Close()

	aq, err := d.Upload(unsafe.Slice((*byte)(unsafe.Pointer(&q[0])), len(q)*4))
	if err != nil {
		tb.Fatal(err)
	}
	defer aq.Close()
	as, err := d.Upload(asBytes(scales))
	if err != nil {
		tb.Fatal(err)
	}
	defer as.Close()
	y, err := d.Readback(m.Rows*4, bufferUsageStorage)
	if err != nil {
		tb.Fatal(err)
	}
	defer y.Close()

	set, err := pipe.NewSet([]*Buffer{w, aq, as, y})
	if err != nil {
		tb.Fatal(err)
	}
	push := moePush{dim: uint32(m.Rows), ffn: uint32(m.Cols), used: 1, split: 1}
	if err := d.Submit(func(r *Recorder) {
		r.Dispatch(set, groupsOf(m.Rows, matvecOuts), unsafe.Pointer(&push))
	}); err != nil {
		tb.Fatal(err)
	}
	out := make([]float32, m.Rows)
	copy(out, y.Floats()[:m.Rows])
	return out
}
