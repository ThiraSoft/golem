package vk

// llama.cpp's own cooperative Q4_0 product, dispatched by golem.
//
// vk/shaders/matmul_coop.comp is nine and a half microseconds a column on the
// 12B's ffn_gate and llama.cpp is two, and the file that records that gap has
// closed every explanation it could think of inside the shader: the tile in
// four dimensions, the accumulator width, the shared-store width, the global
// load width, the dispatch order. So the question this asks is the other one.
// Take their shader exactly as their build produces it — mul_mm.comp compiled
// for Q4_0 with COOPMAT, fp16 accumulators, the aligned loads, and the
// specialisation constants ggml-vulkan chooses for this card baked in with
// spirv-opt — give it golem's buffers and golem's dispatch, and see what it
// does here.
//
// If it reaches two microseconds, the difference is in our shader and the
// shader is worth more work. If it does not, the difference is around it and
// no amount of work on matmul_coop.comp will find it.
//
// The two binaries are not built here. They come out of llama.cpp's own
// shader build, with the constants baked in as defaults so that no
// specialisation info has to reach vkCreateComputePipelines:
//
//	cd <llama.cpp>/build/ggml/src/ggml-vulkan/vulkan-shaders.spv
//	spirv-opt --set-spec-const-default-value \
//	  "0:256 1:128 2:128 3:32 4:64 5:64 6:2 7:16 8:16 9:16 10:64" \
//	  matmul_q4_0_f16_aligned_f16acc_cm1.spv -o llama_mul_mm_q4_0.spv
//	spirv-opt --set-spec-const-default-value \
//	  "0:128 1:64 2:64 3:32 4:64 5:32 6:2 7:16 8:16 9:16 10:64" \
//	  matmul_q4_0_f16_aligned_f16acc_cm1.spv -o llama_mul_mm_q4_0_m.spv
//
// The specialisation constants are l_warptile_mmq under ggml-vulkan's
// AMD-with-coopmat-on-RADV branch: BLOCK_SIZE 256, BM 128, BN 128, BK 32,
// WM 64, WN 64, WMITER 2, TM/TN/TK 16 — the device's only cooperative matrix
// shape — and WARP 64, which is the wave RADV runs compute at and the one the
// pipeline asks for.

import (
	_ "embed"
	"math"
	"testing"

	"github.com/ThiraSoft/golem/nn"
	"unsafe"
)

//go:embed shaders/llama_mul_mm_q4_0.spv
var llamaMulMMSPIRV []byte

// The same shader at ggml-vulkan's medium warptile — a hundred and twenty-eight
// threads over a sixty-four by sixty-four tile, which is the geometry golem's
// own cooperative kernel is closest to. It is here to ask whether their speed
// is the tile or the loop inside it.
//
//go:embed shaders/llama_mul_mm_q4_0_m.spv
var llamaMulMMMediumSPIRV []byte

// And the same shader with a float accumulator instead of an fp16 one, which
// is the variant ggml-vulkan uses when a caller asks for precision. It is
// here because it is the one golem could adopt: two per cent of relative
// error on a product is not something this engine's parity tests survive.
//
//go:embed shaders/llama_mul_mm_q4_0_f32acc.spv
var llamaMulMMF32AccSPIRV []byte

// A llamaTile is one of ggml-vulkan's warptiles: the binary with its
// specialisation constants baked in, and the workgroup denominators that go
// with them.
type llamaTile struct {
	name   string
	spirv  []byte
	bm, bn int
}

var llamaTiles = []llamaTile{
	{"l", llamaMulMMSPIRV, 128, 128},
	{"m", llamaMulMMMediumSPIRV, 64, 64},
	{"f32acc", llamaMulMMF32AccSPIRV, 128, 128},
}

// llamaMMPush is vk_mat_mat_push_constants. The shader reads sixteen of the
// seventeen — padded_n is the host's business — and they are passed whole.
type llamaMMPush struct {
	m, n, k                   uint32
	strideA, strideB, strideD uint32
	batchStrideA              uint32
	batchStrideB              uint32
	batchStrideD              uint32
	baseWorkGroupZ            uint32
	numBatches                uint32
	kSplit                    uint32
	ne02, ne12                uint32
	broadcast2, broadcast3    uint32
	paddedN                   uint32
}

// halfOf is the fp16 bit pattern of a float, which is how B reaches the
// shader: mul_mm's aligned variant reads it as f16vec4.
func halfOf(f float32) uint16 {
	bits := math.Float32bits(f)
	sign := uint16((bits >> 16) & 0x8000)
	exp := int((bits>>23)&0xFF) - 127 + 15
	man := bits & 0x7FFFFF
	switch {
	case bits&0x7FFFFFFF == 0:
		return sign
	case exp >= 0x1F:
		return sign | 0x7C00
	case exp <= 0:
		if exp < -10 {
			return sign
		}
		man |= 0x800000
		shift := uint(14 - exp)
		return sign | uint16(man>>shift)
	}
	return sign | uint16(exp)<<10 | uint16(man>>13)
}

func floatOfHalf(h uint16) float32 {
	sign := uint32(h&0x8000) << 16
	exp := uint32(h>>10) & 0x1F
	man := uint32(h & 0x3FF)
	switch {
	case exp == 0 && man == 0:
		return math.Float32frombits(sign)
	case exp == 0:
		e := 0
		for man&0x400 == 0 {
			man <<= 1
			e++
		}
		man &= 0x3FF
		exp = uint32(127 - 15 - e + 1)
	case exp == 0x1F:
		return math.Float32frombits(sign | 0x7F800000 | man<<13)
	default:
		exp += 127 - 15
	}
	return math.Float32frombits(sign | exp<<23 | man<<13)
}

// dequantQ4_0Row unpacks one row of a Q4_0 matrix, which the reference product
// needs and nothing else here does.
func dequantQ4_0Row(data []byte, row, cols int) []float32 {
	blocks := cols / 32
	out := make([]float32, cols)
	base := row * blocks * 18
	for b := 0; b < blocks; b++ {
		at := base + b*18
		scale := floatOfHalf(uint16(data[at]) | uint16(data[at+1])<<8)
		for j := 0; j < 16; j++ {
			q := data[at+2+j]
			out[b*32+j] = scale * (float32(q&0x0F) - 8)
			out[b*32+16+j] = scale * (float32(q>>4) - 8)
		}
	}
	return out
}

// llamaMM is one instance of their pipeline over one matrix.
type llamaMM struct {
	pipe                *Pipeline
	set                 *Set
	a, b, d, stage      *Buffer
	rows, cols, columns int
	tile                llamaTile
	push                llamaMMPush
}

func newLlamaMM(dev *Device, tile llamaTile, weights []byte, rows, cols, columns int, bhalf []uint16) (*llamaMM, error) {
	l := &llamaMM{rows: rows, cols: cols, columns: columns, tile: tile}
	var err error
	if l.a, err = dev.Upload(weights); err != nil {
		return nil, err
	}
	if l.b, err = dev.Local(columns*cols*2, bufferUsageStorage|bufferUsageTransferDst); err != nil {
		return nil, err
	}
	if l.d, err = dev.Local(rows*columns*4, bufferUsageStorage|bufferUsageTransferSrc); err != nil {
		return nil, err
	}
	if l.stage, err = dev.Host(columns*cols*2, bufferUsageTransferSrc); err != nil {
		return nil, err
	}
	raw := l.stage.Bytes()
	for i, h := range bhalf {
		raw[2*i], raw[2*i+1] = byte(h), byte(h>>8)
	}
	if l.pipe, err = dev.newPipeline(tile.spirv, 3, uint32(unsafe.Sizeof(llamaMMPush{})), coopmatWave, nil); err != nil {
		return nil, err
	}
	if l.set, err = l.pipe.NewSet([]*Buffer{l.a, l.b, l.d}); err != nil {
		return nil, err
	}
	l.push = llamaMMPush{
		m: uint32(rows), n: uint32(columns), k: uint32(cols),
		strideA: uint32(cols), strideB: uint32(cols), strideD: uint32(rows),
		batchStrideA: uint32(rows * cols), batchStrideB: uint32(columns * cols),
		batchStrideD:   uint32(rows * columns),
		baseWorkGroupZ: 0, numBatches: 1, kSplit: uint32(cols),
		ne02: 1, ne12: 1, broadcast2: 1, broadcast3: 1, paddedN: uint32(columns),
	}
	return l, nil
}

func (l *llamaMM) upload(r *Recorder) {
	r.CopyFrom(l.b, 0, l.stage, 0, len(l.stage.Bytes()))
	r.Barrier()
}

func (l *llamaMM) pass(r *Recorder) {
	x := uint32((l.rows + l.tile.bm - 1) / l.tile.bm)
	y := uint32((l.columns + l.tile.bn - 1) / l.tile.bn)
	r.DispatchColumns(l.set, x, y, unsafe.Pointer(&l.push))
}

func (l *llamaMM) Close() {
	for _, c := range []interface{ Close() }{l.set, l.pipe} {
		if c != nil {
			c.Close()
		}
	}
	for _, b := range []*Buffer{l.a, l.b, l.d, l.stage} {
		if b != nil {
			b.Close()
		}
	}
}

// llamaColumns is the activation side, as halves. Their kernel takes B in
// fp16 where golem's takes it in Q8_0, so this is the same batch the other
// tests use, rounded the way their pipeline would have rounded it.
func llamaColumns(width, n int) []uint16 {
	out := make([]uint16, n*width)
	for c := 0; c < n; c++ {
		for i := 0; i < width; i++ {
			v := float32(math.Sin(float64(i)*0.37+float64(c)*1.7)) * float32(1+(i+c)%17) * 0.11
			out[c*width+i] = halfOf(v)
		}
	}
	return out
}

// TestLlamaMulMMRuns is the guard on the benchmark below: a pipeline built at
// the wrong wave or fed the wrong strides still runs, and a benchmark of a
// kernel that answers nothing is a benchmark of nothing.
func TestLlamaMulMMRuns(t *testing.T) {
	g, m := namedQ4_0(t, "blk.0.ffn_gate.weight")
	defer g.Close()
	d := open(t)
	defer d.Close()
	if !d.Coopmat() {
		t.Skip("no cooperative matrices on this device")
	}

	for _, tile := range llamaTiles {
		t.Run(tile.name, func(t *testing.T) { llamaMulMMRuns(t, d, m, tile) })
	}
}

func llamaMulMMRuns(t *testing.T, d *Device, m nn.Matrix, tile llamaTile) {
	tolerance := 2e-2
	if tile.name == "f32acc" {
		tolerance = 1e-3
	}
	const columns = 256
	bh := llamaColumns(m.Cols, columns)
	l, err := newLlamaMM(d, tile, m.Data, m.Rows, m.Cols, columns, bh)
	if err != nil {
		t.Fatal(err)
	}
	defer l.Close()

	back, err := d.Readback(m.Rows*columns*4, bufferUsageTransferDst)
	if err != nil {
		t.Fatal(err)
	}
	defer back.Close()
	if err := d.Submit(func(r *Recorder) {
		l.upload(r)
		l.pass(r)
		r.Barrier()
		r.CopyFrom(back, 0, l.d, 0, m.Rows*columns*4)
	}); err != nil {
		t.Fatal(err)
	}
	got := back.Floats()

	// A handful of rows against the same product done here, which is enough:
	// a wave mismatch or a stride mismatch is wrong everywhere, not in a
	// corner.
	var worst, scale float64
	for _, row := range []int{0, 1, 17, m.Rows / 2, m.Rows - 1} {
		w := dequantQ4_0Row(m.Data, row, m.Cols)
		for _, col := range []int{0, 3, 128, columns - 1} {
			var want float64
			for i := 0; i < m.Cols; i++ {
				want += float64(w[i]) * float64(floatOfHalf(bh[col*m.Cols+i]))
			}
			have := float64(got[col*m.Rows+row])
			scale = math.Max(scale, math.Abs(want))
			worst = math.Max(worst, math.Abs(have-want))
		}
	}
	// Two per cent, which is not slack: this is their f16acc pipeline, and
	// three thousand eight hundred products summed in an fp16 accumulator
	// carry about that. golem's own cooperative kernel keeps a float
	// accumulator and lands a hundredth of it — see the header of
	// shaders/matmul_coop.comp, where the narrow accumulator was tried and
	// rejected for exactly this.
	if worst > tolerance*scale {
		t.Fatalf("their kernel is off by %g of a peak of %g", worst, scale)
	}
	t.Logf("%d rows by %d columns, worst gap %g of a peak of %g", m.Rows, columns, worst, scale)
}

// BenchmarkLlamaMulMMCold is BenchmarkMatMulCold's twin, copy for copy, so
// that the two numbers may be read against each other: the same matrix, the
// same twenty-four copies walked in turn so nothing comes out of cache, the
// same microseconds a column at the end.
func BenchmarkLlamaMulMMCold(b *testing.B) {
	const copies = 24
	for _, shape := range []struct{ name, tensor string }{
		{"down", "blk.0.ffn_down.weight"},
		{"gate", "blk.0.ffn_gate.weight"},
	} {
		for _, tile := range llamaTiles {
			for _, columns := range []int{64, 256} {
				tile := tile
				b.Run(shape.name+"/"+tile.name+itoa(columns), func(b *testing.B) {
					g, m := namedQ4_0(b, shape.tensor)
					defer g.Close()
					d := open(b)
					defer d.Close()
					if !d.Coopmat() {
						b.Skip("no cooperative matrices on this device")
					}

					bh := llamaColumns(m.Cols, columns)
					mms := make([]*llamaMM, copies)
					for k := range mms {
						l, err := newLlamaMM(d, llamaTiles[0], m.Data, m.Rows, m.Cols, columns, bh)
						if err != nil {
							b.Fatal(err)
						}
						defer l.Close()
						mms[k] = l
					}
					if err := d.Submit(func(r *Recorder) {
						for _, l := range mms {
							l.upload(r)
						}
					}); err != nil {
						b.Fatal(err)
					}
					b.ResetTimer()
					for i := 0; i < b.N; i++ {
						if err := d.Submit(func(r *Recorder) {
							for k, l := range mms {
								if k > 0 {
									r.Barrier()
								}
								l.pass(r)
							}
						}); err != nil {
							b.Fatal(err)
						}
					}
					b.StopTimer()
					seconds := b.Elapsed().Seconds() / float64(b.N) / copies
					b.ReportMetric(seconds*1e6/float64(columns), "us/column")
					b.ReportMetric(2*float64(m.Rows)*float64(m.Cols)*float64(columns)/seconds/1e12, "TFLOP/s")
				})
			}
		}
	}
}
