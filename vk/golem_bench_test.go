package vk

// What the golem product costs on the card, against the Q4_0 matvec of the same
// shape. The tiers read fewer bytes than Q4_0, and this is where it was first
// noticed that they all took the same time anyway.
//
// It measures one build against another package's, which is what a Go benchmark
// can do: the two are far enough apart that the card's clock drifting between
// them does not decide the answer. Two shapes of this kernel are not that far
// apart — TestGolemSweep and cmd/golemtune exist because comparing those needs
// one process and round-robin, and a benchmark run in sequence was believed
// twice before that was noticed.

import (
	"math/rand"
	"testing"
	"unsafe"

	"github.com/ThiraSoft/golem/nn"
)

func benchGolemShape(b *testing.B, kind nn.Quant, rows, cols, columns int) {
	d := open(b)
	defer d.Close()

	// Random bytes: the trellis decodes data-independently, and a real
	// encode of a 25M-weight matrix is minutes of Viterbi on the processor.
	data := make([]byte, rows*(nn.Matrix{Quant: kind, Cols: cols}).RowBytes())
	r := rand.New(rand.NewSource(23))
	r.Read(data)
	k, err := NewGolemKernels(d, kind)
	if err != nil {
		b.Fatal(err)
	}
	defer k.Close()
	act, err := d.Host(cols*columns*4, bufferUsageStorage)
	if err != nil {
		b.Fatal(err)
	}
	defer act.Close()
	out, err := d.Readback(rows*columns*4, bufferUsageStorage)
	if err != nil {
		b.Fatal(err)
	}
	defer out.Close()
	m, err := NewGolemMatrixOn(k, data, rows, cols, act, out)
	if err != nil {
		b.Fatal(err)
	}
	defer m.Close()

	const times = 64
	push := m.Push(0)
	set := m.Set(columns)
	if set == nil {
		b.Skip("no pass of that width")
	}
	bytes := len(data)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := set.DispatchTimes(m.Groups(columns), unsafe.Pointer(&push), times); err != nil {
			b.Fatal(err)
		}
	}
	b.StopTimer()
	seconds := b.Elapsed().Seconds() / float64(b.N) / times
	b.ReportMetric(float64(bytes)/seconds/1e9, "GB/s")
	b.ReportMetric(seconds*1e6, "us")
	b.ReportMetric(float64(rows)*float64(cols)*float64(columns)/seconds/1e9, "Gweight/s")
}

func BenchmarkGolemMatVec(b *testing.B) {
	// blk.0.ffn_gate of the 4B: 9728 rows by 2560.
	const rows, cols = 9728, 2560
	for _, kind := range []nn.Quant{nn.T3G, nn.T4G, nn.T5G} {
		for _, columns := range []int{1, 8} {
			b.Run(kind.String()+"/c"+itoa(columns), func(b *testing.B) {
				benchGolemShape(b, kind, rows, cols, columns)
			})
		}
	}
}

// BenchmarkQ40MatVecRef is the same shape through matvec_q40.comp, which is the
// bar: it reads 4.5 bits a weight where T4G reads 4.19 and T3G 3.25, so a
// bandwidth-bound golem kernel would have to be at least as fast.
func BenchmarkQ40MatVecRef(b *testing.B) {
	const rows, cols = 9728, 2560
	for _, columns := range []int{1, 8} {
		b.Run("c"+itoa(columns), func(b *testing.B) {
			d := open(b)
			defer d.Close()

			data := make([]byte, rows*rowBytesQ4_0(cols))
			r := rand.New(rand.NewSource(23))
			r.Read(data)
			spirv := map[int][]byte{1: matvecQ40SPIRV, 8: matvecQ40_8SPIRV}[columns]
			pipe, err := d.NewPipeline(spirv, 3, uint32(unsafe.Sizeof(matvecKPush{})))
			if err != nil {
				b.Fatal(err)
			}
			defer pipe.Close()
			w, err := d.Upload(data)
			if err != nil {
				b.Fatal(err)
			}
			defer w.Close()
			act, err := d.Host(cols*columns*4, bufferUsageStorage)
			if err != nil {
				b.Fatal(err)
			}
			defer act.Close()
			out, err := d.Readback(rows*columns*4, bufferUsageStorage)
			if err != nil {
				b.Fatal(err)
			}
			defer out.Close()
			set, err := pipe.NewSet([]*Buffer{w, act, out})
			if err != nil {
				b.Fatal(err)
			}
			defer set.Close()

			const times = 64
			push := matvecKPush{Dim: rows, FFN: cols}
			groups := uint32((rows + 15) / 16)
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				if err := set.DispatchTimes(groups, unsafe.Pointer(&push), times); err != nil {
					b.Fatal(err)
				}
			}
			b.StopTimer()
			seconds := b.Elapsed().Seconds() / float64(b.N) / times
			b.ReportMetric(float64(len(data))/seconds/1e9, "GB/s")
			b.ReportMetric(seconds*1e6, "us")
			b.ReportMetric(float64(rows)*float64(cols)*float64(columns)/seconds/1e9, "Gweight/s")
		})
	}
}

// BenchmarkQ40DotMatVecRef is the bar that matters, and it is not the one
// above.
//
// matvec_q40.comp reads float activations; matvec.comp reads them quantized to
// Q8_0 and spends one dotPacked4x8AccSatEXT on eight weights, which is the path
// most of a Q4_0 model's projections actually take. A trellis kernel that beat
// the float one and lost to this would still lose the model, so this is what a
// golem product has to be measured against.
func BenchmarkQ40DotMatVecRef(b *testing.B) {
	const rows, cols = 9728, 2560
	for _, columns := range []int{1, 8} {
		b.Run("c"+itoa(columns), func(b *testing.B) {
			d := open(b)
			defer d.Close()
			spirv := map[int][]byte{1: matvecSPIRV, 8: matvecWideSPIRV}[columns]
			if spirv == nil {
				b.Skip("no binary of that width")
			}
			pipe, err := d.NewPipeline(spirv, 4, uint32(unsafe.Sizeof(moePush{})))
			if err != nil {
				b.Fatal(err)
			}
			defer pipe.Close()

			data := make([]byte, rows*rowBytesQ4_0(cols))
			r := rand.New(rand.NewSource(23))
			r.Read(data)
			w, err := d.Upload(data)
			if err != nil {
				b.Fatal(err)
			}
			defer w.Close()
			// The activation in its Q8_0 form: values, then a scale and a
			// correction for every block of thirty-two.
			aq, err := d.Host(cols*columns, bufferUsageStorage)
			if err != nil {
				b.Fatal(err)
			}
			defer aq.Close()
			as, err := d.Host(2*cols/32*columns*4, bufferUsageStorage)
			if err != nil {
				b.Fatal(err)
			}
			defer as.Close()
			out, err := d.Readback(rows*columns*4, bufferUsageStorage)
			if err != nil {
				b.Fatal(err)
			}
			defer out.Close()
			set, err := pipe.NewSet([]*Buffer{w, aq, as, out})
			if err != nil {
				b.Fatal(err)
			}
			defer set.Close()

			const times = 64
			push := moePush{dim: rows, ffn: cols, used: 1}
			groups := uint32((rows + 15) / 16)
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				if err := set.DispatchTimes(groups, unsafe.Pointer(&push), times); err != nil {
					b.Fatal(err)
				}
			}
			b.StopTimer()
			seconds := b.Elapsed().Seconds() / float64(b.N) / times
			b.ReportMetric(float64(len(data))/seconds/1e9, "GB/s")
			b.ReportMetric(seconds*1e6, "us")
		})
	}
}
