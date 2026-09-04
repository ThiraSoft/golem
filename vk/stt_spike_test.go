package vk

// THROWAWAY — spike, poc/stt-vulkan-batch.
//
// Does the STT trunk batched over concurrent streams beat the processor?
//
// One microphone already saturates eight cores (stt.TestConcurrentStreams), so
// a second client has nowhere to go on the processor. The card's answer to that
// is width: N streams stepping together turn the trunk's mat-vec into a mat-mat,
// which is the one regime where the card has an argument it does not have at
// batch one.
//
// This measures the four matrices of stt/layer.go at their real shapes and
// nothing else — no attention, no codec, no correctness. The products are the
// bulk of the trunk, so if they do not win, nothing downstream will.

import (
	"fmt"
	"os"
	"math/rand"
	"testing"
	"time"

	"github.com/ThiraSoft/golem/nn"
)

// spikeOnly keeps these out of an ordinary run. They measure and print; they
// assert nothing and cannot fail, so a suite that runs them only spends a
// card's time to learn nothing it did not already record.
func spikeOnly(t *testing.T) {
	if os.Getenv("GOLEM_VK_SPIKE") == "" {
		t.Skip("GOLEM_VK_SPIKE not set: these measure, they do not check")
	}
}

// The trunk of stt/layer.go: sixteen blocks of these four products.
var sttShapes = []struct {
	name       string
	rows, cols int
}{
	{"InProj", 6144, 2048},
	{"OutProj", 2048, 2048},
	{"GateIn", 11264, 2048},
	{"GateOut", 2048, 5632},
}

const sttLayers = 16

// frameBudget is what one 80 ms frame of audio may cost and still keep up.
const frameBudget = 80 * time.Millisecond

func TestSTTSpike(t *testing.T) {
	spikeOnly(t)
	d := open(t)
	defer d.Close()

	// The tiled product is built at these widths and no others; widths one to
	// four are the mat-vec kernel, a different shader and a different question.
	fmt.Printf("\n%-7s %-6s %10s %12s %12s %10s %10s\n",
		"width", "coop", "gpu/pass", "gpu trunk", "cpu trunk", "gpu x-rt", "cpu x-rt")

	for _, w := range []struct {
		width int
		coop  bool
	}{{8, false}, {32, false}, {32, true}, {64, false}, {64, true}, {128, true}} {
		width, coop := w.width, w.coop
		var gpuTotal, cpuTotal time.Duration
		ok := true
		for _, s := range sttShapes {
			g, err := gpuPass(t, d, s.rows, s.cols, width, coop)
			if err != nil {
				t.Logf("%s at width %d: %v", s.name, width, err)
				ok = false
				break
			}
			gpuTotal += g
			cpuTotal += cpuPass(t, s.rows, s.cols, width)
		}
		if !ok {
			continue
		}
		// One frame is one step through all sixteen blocks. The card pays one
		// submission for the whole frame, not one per product.
		gpuFrame := gpuTotal*sttLayers + 63*time.Microsecond
		cpuFrame := cpuTotal * sttLayers
		fmt.Printf("%-7d %-6t %10s %12s %12s %10.2f %10.2f\n",
			width, coop,
			(gpuTotal / time.Duration(len(sttShapes))).Round(time.Microsecond),
			gpuFrame.Round(time.Microsecond),
			cpuFrame.Round(time.Microsecond),
			float64(frameBudget)/float64(gpuFrame),
			float64(frameBudget)/float64(cpuFrame))
	}
	fmt.Println()
}

// gpuPass is the time of one product of this shape at this width, measured the
// way vk's own benchmarks do it: many passes inside one submission, so what is
// timed is the kernel and not the round trip.
func gpuPass(t *testing.T, d *Device, rows, cols, width int, coop bool) (time.Duration, error) {
	data := make([]byte, rows*(nn.Matrix{Quant: nn.Q4_0, Cols: cols}).RowBytes())
	rand.New(rand.NewSource(7)).Read(data)
	m, err := NewMatMulQuant(d, data, rows, cols, width, coop, nn.Q4_0)
	if err != nil {
		return 0, err
	}
	defer m.Close()

	b := nn.NewBatch(cols, 1)
	for i := range b.F[0] {
		b.F[0][i] = float32(i%13) * 0.01
	}
	b.Quantize()
	for c := 0; c < width; c++ {
		if err := m.SetColumn(c, b); err != nil {
			return 0, err
		}
	}
	const times = 64
	if err := m.RunTimes(4); err != nil { // warm the clocks
		return 0, err
	}
	best := time.Hour
	for try := 0; try < 3; try++ {
		start := time.Now()
		if err := m.RunTimes(times); err != nil {
			return 0, err
		}
		if e := time.Since(start) / times; e < best {
			best = e
		}
	}
	return best, nil
}

// cpuPass is the same product on the processor: width separate mat-vecs,
// because one stream already fills every core and a second one only queues.
func cpuPass(t *testing.T, rows, cols, width int) time.Duration {
	data := make([]byte, rows*(nn.Matrix{Quant: nn.Q4_0, Cols: cols}).RowBytes())
	rand.New(rand.NewSource(7)).Read(data)
	m := nn.Matrix{Data: data, Quant: nn.Q4_0, Rows: rows, Cols: cols}
	b := nn.NewBatch(cols, 1)
	for i := range b.F[0] {
		b.F[0][i] = float32(i%13) * 0.01
	}
	b.Quantize()
	out := make([]float32, rows)

	for i := 0; i < 4; i++ {
		m.MatVec(b, out)
	}
	best := time.Hour
	for try := 0; try < 3; try++ {
		const times = 16
		start := time.Now()
		for i := 0; i < times; i++ {
			m.MatVec(b, out)
		}
		if e := time.Since(start) / times; e < best {
			best = e
		}
	}
	return best * time.Duration(width)
}

// TestSTTRoundTrip is what the spike missed.
//
// Attention sits between InProj and OutProj, and it reads a cache that lives on
// the processor. So a card that carries only the four products does not run a
// block: it runs a quarter of one, hands the answer back, and waits. What that
// costs is not the kernel — RunTimes measures the kernel, with the inputs
// already in place and no answer copied out — but Run, which stages the
// activation, dispatches, and copies the answer back across the bus.
//
// Sixty-four of those a frame is the price of leaving attention where it is.
func TestSTTRoundTrip(t *testing.T) {
	spikeOnly(t)
	d := open(t)
	defer d.Close()

	const width = 8
	fmt.Printf("\n%-9s %12s %12s %12s\n", "shape", "kernel", "round trip", "overhead")
	var kernels, trips time.Duration
	for _, s := range sttShapes {
		kernel, err := gpuPass(t, d, s.rows, s.cols, width, false)
		if err != nil {
			t.Skip(err)
		}
		trip, err := gpuRoundTrip(t, d, s.rows, s.cols, width)
		if err != nil {
			t.Fatal(err)
		}
		kernels += kernel
		trips += trip
		fmt.Printf("%-9s %12s %12s %12s\n", s.name,
			kernel.Round(time.Microsecond), trip.Round(time.Microsecond),
			(trip - kernel).Round(time.Microsecond))
	}
	fmt.Printf("%-9s %12s %12s %12s\n", "block", kernels.Round(time.Microsecond),
		trips.Round(time.Microsecond), (trips - kernels).Round(time.Microsecond))
	fmt.Printf("%-9s %12s %12s %12s   (16 blocks, of an 80 ms frame)\n", "frame",
		(kernels * sttLayers).Round(time.Microsecond),
		(trips * sttLayers).Round(time.Microsecond),
		((trips - kernels) * sttLayers).Round(time.Microsecond))
	fmt.Println()
}

// gpuRoundTrip is one product paid for the way a processor-side attention makes
// you pay for it: staged, dispatched and read back, once per call.
func gpuRoundTrip(t *testing.T, d *Device, rows, cols, width int) (time.Duration, error) {
	data := make([]byte, rows*(nn.Matrix{Quant: nn.Q4_0, Cols: cols}).RowBytes())
	rand.New(rand.NewSource(7)).Read(data)
	m, err := NewMatMulQuant(d, data, rows, cols, width, false, nn.Q4_0)
	if err != nil {
		return 0, err
	}
	defer m.Close()

	b := nn.NewBatch(cols, 1)
	for i := range b.F[0] {
		b.F[0][i] = float32(i%13) * 0.01
	}
	b.Quantize()
	out := make([][]float32, width)
	for i := range out {
		out[i] = make([]float32, rows)
	}
	for c := 0; c < width; c++ {
		if err := m.SetColumn(c, b); err != nil {
			return 0, err
		}
	}
	for i := 0; i < 4; i++ {
		if err := m.Run(out); err != nil {
			return 0, err
		}
	}
	best := time.Hour
	for try := 0; try < 5; try++ {
		const times = 32
		start := time.Now()
		for i := 0; i < times; i++ {
			if err := m.Run(out); err != nil {
				return 0, err
			}
		}
		if e := time.Since(start) / times; e < best {
			best = e
		}
	}
	return best, nil
}
