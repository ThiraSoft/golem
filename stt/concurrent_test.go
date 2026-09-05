package stt

// THROWAWAY — spike, poc/stt-vulkan-batch.
//
// How many microphones one processor carries at once.
//
// One frame of 80 ms costs 39.6 ms on eight threads, so a single stream eats
// about half the machine and two should saturate it. That is the arithmetic;
// this measures the machine. The number that matters is not the wall clock but
// the ratio — seconds of audio consumed per second elapsed — because a stream
// that falls under 1.0 is a microphone that falls behind and never catches up.
//
// The streams are independent by construction: each Live owns its codec state,
// its KV and its scratch, and the weights are read-only. So anything below
// linear scaling up to the core count is contention, not the model.

import (
	"context"
	"fmt"
	"os"
	"sync"
	"syscall"
	"testing"
	"time"

	"github.com/ThiraSoft/golem/internal/heavy"
	"github.com/ThiraSoft/golem/internal/kyutai/mimi"
	"github.com/ThiraSoft/golem/nn"
)

// concurrentSeconds is how much audio each stream carries, and it is past the
// window on purpose. Attention walks the visible past, so a stream that has
// been open eight seconds attends a hundred positions and one that has been
// open a minute attends the whole seven hundred and fifty: a short clip
// measures the cheap end and calls it the cost. Sixty seconds of audio reaches
// the steady state and spends most of its frames there.
const concurrentSeconds = 70

// TestConcurrentStreams is a capacity bench and not a test: it carries seventy
// seconds of audio an arm through eight configurations and takes ten minutes,
// which has no business in `go test ./stt/`. It runs when asked for by name.
func TestConcurrentStreams(t *testing.T) {
	heavy.Skip(t, "it puts a model on the card")
	dir := os.Getenv("GOLEM_STT")
	if dir == "" {
		t.Skip("GOLEM_STT not set")
	}
	if os.Getenv("GOLEM_STT_CAPACITY") == "" {
		t.Skip("GOLEM_STT_CAPACITY not set: this is a ten-minute capacity bench")
	}
	o, err := Locate(dir)
	if err != nil {
		t.Fatal(err)
	}
	o.Quant = nn.Q8_0
	m, err := Open(o)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()

	frames := concurrentSeconds * 1000 / 80
	audio := float64(frames) * 0.08

	// The card is a second model: UseVulkan is once per model, and the width it
	// is built for is the widest group below.
	const widest = 4
	card, err := Open(o)
	if err != nil {
		t.Fatal(err)
	}
	defer card.Close()
	onCard := card.UseVulkan(widest) == nil

	fmt.Printf("\n%-16s %-8s %10s %13s %12s %8s\n",
		"mode", "streams", "wall (s)", "x-real-time", "aggregate", "cores")
	for _, mode := range []string{"separate", "grouped", "grouped+vulkan"} {
		if mode == "grouped+vulkan" && !onCard {
			t.Log("no Vulkan device: the card rows are skipped")
			continue
		}
		use := m
		if mode == "grouped+vulkan" {
			use = card
		}
		for _, n := range []int{1, 2, 3, 4} {
			var g *Group
			open := func() *Live { return use.Stream(context.Background()) }
			if mode != "separate" {
				g = use.Group(context.Background(), n)
				open = func() *Live {
					live, err := g.Stream(context.Background())
					if err != nil {
						t.Fatal(err)
					}
					return live
				}
			}

			var wg sync.WaitGroup
			before := cpuSeconds()
			start := time.Now()
			for i := 0; i < n; i++ {
				live := open()
				wg.Add(1)
				go func() {
					defer wg.Done()
					go drain(live)
					frame := make([]float32, mimi.SamplesPerFrame)
					for f := 0; f < frames; f++ {
						live.Write(frame)
					}
					live.Close()
				}()
			}
			wg.Wait()
			wall := time.Since(start).Seconds()
			cores := (cpuSeconds() - before) / wall
			if g != nil {
				g.Close()
			}
			fmt.Printf("%-16s %-8d %10.2f %13.2f %12.2f %8.2f\n",
				mode, n, wall, audio/wall, float64(n)*audio/wall, cores)
		}
	}
	fmt.Println()
}

// cpuSeconds is user plus system time burned by this process so far. Divided
// by the wall clock it says how many cores the work actually kept busy, which
// is the one number that separates a saturated machine from a serialized one.
func cpuSeconds() float64 {
	var r syscall.Rusage
	if err := syscall.Getrusage(syscall.RUSAGE_SELF, &r); err != nil {
		return 0
	}
	sec := func(t syscall.Timeval) float64 { return float64(t.Sec) + float64(t.Usec)/1e6 }
	return sec(r.Utime) + sec(r.Stime)
}

func drain(l *Live) {
	for range l.Text() {
	}
}

// THROWAWAY — spike. What inside the trunk is actually the products.
//
// BenchmarkTrunk times sixteen blocks plus the final norm and the head
// together, and a plan to move "the trunk" onto the card has to know which of
// those three it is moving. The head is a bfloat16 8000x2048 read once a frame
// — thirty-three megabytes — and it is not one of the four matrices a batched
// product would carry.
func BenchmarkTrunkParts(b *testing.B) {
	m := benchModel(b, nn.Q8_0)
	kv := NewKV()
	scratch := NewScratch()
	x := make([]float32, DModel)
	logits := make([]float32, TextCard)
	for i := range x {
		x[i] = float32(i%13) * 0.01
	}

	b.Run("layers", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			for l, layer := range m.weights.Layers {
				layer.Step(x, kv[l], scratch)
			}
		}
	})
	b.Run("head", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			nn.RMSNormPlain(x, m.weights.OutNorm, NormEps)
			product(m.weights.Head, scratch.wide, x, logits)
		}
	})
}

// THROWAWAY — spike. Products against attention, at a full cache.
//
// BenchmarkTrunk starts from an empty KV, so its attention walks an average of
// half the positions the steady state walks. A microphone that has been open
// for a minute is at Context, and that is the cost a capacity plan has to use.
func BenchmarkLayerParts(b *testing.B) {
	m := benchModel(b, nn.Q8_0)
	scratch := NewScratch()
	x := make([]float32, DModel)
	for i := range x {
		x[i] = float32(i%13) * 0.01
	}
	layer := m.weights.Layers[0]

	full := &KV{
		K:        make([]float32, Context*NumHeads*HeadDim),
		V:        make([]float32, Context*NumHeads*HeadDim),
		Position: Context, // the steady state: the whole window is visible
	}
	for i := range full.K {
		full.K[i] = float32(i%31) * 0.003
		full.V[i] = float32(i%29) * 0.004
	}

	b.Run("step-full-kv", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			full.Position = Context
			layer.Step(x, full, scratch)
		}
	})
	b.Run("attend-only", func(b *testing.B) {
		q := make([]float32, DModel)
		copy(q, x)
		for i := 0; i < b.N; i++ {
			full.Position = Context
			_ = full.attend(q, scratch)
		}
	})
	b.Run("products-only", func(b *testing.B) {
		qkv := scratch.qkv
		out := scratch.out
		gate := scratch.gate
		ff := scratch.ff
		for i := 0; i < b.N; i++ {
			product(layer.InProj, scratch.wide, x, qkv)
			product(layer.OutProj, scratch.wide, x, out)
			product(layer.GateIn, scratch.wide, x, gate)
			product(layer.GateOut, scratch.deep, gate[:DimFF], ff)
		}
	})
}

// THROWAWAY — spike. What N streams cost when their activations share one
// read of the weights.
//
// The trunk's working set is fifty-four megabytes a layer in Q8_0, far past any
// cache, so N streams stepping separately read the same weights N times from
// main memory. nn.MatVecBatch already exists for exactly this — the row is the
// outer loop and the batch the inner one — and it is what a prompt uses. This
// asks whether concurrent microphones can use it too, which would be multi
// client without a line of Vulkan.
func BenchmarkBatchedProducts(b *testing.B) {
	m := benchModel(b, nn.Q8_0)
	layer := m.weights.Layers[0]

	for _, width := range []int{1, 2, 4, 8} {
		b.Run(fmt.Sprintf("separate-%d", width), func(b *testing.B) {
			one := nn.NewBatch(DModel, 1)
			deep := nn.NewBatch(DimFF, 1)
			fill(one.F[0])
			fill(deep.F[0])
			one.Quantize()
			deep.Quantize()
			qkv := make([]float32, 3*DModel)
			out := make([]float32, DModel)
			gate := make([]float32, 2*DimFF)
			ff := make([]float32, DModel)
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				for s := 0; s < width; s++ {
					layer.InProj.MatVec(one, qkv)
					layer.OutProj.MatVec(one, out)
					layer.GateIn.MatVec(one, gate)
					layer.GateOut.MatVec(deep, ff)
				}
			}
		})
		b.Run(fmt.Sprintf("batched-%d", width), func(b *testing.B) {
			wide := nn.NewBatch(DModel, width)
			deep := nn.NewBatch(DimFF, width)
			for c := 0; c < width; c++ {
				fill(wide.F[c])
				fill(deep.F[c])
			}
			wide.Quantize()
			deep.Quantize()
			qkv := columns(width, 3*DModel)
			out := columns(width, DModel)
			gate := columns(width, 2*DimFF)
			ff := columns(width, DModel)
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				layer.InProj.MatVecBatch(wide, qkv)
				layer.OutProj.MatVecBatch(wide, out)
				layer.GateIn.MatVecBatch(wide, gate)
				layer.GateOut.MatVecBatch(deep, ff)
			}
		})
	}
}

func fill(x []float32) {
	for i := range x {
		x[i] = float32(i%13) * 0.01
	}
}

func columns(n, width int) [][]float32 {
	ys := make([][]float32, n)
	for i := range ys {
		ys[i] = make([]float32, width)
	}
	return ys
}
