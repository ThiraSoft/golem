package stt

// Where the time of one frame actually goes.
//
// A CPU profile answers "which code burns cycles" and not "what is the clock
// waiting for": the goroutine that hands work to the pool and blocks is not
// sampled while it blocks, so a stage that is serial and slow can look small.
// These benchmarks time the stages on the wall, one at a time, which is the
// only way to see the serial ones.

import (
	"context"
	"os"
	"testing"

	"github.com/ThiraSoft/golem/internal/kyutai/mimi"
	"github.com/ThiraSoft/golem/nn"
)

// benchModel opens the checkpoint in the format asked, or skips.
func benchModel(b *testing.B, quant nn.Quant) *Model {
	b.Helper()
	dir := os.Getenv("GOLEM_STT")
	if dir == "" {
		b.Skip("GOLEM_STT not set")
	}
	o, err := Locate(dir)
	if err != nil {
		b.Fatal(err)
	}
	o.Quant = quant
	m, err := Open(o)
	if err != nil {
		b.Fatal(err)
	}
	b.Cleanup(func() { m.Close() })
	return m
}

// silence is one frame of nothing, which costs the model exactly what a frame
// of speech costs: the work per frame does not depend on what is in it.
func silence() []float32 { return make([]float32, mimi.SamplesPerFrame) }

// BenchmarkFrame is the number that decides whether the microphone keeps up.
// One frame is 80 ms of audio, so anything under 80 ms per iteration is faster
// than real time.
func BenchmarkFrame(b *testing.B) {
	for _, c := range []struct {
		name  string
		quant nn.Quant
	}{{"bf16", nn.BF16}, {"q8_0", nn.Q8_0}, {"q4_0", nn.Q4_0}} {
		b.Run(c.name, func(b *testing.B) {
			m := benchModel(b, c.quant)
			live := m.Stream(context.Background())
			go func() {
				for range live.Text() {
				}
			}()
			frame := silence()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				live.Write(frame)
			}
			b.StopTimer()
			live.Close()
		})
	}
}

// BenchmarkEncoder is the Mimi half alone: sound to latents, no trunk.
func BenchmarkEncoder(b *testing.B) {
	m := benchModel(b, nn.Q4_0)
	state := m.encoder.NewState()
	frame := silence()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		m.encoder.Push(frame, state)
	}
}

// BenchmarkQuantizer is the split RVQ alone: one latent to thirty-two codes.
// It is a single thread walking 32 codebooks of 2048 entries in 256 dimensions
// — sixteen million multiply-adds a frame, and nothing about it is parallel.
func BenchmarkQuantizer(b *testing.B) {
	m := benchModel(b, nn.Q4_0)
	latent := make([]float32, mimi.STTConfig.LatentDim)
	for i := range latent {
		latent[i] = float32(i%17) * 0.01
	}
	codes := make([]int, m.quantizer.Codebooks)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		m.quantizer.Encode(latent, codes)
	}
}

// BenchmarkTrunk is the sixteen blocks and the head, without the codec.
func BenchmarkTrunk(b *testing.B) {
	for _, c := range []struct {
		name  string
		quant nn.Quant
	}{{"bf16", nn.BF16}, {"q8_0", nn.Q8_0}, {"q4_0", nn.Q4_0}} {
		b.Run(c.name, func(b *testing.B) {
			m := benchModel(b, c.quant)
			kv := NewKV()
			scratch := NewScratch()
			x := make([]float32, DModel)
			logits := make([]float32, TextCard)
			for i := range x {
				x[i] = float32(i%13) * 0.01
			}
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				for l, layer := range m.weights.Layers {
					layer.Step(x, kv[l], scratch)
				}
				nn.RMSNormPlain(x, m.weights.OutNorm, NormEps)
				product(m.weights.Head, scratch.wide, x, logits)
			}
		})
	}
}
