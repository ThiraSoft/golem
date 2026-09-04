package stt

import (
	"bytes"
	"context"
	"os"
	"strings"
	"sync"
	"testing"

	"github.com/ThiraSoft/golem/audio/decode"
	"github.com/ThiraSoft/golem/audio/resample"
	"github.com/ThiraSoft/golem/nn"
)

// TestStepBatchMatchesStep is the batched block against the one it replaces.
//
// Three streams at three different positions, so that nothing agrees by
// accident: the windows differ, the caches differ, and a block that mixed two
// streams' activations or read the wrong cache would show here rather than
// three hundred frames into a transcript.
func TestStepBatchMatchesStep(t *testing.T) {
	m := testModel(t)
	layer := m.weights.Layers[0]
	const n = 3

	alone := make([][]float32, n)
	together := make([][]float32, n)
	aloneKV := make([]*KV, n)
	batchKV := make([]*KV, n)
	for i := 0; i < n; i++ {
		alone[i] = make([]float32, DModel)
		together[i] = make([]float32, DModel)
		for j := range alone[i] {
			alone[i][j] = float32((j+7*i)%13) * 0.01
			together[i][j] = alone[i][j]
		}
		aloneKV[i] = filledKV(i, 40*i+11)
		batchKV[i] = filledKV(i, 40*i+11)
	}

	s := NewScratch()
	for i := 0; i < n; i++ {
		layer.Step(alone[i], aloneKV[i], s)
	}
	layer.StepBatch(together, batchKV, NewBatchScratch(n))

	for i := 0; i < n; i++ {
		for j := range alone[i] {
			if d := alone[i][j] - together[i][j]; d > 1e-4 || d < -1e-4 {
				t.Fatalf("stream %d element %d: alone %g, batched %g", i, j, alone[i][j], together[i][j])
			}
		}
		if aloneKV[i].Position != batchKV[i].Position {
			t.Fatalf("stream %d: position %d alone, %d batched", i, aloneKV[i].Position, batchKV[i].Position)
		}
	}
}

// TestStepBatchOneMatchesStep is the narrow case: a group carrying one stream
// must be the lone path exactly, because that is what a server with a single
// client runs.
func TestStepBatchOneMatchesStep(t *testing.T) {
	m := testModel(t)
	layer := m.weights.Layers[0]

	alone := make([]float32, DModel)
	together := make([]float32, DModel)
	for j := range alone {
		alone[j] = float32(j%13) * 0.01
		together[j] = alone[j]
	}
	kvA, kvB := filledKV(0, 300), filledKV(0, 300)

	layer.Step(alone, kvA, NewScratch())
	layer.StepBatch([][]float32{together}, []*KV{kvB}, NewBatchScratch(4))

	for j := range alone {
		if d := alone[j] - together[j]; d > 1e-4 || d < -1e-4 {
			t.Fatalf("element %d: alone %g, batched %g", j, alone[j], together[j])
		}
	}
}

// TestGroupTranscriptMatchesAlone is the whole point, end to end: three streams
// stepped together must say exactly what each of them says on its own.
func TestGroupTranscriptMatchesAlone(t *testing.T) {
	m := testModel(t)
	clip := speech(t)

	alone := transcribeAlone(t, m, clip)
	t.Logf("alone: %q", alone)

	const n = 3
	g := m.Group(context.Background(), n)
	defer g.Close()

	got := make([]string, n)
	var wg sync.WaitGroup
	for i := 0; i < n; i++ {
		live, err := g.Stream()
		if err != nil {
			t.Fatal(err)
		}
		wg.Add(1)
		go func(i int, live *Live) {
			defer wg.Done()
			var text strings.Builder
			done := make(chan struct{})
			go func() {
				defer close(done)
				for seg := range live.Text() {
					text.WriteString(seg.Text)
				}
			}()
			live.Write(clip)
			live.Close()
			<-done
			got[i] = strings.TrimSpace(text.String())
		}(i, live)
	}
	wg.Wait()

	for i, g := range got {
		if g != alone {
			t.Errorf("stream %d of the group:\n got %q\nwant %q", i, g, alone)
		}
	}
}

// TestGroupRefusesPastItsSize keeps the scratch honest: a batch wider than the
// columns it was built for has nowhere to put them.
func TestGroupRefusesPastItsSize(t *testing.T) {
	m := testModel(t)
	g := m.Group(context.Background(), 1)
	defer g.Close()

	first, err := g.Stream()
	if err != nil {
		t.Fatal(err)
	}
	defer first.Close()
	if _, err := g.Stream(); err == nil {
		t.Fatal("a group of one opened a second stream")
	}
}

func testModel(t *testing.T) *Model {
	t.Helper()
	dir := os.Getenv("GOLEM_STT")
	if dir == "" {
		t.Skip("GOLEM_STT not set")
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
	t.Cleanup(func() { m.Close() })
	return m
}

// filledKV is a cache with a past in it, at the position asked.
func filledKV(seed, position int) *KV {
	kv := &KV{
		K:        make([]float32, Context*NumHeads*HeadDim),
		V:        make([]float32, Context*NumHeads*HeadDim),
		Position: position,
	}
	for i := range kv.K {
		kv.K[i] = float32((i+seed*13)%31) * 0.003
		kv.V[i] = float32((i+seed*17)%29) * 0.004
	}
	return kv
}

func speech(t *testing.T) []float32 {
	t.Helper()
	file, err := os.ReadFile("../testdata/audio/speech.wav")
	if err != nil {
		t.Skipf("no speech.wav: %v", err)
	}
	samples, rate, channels, err := decode.Decode(bytes.NewReader(file))
	if err != nil {
		t.Fatal(err)
	}
	return resample.To(resample.Mono(samples, channels), rate, SampleRate)
}

func transcribeAlone(t *testing.T, m *Model, clip []float32) string {
	t.Helper()
	live := m.Stream(context.Background())
	var text strings.Builder
	done := make(chan struct{})
	go func() {
		defer close(done)
		for seg := range live.Text() {
			text.WriteString(seg.Text)
		}
	}()
	live.Write(clip)
	live.Close()
	<-done
	return strings.TrimSpace(text.String())
}
