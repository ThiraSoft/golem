package stt

import (
	"bytes"
	"context"
	"errors"
	"math"
	"os"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/ThiraSoft/golem/audio/decode"
	"github.com/ThiraSoft/golem/audio/resample"
	"github.com/ThiraSoft/golem/internal/heavy"
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
	heavy.Skip(t, "it takes thirty-two seconds transcribing the same speech twice")
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
		live, err := g.Stream(context.Background())
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

	first, err := g.Stream(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	defer first.Close()
	if _, err := g.Stream(context.Background()); err == nil {
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

// TestGroupEvictsAStreamNobodyReads is the one that stops a bad client from
// stopping the good ones.
//
// A segment is handed to the stream's reader from inside the group's pass, and
// that pass is carrying every other stream. So a reader that has stopped must
// end its own stream rather than hold the pass: this fills the buffer, sends
// one more, and asks that the send returned and the stream is over.
func TestGroupEvictsAStreamNobodyReads(t *testing.T) {
	m := testModel(t)
	g := m.Group(context.Background(), 2)
	defer g.Close()

	live, err := g.Stream(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	defer live.Close()

	for i := 0; i < cap(live.textCh); i++ {
		live.textCh <- Segment{Text: "x"}
	}

	sent := make(chan struct{})
	go func() {
		defer close(sent)
		live.send(Segment{Text: "one too many"})
	}()
	select {
	case <-sent:
	case <-time.After(2 * time.Second):
		t.Fatal("send blocked on a reader that had stopped")
	}
	select {
	case <-live.ctx.Done():
	case <-time.After(2 * time.Second):
		t.Fatal("the stream was not ended")
	}
}

// TestGroupReleasesSlotOnCancel checks that a caller who walks away without
// closing gives the slot back anyway: a group that kept counting it would make
// every remaining stream pay the gathering window on every frame, for ever.
func TestGroupReleasesSlotOnCancel(t *testing.T) {
	m := testModel(t)
	g := m.Group(context.Background(), 1)
	defer g.Close()

	ctx, cancel := context.WithCancel(context.Background())
	if _, err := g.Stream(ctx); err != nil {
		t.Fatal(err)
	}
	if _, err := g.Stream(context.Background()); !errors.Is(err, ErrGroupFull) {
		t.Fatalf("a group of one opened a second stream: %v", err)
	}
	cancel()

	deadline := time.Now().Add(2 * time.Second)
	for {
		second, err := g.Stream(context.Background())
		if err == nil {
			second.Close()
			return
		}
		if time.Now().After(deadline) {
			t.Fatalf("the slot was never given back: %v", err)
		}
		time.Sleep(10 * time.Millisecond)
	}
}

// TestVulkanBlockMatchesProcessor is the card's block against the one it
// mirrors, at three streams sitting at three different positions.
//
// A product on the card is the same arithmetic in a different order, so this
// does not ask for equality: Q8_0 weights against a Q8_0 activation are exact
// in the integers and the accumulation is not, and sixteen of these compound.
// What it asks is that a block agree to a part in ten thousand of its own
// scale, which a transposed column or a cache read for the wrong stream misses
// by whole units.
func TestVulkanBlockMatchesProcessor(t *testing.T) {
	m := testModel(t)
	const n = 3
	if err := m.UseVulkan(n); err != nil {
		t.Skipf("no Vulkan trunk: %v", err)
	}
	if on, width := m.Vulkan(); !on || width != n {
		t.Fatalf("UseVulkan(%d) reports %v at %d", n, on, width)
	}
	layer := m.weights.Layers[0]

	alone := make([][]float32, n)
	onCard := make([][]float32, n)
	aloneKV := make([]*KV, n)
	cardKV := make([]*KV, n)
	for i := 0; i < n; i++ {
		alone[i] = make([]float32, DModel)
		onCard[i] = make([]float32, DModel)
		for j := range alone[i] {
			alone[i][j] = float32((j+7*i)%13) * 0.01
			onCard[i][j] = alone[i][j]
		}
		aloneKV[i] = filledKV(i, 40*i+11)
		cardKV[i] = filledKV(i, 40*i+11)
	}

	s := NewScratch()
	for i := 0; i < n; i++ {
		layer.Step(alone[i], aloneKV[i], s)
	}
	if err := m.card.StepBatchOn(0, layer, onCard, cardKV, NewBatchScratch(n)); err != nil {
		t.Fatal(err)
	}

	for i := 0; i < n; i++ {
		var peak float64
		for _, v := range alone[i] {
			if a := math.Abs(float64(v)); a > peak {
				peak = a
			}
		}
		tol := float32(peak) * 1e-4
		for j := range alone[i] {
			if d := alone[i][j] - onCard[i][j]; d > tol || d < -tol {
				t.Fatalf("stream %d element %d: processor %g, card %g (tolerance %g)",
					i, j, alone[i][j], onCard[i][j], tol)
			}
		}
		if aloneKV[i].Position != cardKV[i].Position {
			t.Fatalf("stream %d: position %d on the processor, %d on the card", i, aloneKV[i].Position, cardKV[i].Position)
		}
	}
}

// TestVulkanTranscriptMatchesProcessor is the whole path: a group whose trunk
// is on the card must say what the processor says, word for word.
func TestVulkanTranscriptMatchesProcessor(t *testing.T) {
	m := testModel(t)
	clip := speech(t)
	want := transcribeAlone(t, m, clip)

	card := testModel(t)
	const n = 2
	if err := card.UseVulkan(n); err != nil {
		t.Skipf("no Vulkan trunk: %v", err)
	}
	g := card.Group(context.Background(), n)
	defer g.Close()

	got := make([]string, n)
	var wg sync.WaitGroup
	for i := 0; i < n; i++ {
		live, err := g.Stream(context.Background())
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
	if err := g.Err(); err != nil {
		t.Fatal(err)
	}
	for i, g := range got {
		if g != want {
			t.Errorf("stream %d on the card:\n got %q\nwant %q", i, g, want)
		}
	}
}

// TestVulkanSharedByTwoGroups is the defect the card introduced into a shared
// model.
//
// A Model is read by everything at once, which cost nothing while the weights
// were only read. A card is written to: a product stages its columns into
// buffers that belong to the matrix. Two groups on one Model would write each
// other's activations and each read a mixture — no failure, just transcripts of
// the wrong sound. Two groups here transcribe the same clip at the same time,
// and both must say what the processor says.
func TestVulkanSharedByTwoGroups(t *testing.T) {
	m := testModel(t)
	clip := speech(t)
	want := transcribeAlone(t, m, clip)

	card := testModel(t)
	if err := card.UseVulkan(2); err != nil {
		t.Skipf("no Vulkan trunk: %v", err)
	}

	const groups = 2
	got := make([]string, groups)
	var wg sync.WaitGroup
	for i := 0; i < groups; i++ {
		g := card.Group(context.Background(), 2)
		defer g.Close()
		live, err := g.Stream(context.Background())
		if err != nil {
			t.Fatal(err)
		}
		wg.Add(1)
		go func(i int, g *Group, live *Live) {
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
			if err := g.Err(); err != nil {
				t.Error(err)
			}
			got[i] = strings.TrimSpace(text.String())
		}(i, g, live)
	}
	wg.Wait()

	for i, g := range got {
		if g != want {
			t.Errorf("group %d sharing the card:\n got %q\nwant %q", i, g, want)
		}
	}
}
