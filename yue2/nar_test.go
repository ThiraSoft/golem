package yue2

import (
	"fmt"
	"math/rand"
	"path/filepath"
	"sync"
	"testing"
	"time"

	"github.com/ThiraSoft/golem/internal/heavy"
	"github.com/ThiraSoft/golem/nn"
	"github.com/ThiraSoft/golem/vk"
)

// The recording's first evaluation: 100 frames of noise at t = 1, whose raw
// timestep nar_logit_clamped saturates at 20, over the 3107 keys of the
// short replay's prefix.
const (
	narFrames = 100
	narRawT   = 20
)

// The NAR on the card with the recording's prefix, and the waypoints of its
// first evaluation, made once for the three tests that read them: the
// weights take seconds to widen and upload.
var narShared struct {
	once sync.Once
	skip string
	err  error
	n    *NAR
	got  map[string][]float32
}

func loadNARWeights(t testing.TB) *Backbone {
	t.Helper()
	b, err := OpenBackbone(filepath.Join(weightsDir(), "YuE2-3B-Q8_0.gguf"))
	if err != nil {
		t.Skipf("no backbone weights (%v): see ref/yue2/README.md", err)
	}
	return b
}

// prefixLayers cuts a recording of [layers][arLen][8][128] halves into its
// layers.
func prefixLayers(all []uint16, layers int) [][]uint16 {
	out := make([][]uint16, layers)
	per := len(all) / layers
	for l := range out {
		out[l] = all[l*per : (l+1)*per]
	}
	return out
}

func sharedNAR(t *testing.T) (*NAR, map[string][]float32) {
	t.Helper()
	fix := loadFixtures(t)
	arLen := len(fix.readInt32(t, "nar/ar_ids"))
	kAll, vAll := fix.readF16(t, "nar/kv_all_k.f16"), fix.readF16(t, "nar/kv_all_v.f16")
	noise := fix.read(t, "nar/noise")
	d := testDevice(t)
	s := &narShared
	s.once.Do(func() {
		b, err := OpenBackbone(filepath.Join(weightsDir(), "YuE2-3B-Q8_0.gguf"))
		if err != nil {
			s.skip = fmt.Sprintf("no backbone weights (%v): see ref/yue2/README.md", err)
			return
		}
		defer b.Close()
		if !d.Coopmat() {
			s.skip = "the NAR wants cooperative matrices"
			return
		}
		if s.n, s.err = NewNAR(d, b, narFrames, arLen); s.err != nil {
			return
		}
		if s.err = s.n.SetPrefix(prefixLayers(kAll, b.Cfg.Layers), prefixLayers(vAll, b.Cfg.Layers), arLen); s.err != nil {
			return
		}
		s.got, s.err = s.n.waypoints(noise, narFrames, narRawT)
	})
	if s.skip != "" {
		t.Skip(s.skip)
	}
	if s.err != nil {
		t.Fatal(s.err)
	}
	return s.n, s.got
}

// narCompare holds a waypoint to its bar against nar_exact, the recording of
// the same evaluation with yue2.cpp's products summing fp16 operands in
// float32 as golem's do, and logs its
// distance to nar, the default recording, whose products quantize their
// activations to Q8_1 on this card: that alone puts the default
// hidden_after_input 0.34% from the exact sum of its operands, where golem's
// lands (ref/yue2/README.md).
func narCompare(t *testing.T, fix *fixtures, name string, got []float32, bar float64) {
	t.Helper()
	t.Logf("%s: against Q8_1 activations %.4f%%", name, 100*rmsError(got, fix.read(t, "nar/"+name)))
	compareRMS(t, name, got, fix.read(t, "nar_exact/"+name), bar)
}

// TestNARInput holds the timestep embedding and the block's first hidden
// state, the latents' projection plus both embeddings.
func TestNARInput(t *testing.T) {
	fix := loadFixtures(t)
	_, got := sharedNAR(t)
	for _, name := range []string{"temb_t", "hidden_after_input"} {
		narCompare(t, fix, name, got[name], 1e-3)
	}
}

// TestNARLayers holds the first block's attention output and the stream
// after five of the 28 blocks against the recording. Measured: 0.019%,
// 0.015%, 0.019%, 0.042%, 0.054%, 0.051%.
func TestNARLayers(t *testing.T) {
	fix := loadFixtures(t)
	_, got := sharedNAR(t)
	for _, w := range []struct {
		name string
		bar  float64
	}{
		{"layer0_sa_output", 1e-2},
		{"hidden_after_layer0", 1e-2},
		{"hidden_after_layer7", 1e-2},
		{"hidden_after_layer14", 1e-2},
		{"hidden_after_layer21", 1e-2},
		{"hidden_after_layer27", 1e-2},
	} {
		narCompare(t, fix, w.name, got[w.name], w.bar)
	}
}

// TestNARVelocity is one whole evaluation, the program the ODE replays,
// against the recording's first velocity, which holds the same bar against
// the default recording too; the waypoints' pieces must come to the same
// answer to the bit.
func TestNARVelocity(t *testing.T) {
	fix := loadFixtures(t)
	n, got := sharedNAR(t)
	v, err := n.Velocity(fix.read(t, "nar/noise"), narFrames, narRawT)
	if err != nil {
		t.Fatal(err)
	}
	narCompare(t, fix, "v_first_0", v, 2e-2)
	compareRMS(t, "v_first_0 (Q8_1 activations)", v, fix.read(t, "nar/v_first_0"), 2e-2)
	if e := rmsError(got["velocity"], v); e != 0 {
		t.Errorf("the pieces and the whole differ by %.6f%%", 100*e)
	}
}

// BenchmarkNARVelocity times an evaluation at the size of the Guerreros song,
// 4870 frames over a prefix of 2600 keys, both made up, against yue2.cpp's
// 1030 ms (2.06 s a midpoint step of two evaluations), then says where an
// evaluation's time goes.
func BenchmarkNARVelocity(b *testing.B) {
	heavy.Skip(b, "a measurement over 4870 frames")
	const frames, arLen = 4870, 2600
	bb := loadNARWeights(b)
	d, err := vk.Open()
	if err != nil {
		b.Skipf("no Vulkan device: %v", err)
	}
	defer d.Close()
	if !d.Coopmat() {
		b.Skip("the NAR wants cooperative matrices")
	}
	n, err := NewNAR(d, bb, frames, arLen)
	bb.Close()
	if err != nil {
		b.Fatal(err)
	}
	defer n.Close()
	rng := rand.New(rand.NewSource(1))
	width := n.cfg.KVHeads * n.cfg.HeadDim
	k, v := make([][]uint16, n.cfg.Layers), make([][]uint16, n.cfg.Layers)
	for l := range k {
		k[l], v[l] = make([]uint16, arLen*width), make([]uint16, arLen*width)
		for i := range k[l] {
			k[l][i] = nn.Half(float32(rng.NormFloat64()))
			v[l][i] = nn.Half(float32(rng.NormFloat64()))
		}
	}
	if err := n.SetPrefix(k, v, arLen); err != nil {
		b.Fatal(err)
	}
	x := make([]float32, 64*frames)
	for i := range x {
		x[i] = float32(rng.NormFloat64())
	}
	b.Logf("weights %.2f GB, cache %.2f GB, arena %.2f GB",
		float64(n.WeightBytes())/1e9, float64(n.CacheBytes())/1e9, float64(n.ArenaBytes())/1e9)
	if _, err := n.Velocity(x, frames, narRawT); err != nil {
		b.Fatal(err)
	}
	b.ResetTimer()
	for range b.N {
		if _, err := n.Velocity(x, frames, narRawT); err != nil {
			b.Fatal(err)
		}
	}
	b.StopTimer()
	b.ReportMetric(float64(b.Elapsed().Milliseconds())/float64(b.N), "ms/eval")

	tl, err := d.NewTimeline(512)
	if err != nil {
		b.Fatal(err)
	}
	defer tl.Close()
	tokens := frames + 2
	p, err := n.k.Compile(func(r *vk.Recorder) {
		tl.Reset(r)
		n.timeline = tl
		tl.Stamp(r, "start")
		n.recordInput(r, tokens)
		for l := range n.layers {
			n.recordLayer(r, l, tokens, false)
		}
		n.recordHead(r, tokens)
		n.timeline = nil
	})
	if err != nil {
		b.Fatal(err)
	}
	defer p.Close()
	if err := p.Run(); err != nil {
		b.Fatal(err)
	}
	start := time.Now()
	if err := p.Run(); err != nil {
		b.Fatal(err)
	}
	rep, err := tl.Report(time.Since(start))
	if err != nil {
		b.Fatal(err)
	}
	b.Logf("one evaluation\n%s", rep)
}
