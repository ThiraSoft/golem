package vk

import (
	"math"
	"math/rand"
	"testing"
	"time"

	"github.com/ThiraSoft/golem/internal/heavy"
)

// What the DiT's kernels do at 768 × 1024, in TFLOPS: a measurement, not a
// check. The products read X in fp16, as the DiT's step feeds them.
func TestK2Throughput(t *testing.T) {
	heavy.Skip(t, "a measurement")
	r := rand.New(rand.NewSource(1))
	const N = 3098
	time1 := func(k *K2, reps int, rec func(r *Recorder)) time.Duration {
		p, err := k.Compile(func(r *Recorder) {
			for i := 0; i < reps; i++ {
				rec(r)
			}
		})
		if err != nil {
			t.Fatal(err)
		}
		defer p.Close()
		p.Run()
		start := time.Now()
		if err := p.Run(); err != nil {
			t.Fatal(err)
		}
		return time.Since(start) / time.Duration(reps)
	}
	for _, s := range [][2]int{{16384, 6144}, {6144, 16384}, {6144, 6144}, {1536, 6144}} {
		outs, ins := s[0], s[1]
		k := newK2(t, N*ins+N*outs, nil)
		w, _ := k.AddWeights(randomFP8(r, outs*ins))
		d := time1(k, 5, func(rec *Recorder) {
			k.MM(rec, w, true, K2MM{Outputs: uint32(outs), Inputs: uint32(ins), Cols: N, XStride: uint32(ins), Y: uint32(N * ins), YStride: uint32(outs), Bias: K2None, Scale: 1, XHalves: 1})
		})
		t.Logf("mm %d×%d×%d: %v, %.1f TFLOPS", outs, ins, N, d, 2*float64(outs*ins*N)/d.Seconds()/1e12)
	}
	k := newK2(t, N*6144*3, nil)
	a := K2Attn{Q: 0, QStride: 6144, K: N * 6144, KStride: 1536, V: N*6144 + N*1536, VStride: 1536, O: 2 * N * 6144, OStride: 6144,
		Queries: N, Keys: N, Group: 4, Scale: float32(1 / math.Sqrt(128)), Heads: 48, Seqs: 1, HeadDim: 128}
	d := time1(k, 3, func(rec *Recorder) { k.Attn(rec, a) })
	t.Logf("attention %d tokens, 48 heads: %v, %.1f TFLOPS (three products counted)", N, d, 48*float64(N)*float64(N)*128*2*3/d.Seconds()/1e12)
}

// What the attention does at the DiTs' sizes, in TFLOPS: FLUX 3's 24 heads
// at 1, 4 and 4 MP with four references, Krea 2's 48 over twelve at
// 768 × 1024. A measurement, not a check.
func TestK2AttnThroughput(t *testing.T) {
	heavy.Skip(t, "a measurement")
	for _, c := range []struct {
		name            string
		heads, group, n int
	}{
		{"flux3 1 MP", 24, 1, 4176},
		{"flux3 4 MP", 24, 1, 16544},
		{"flux3 4 MP + 4 refs", 24, 1, 32755},
		{"krea2 768x1024", 48, 4, 3098},
	} {
		qs, ks := c.heads*128, c.heads/c.group*128
		n := c.n
		k := newK2(t, n*(2*qs+2*ks), nil)
		k.ReserveAttention(n, c.heads/c.group)
		a := K2Attn{Q: 0, QStride: uint32(qs), K: uint32(n * qs), KStride: uint32(ks), V: uint32(n*qs + n*ks), VStride: uint32(ks),
			O: uint32(n*qs + 2*n*ks), OStride: uint32(qs), Queries: uint32(n), Keys: uint32(n), Group: uint32(c.group),
			Scale: float32(1 / math.Sqrt(128)), Heads: c.heads, Seqs: 1, HeadDim: 128}
		p, err := k.Compile(func(r *Recorder) { k.Attn(r, a) })
		if err != nil {
			t.Fatal(err)
		}
		// A second of it first: the card's clock climbs over the first
		// hundreds of milliseconds, and the first shape would pay for it.
		for start := time.Now(); time.Since(start) < time.Second; {
			p.Run()
		}
		best := time.Hour
		for i := 0; i < 5; i++ {
			start := time.Now()
			if err := p.Run(); err != nil {
				t.Fatal(err)
			}
			best = min(best, time.Since(start))
		}
		p.Close()
		t.Logf("%s, %d tokens, %d heads: %v, %.1f TFLOPS", c.name, n, c.heads, best, 4*float64(c.heads)*float64(n)*float64(n)*128/best.Seconds()/1e12)
	}
}
