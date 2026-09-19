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
