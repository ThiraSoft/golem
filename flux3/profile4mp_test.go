package flux3

import (
	"testing"
	"time"

	"github.com/ThiraSoft/golem/internal/heavy"
	"github.com/ThiraSoft/golem/vk"
)

// Where a 4 MP step goes: each submission of the step timed alone, then one
// single block of the joint sequence with a stamp after every dispatch. A
// measurement, not a check.
func TestStepProfile4MP(t *testing.T) {
	heavy.Skip(t, "a measurement at 4 MP")
	path := ditPath()
	needFile(t, path)
	f := loadFixtures(t)
	d := device(t)
	m, err := OpenDiTSized(d, path, PrecisionFP16, 4456448/256, MaxRefs*RefTokens)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()

	for _, stage := range []string{"t2i4", "mi2i4"} {
		req := readStageRequest(t, f, stage)
		inX := f.read(t, stage+"/in.x")
		inT := f.read(t, stage+"/in.t")[0]
		inNext := f.read(t, stage+"/in.next")[0]
		inCtx := f.read(t, stage+"/in.ctx")
		if err := m.SetText(0, inCtx, len(inCtx)/CondWidth); err != nil {
			t.Fatal(err)
		}
		var refs [][]float32
		if len(req.RefShapes) > 0 {
			refs = refPlanes(f.read(t, stage+"/ref_latents"), req.RefShapes)
		}
		if err := m.SetRefs(0, refs, req.RefShapes); err != nil {
			t.Fatal(err)
		}
		h, w := req.Height/16, req.Width/16
		if _, err := m.Step(0, inX, h, w, inT, inNext); err != nil {
			t.Fatal(err)
		}
		start := time.Now()
		if _, err := m.Step(0, inX, h, w, inT, inNext); err != nil {
			t.Fatal(err)
		}
		step := time.Since(start)
		R, T := m.refTokens[0], m.ctxTokens[0]
		total := h*w + R + T
		progs, err := m.stepPrograms(0, h, w, R, T)
		if err != nil {
			t.Fatal(err)
		}
		var subs []time.Duration
		var sum time.Duration
		for _, p := range progs {
			start := time.Now()
			if err := p.p.Run(); err != nil {
				t.Fatal(err)
			}
			subs = append(subs, time.Since(start))
			sum += subs[len(subs)-1]
		}
		t.Logf("%s: %d tokens (%d image, %d ref, %d text); step %v; submissions %v, sum %v",
			stage, total, h*w, R, T, step, subs, sum)

		tl, err := d.NewTimeline(256)
		if err != nil {
			t.Fatal(err)
		}
		spans := []span{{start: 0, end: total, modOffset: m.modSingleImg}}
		p, err := m.k.Compile(func(r *vk.Recorder) {
			tl.Reset(r)
			m.timeline = tl
			tl.Stamp(r, "start")
			m.recordBlock(r, &m.singleBlocks[0], total, m.x, m.table, spans)
			m.timeline = nil
		})
		if err != nil {
			t.Fatal(err)
		}
		p.Run()
		start = time.Now()
		if err := p.Run(); err != nil {
			t.Fatal(err)
		}
		took := time.Since(start)
		rep, err := tl.Report(took)
		if err != nil {
			t.Fatal(err)
		}
		t.Logf("%s: one single block\n%s", stage, rep)
		p.Close()
		tl.Close()
	}
}
