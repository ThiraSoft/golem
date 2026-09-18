package vk

// Talking Head(?) Anime 3 on the card: building and running a graph.

import (
	"fmt"
	"time"
	"unsafe"
)

type tha3Kernel int

const (
	tha3Copy tha3Kernel = iota
	tha3Pose
	tha3Blend
	tha3Resize
	tha3Warp
	tha3DW
	tha3DWT
	tha3Head
	tha3PW
	tha3Stats
	tha3Norm
)

// tha3KernelSpec is one shader and the size of its push constants. Each ops
// file registers its own kernels in init, so a pipeline exists for every
// shader the package embeds and for no other.
type tha3KernelSpec struct {
	spirv []byte
	push  uintptr
}

var tha3Kernels = map[tha3Kernel]tha3KernelSpec{}

// THA3Runner is a built graph: its buffers, its pipelines and one recorded
// program per phase.
type THA3Runner struct {
	d           *Device
	g           *THA3Graph
	pipes       map[tha3Kernel]*Pipeline
	sets        map[tha3Kernel]*Set
	arena       *Buffer
	weights     *Buffer
	pose        *Buffer
	out         *Buffer
	trace       *Buffer
	programs    [2]*Program
	timeline    *Timeline
	outAt       []int
	traceAt     map[string][2]int // offset and length, in floats
	arenaFloats int
}

func groups256(n int) int { return (n + 255) / 256 }

func floatBytes(f []float32) []byte {
	return unsafe.Slice((*byte)(unsafe.Pointer(&f[0])), len(f)*4)
}

// Build plans the arena, uploads the weights, creates a pipeline and one
// descriptor set per kernel, and records the two phases. With profile set, a
// timeline stamps the pose phase wherever the graph asked for a Stamp.
func (g *THA3Graph) Build(d *Device, profile bool) (*THA3Runner, error) {
	x := &THA3Runner{d: d, g: g, pipes: map[tha3Kernel]*Pipeline{}, sets: map[tha3Kernel]*Set{}, traceAt: map[string][2]int{}}
	fail := func(err error) (*THA3Runner, error) {
		x.Close()
		return nil, err
	}
	x.arenaFloats = g.plan()
	bytes := uint64(max(x.arenaFloats, 1)) * 4
	if bytes >= 1<<32 {
		return fail(fmt.Errorf("vk: THA3 arena of %d bytes is past what one storage buffer addresses", bytes))
	}
	if free := d.DeviceLocalFree(); free > 0 && bytes > free {
		return fail(fmt.Errorf("vk: THA3 arena wants %d bytes, the device has %d free", bytes, free))
	}
	var err error
	if x.arena, err = d.Local(int(bytes), bufferUsageStorage|bufferUsageTransferSrc|bufferUsageTransferDst); err != nil {
		return fail(err)
	}
	weights := g.weights
	if len(weights) == 0 {
		weights = []float32{0}
	}
	if x.weights, err = d.Upload(floatBytes(weights)); err != nil {
		return fail(err)
	}
	if x.pose, err = d.Host(256, bufferUsageStorage); err != nil {
		return fail(err)
	}
	outFloats := 0
	for _, t := range g.outputs {
		x.outAt = append(x.outAt, outFloats)
		outFloats += t.C * t.H * t.W
	}
	if x.out, err = d.Readback(max(outFloats, 1)*4, bufferUsageStorage|bufferUsageTransferDst); err != nil {
		return fail(err)
	}
	traceFloats := 0
	for i, name := range g.traces {
		t := g.traceT[i]
		x.traceAt[name] = [2]int{traceFloats, t.C * t.H * t.W}
		traceFloats += t.C * t.H * t.W
	}
	if x.trace, err = d.Readback(max(traceFloats, 1)*4, bufferUsageStorage|bufferUsageTransferDst); err != nil {
		return fail(err)
	}
	for k, spec := range tha3Kernels {
		p, err := d.NewPipeline(spec.spirv, 3, uint32(spec.push))
		if err != nil {
			return fail(fmt.Errorf("vk: THA3 kernel %d: %w", k, err))
		}
		x.pipes[k] = p
		s, err := p.NewSet([]*Buffer{x.arena, x.weights, x.pose})
		if err != nil {
			return fail(err)
		}
		x.sets[k] = s
	}
	if profile && g.stamps > 0 {
		if x.timeline, err = d.NewTimeline(g.stamps + 1); err != nil {
			return fail(err)
		}
	}
	for phase := range x.programs {
		prog, err := d.Compile(func(r *Recorder) {
			if phase == THA3PosePhase && x.timeline != nil {
				x.timeline.Reset(r)
				x.timeline.Stamp(r, "start")
			}
			for _, o := range g.ops {
				if o.phase != phase {
					continue
				}
				if o.record != nil {
					o.record(r, x)
					r.Barrier()
				}
				if o.stamp != "" && phase == THA3PosePhase {
					x.timeline.Stamp(r, o.stamp)
				}
			}
		})
		if err != nil {
			return fail(err)
		}
		x.programs[phase] = prog
	}
	return x, nil
}

// at is where t starts in the arena, as a push constant wants it.
func (x *THA3Runner) at(t THA3Tensor) uint32 { return uint32(x.g.offset(t)) }

// dispatch records one kernel over groups by columns workgroups. Vulkan
// promises 65535 on each axis, which every shape of the poser stays under.
func (x *THA3Runner) dispatch(r *Recorder, k tha3Kernel, groups, columns int, push unsafe.Pointer) {
	if groups > 65535 || columns > 65535 {
		panic(fmt.Sprintf("vk: THA3 kernel %d wants %dx%d workgroups", k, groups, columns))
	}
	if groups == 0 || columns == 0 {
		return
	}
	s, ok := x.sets[k]
	if !ok {
		panic(fmt.Sprintf("vk: THA3 kernel %d has no pipeline", k))
	}
	r.DispatchColumns(s, uint32(groups), uint32(columns), push)
}

// Trace keeps a copy of t as it is at this point of the recording, readable
// by name with Waypoint after the pass has run.
func (g *THA3Graph) Trace(name string, t THA3Tensor) {
	for _, n := range g.traces {
		if n == name {
			panic("vk: THA3 waypoint traced twice: " + name)
		}
	}
	g.traces = append(g.traces, name)
	g.traceT = append(g.traceT, t)
	g.op([]THA3Tensor{t}, nil, func(r *Recorder, x *THA3Runner) {
		at := x.traceAt[name]
		r.CopyFrom(x.trace, at[0]*4, x.arena, x.g.offset(t)*4, at[1]*4)
	})
}

// Output copies t to the readback buffer at this point of the recording;
// Output(i) returns the i-th tensor so marked.
func (g *THA3Graph) Output(t THA3Tensor) {
	i := len(g.outputs)
	g.outputs = append(g.outputs, t)
	g.op([]THA3Tensor{t}, nil, func(r *Recorder, x *THA3Runner) {
		r.CopyFrom(x.out, x.outAt[i]*4, x.arena, x.g.offset(t)*4, t.C*t.H*t.W*4)
	})
}

// Write uploads an input tensor.
func (x *THA3Runner) Write(t THA3Tensor, data []float32) error {
	if !x.g.tensors[t.id].input {
		return fmt.Errorf("vk: THA3 Write to a tensor that is not an input")
	}
	if len(data) != t.C*t.H*t.W {
		return fmt.Errorf("vk: THA3 Write of %d floats to a tensor of %d", len(data), t.C*t.H*t.W)
	}
	return x.d.CopyInto(x.arena, x.g.offset(t)*4, floatBytes(data))
}

// SetPose writes the pose the next RunPose reads.
func (x *THA3Runner) SetPose(pose []float32) { copy(x.pose.Floats(), pose) }

// RunImage runs the image phase.
func (x *THA3Runner) RunImage() error { return x.programs[THA3ImagePhase].Run() }

// RunPose runs the pose phase and returns how long the submission took.
func (x *THA3Runner) RunPose() (time.Duration, error) {
	start := time.Now()
	err := x.programs[THA3PosePhase].Run()
	return time.Since(start), err
}

// Output returns a copy of the i-th output of the last run.
func (x *THA3Runner) Output(i int) []float32 {
	t := x.g.outputs[i]
	n := t.C * t.H * t.W
	out := make([]float32, n)
	copy(out, x.out.Floats()[x.outAt[i]:x.outAt[i]+n])
	return out
}

// Waypoint returns a copy of a traced tensor, or nil if none has that name.
func (x *THA3Runner) Waypoint(name string) []float32 {
	at, ok := x.traceAt[name]
	if !ok {
		return nil
	}
	out := make([]float32, at[1])
	copy(out, x.trace.Floats()[at[0]:at[0]+at[1]])
	return out
}

// Spans returns the last pose pass's timeline, or nil when not profiling.
func (x *THA3Runner) Spans() ([]Span, error) {
	if x.timeline == nil {
		return nil, nil
	}
	return x.timeline.Spans()
}

// ArenaBytes is the size of the planned arena.
func (x *THA3Runner) ArenaBytes() int { return x.arenaFloats * 4 }

// Close releases everything Build made. It is safe on a half-built runner.
func (x *THA3Runner) Close() {
	for i, p := range x.programs {
		if p != nil {
			p.Close()
			x.programs[i] = nil
		}
	}
	for k, s := range x.sets {
		s.Close()
		delete(x.sets, k)
	}
	for k, p := range x.pipes {
		p.Close()
		delete(x.pipes, k)
	}
	if x.timeline != nil {
		x.timeline.Close()
		x.timeline = nil
	}
	for _, b := range []**Buffer{&x.arena, &x.weights, &x.pose, &x.out, &x.trace} {
		if *b != nil {
			(*b).Close()
			*b = nil
		}
	}
}
