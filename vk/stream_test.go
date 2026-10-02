package vk

import (
	"math"
	"testing"
	"unsafe"
)

// TestStreamsInFlight puts two streams on the card before waiting for either,
// and checks each buffer once its own stream has been waited for.
func TestStreamsInFlight(t *testing.T) {
	d := open(t)
	defer d.Close()
	var bufs [2]*Buffer
	var streams [2]*Stream
	for i := range bufs {
		b, err := d.Readback(1<<20, UsageStorage|UsageTransferDst)
		if err != nil {
			t.Fatal(err)
		}
		defer b.Close()
		s, err := d.Stream()
		if err != nil {
			t.Fatal(err)
		}
		defer s.Close()
		bufs[i], streams[i] = b, s
	}
	for round := 0; round < 3; round++ {
		for i, s := range streams {
			b, want := bufs[i], float32(round*10+i)
			if err := s.Start(func(r *Recorder) { r.Fill(b, math.Float32bits(want)) }); err != nil {
				t.Fatal(err)
			}
		}
		for i := len(streams) - 1; i >= 0; i-- {
			if err := streams[i].Wait(); err != nil {
				t.Fatal(err)
			}
			if !streams[i].Done() {
				t.Fatalf("round %d: stream %d not done after Wait", round, i)
			}
			want := float32(round*10 + i)
			for j, x := range bufs[i].Floats() {
				if x != want {
					t.Fatalf("round %d: buffer %d float %d = %v, want %v", round, i, j, x, want)
				}
			}
		}
	}
}

// TestTransferBarrier orders a copy after the fill it reads, inside a stream.
func TestTransferBarrier(t *testing.T) {
	d := open(t)
	defer d.Close()
	src, err := d.Readback(4096, UsageStorage|UsageTransferSrc|UsageTransferDst)
	if err != nil {
		t.Fatal(err)
	}
	defer src.Close()
	dst, err := d.Readback(4096, UsageStorage|UsageTransferDst)
	if err != nil {
		t.Fatal(err)
	}
	defer dst.Close()
	s, err := d.Stream()
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	err = s.Start(func(r *Recorder) {
		r.Fill(src, math.Float32bits(2.5))
		r.TransferBarrier()
		r.Copy(dst, 0, src, 4096)
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := s.Wait(); err != nil {
		t.Fatal(err)
	}
	for i, x := range dst.Floats() {
		if x != 2.5 {
			t.Fatalf("float %d = %v", i, x)
		}
	}
}

// TestStreamBesideStart checks that a stream and the device's own Start/Wait
// can be used on the same device, and that Device.Wait covers the stream.
func TestStreamBesideStart(t *testing.T) {
	d := open(t)
	defer d.Close()
	a, err := d.Readback(4096, UsageStorage|UsageTransferDst)
	if err != nil {
		t.Fatal(err)
	}
	defer a.Close()
	b, err := d.Readback(4096, UsageStorage|UsageTransferDst)
	if err != nil {
		t.Fatal(err)
	}
	defer b.Close()
	s, err := d.Stream()
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	if err := s.Start(func(r *Recorder) { r.Fill(a, math.Float32bits(1)) }); err != nil {
		t.Fatal(err)
	}
	if err := d.Start(func(r *Recorder) { r.Fill(b, math.Float32bits(2)) }); err != nil {
		t.Fatal(err)
	}
	if err := d.Wait(); err != nil {
		t.Fatal(err)
	}
	if !s.Done() {
		t.Fatal("stream not done after Device.Wait")
	}
	if a.Floats()[1023] != 1 || b.Floats()[1023] != 2 {
		t.Fatalf("got %v and %v", a.Floats()[1023], b.Floats()[1023])
	}
	if err := s.Wait(); err != nil {
		t.Fatal(err)
	}
}

// TestComputeQueues opens with compute queues, uploads on the main queue and
// reads the upload back from streams on the compute ones, all in flight at
// once: the buffers are shared between the two families.
func TestComputeQueues(t *testing.T) {
	d, err := OpenWith(Options{ComputeQueues: 4})
	if err != nil {
		t.Skipf("no Vulkan compute device: %v", err)
	}
	defer d.Close()
	if len(d.asyncQueues) == 0 {
		t.Skip("no compute-only queue family")
	}
	const n = 8
	src := make([]float32, 1024)
	for i := range src {
		src[i] = float32(i)
	}
	var streams [n]*Stream
	var locals, outs [n]*Buffer
	for i := range n {
		if locals[i], err = d.Local(4096, UsageStorage|UsageTransferSrc|UsageTransferDst); err != nil {
			t.Fatal(err)
		}
		defer locals[i].Close()
		if err := d.CopyInto(locals[i], 0, unsafe.Slice((*byte)(unsafe.Pointer(&src[0])), len(src)*4)); err != nil {
			t.Fatal(err)
		}
		if outs[i], err = d.Readback(4096, UsageStorage|UsageTransferDst); err != nil {
			t.Fatal(err)
		}
		defer outs[i].Close()
		if streams[i], err = d.Stream(); err != nil {
			t.Fatal(err)
		}
		defer streams[i].Close()
	}
	for i, s := range streams {
		l, o := locals[i], outs[i]
		if err := s.Start(func(r *Recorder) { r.Copy(o, 0, l, 4096) }); err != nil {
			t.Fatal(err)
		}
	}
	for i, s := range streams {
		if err := s.Wait(); err != nil {
			t.Fatal(err)
		}
		for j, x := range outs[i].Floats() {
			if x != float32(j) {
				t.Fatalf("stream %d float %d = %v", i, j, x)
			}
		}
	}
}
