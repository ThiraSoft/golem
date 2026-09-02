package vk

// Which part of the trellis product costs what.
//
// A matvec that is not bandwidth-bound is bound by something, and guessing
// which of the window, the hash and the activation load it is has been wrong
// twice. Each variant below removes exactly one of them and keeps the shape,
// the loads it does not remove and the reduction; the difference between two
// rows is that part's cost and nothing else.
//
// It compiles the kernel with -DABLATE, so the variants cannot drift from the
// shader that ships. What it found, on a 9728x2560 T4G product: the stream is
// 27 microseconds of the 80 the kernel took before the shared table went in,
// the 1MAD hash was 35 of them, and the activation was 4 — which is why the
// codebook moved into shared memory and why nothing was done about the
// activation.

import (
	"math/rand"
	"os"
	"os/exec"
	"path/filepath"
	"testing"
	"unsafe"

	"github.com/ThiraSoft/golem/nn"
)

func ablateSPIRV(tb testing.TB, kbits, ablate int) []byte {
	return buildGolemSPIRV(tb, kbits, ablate, 1)
}

// buildGolemSPIRV compiles the kernel with the switches a probe wants. It is
// the shader that ships, so a probe cannot drift from it.
func buildGolemSPIRV(tb testing.TB, kbits, ablate, bfe int) []byte {
	tb.Helper()
	dir := tb.TempDir()
	out := filepath.Join(dir, "a.spv")
	cmd := exec.Command("glslc", "-O",
		"-DKBITS="+itoa(kbits), "-DABLATE="+itoa(ablate), "-DBFE="+itoa(bfe),
		"--target-env=vulkan1.1", "-fshader-stage=compute",
		"shaders/matvec_t4g.comp", "-o", out)
	if b, err := cmd.CombinedOutput(); err != nil {
		tb.Fatalf("glslc: %v\n%s", err, b)
	}
	spirv, err := os.ReadFile(out)
	if err != nil {
		tb.Fatal(err)
	}
	return spirv
}

func BenchmarkGolemAblate(b *testing.B) {
	const rows, cols = 9728, 2560
	for _, kind := range []nn.Quant{nn.T3G, nn.T4G} {
		kbits := 4
		if kind == nn.T3G {
			kbits = 3
		}
		for _, v := range []struct {
			name   string
			ablate int
		}{
			{"full", 0},
			{"nohash", 1},   // the 1MAD, gone: the window is the value
			{"nowindow", 2}, // the window, gone: the stream word is the state
			{"nox", 3},      // the activation load, gone: one float for all
			{"stream", 4},   // nothing but reading the stream and summing it
			{"intfloor", 6}, // the window, the table, an integer add: the floor
		} {
			b.Run(kind.String()+"/"+v.name, func(b *testing.B) {
				d := open(b)
				defer d.Close()
				spirv := ablateSPIRV(b, kbits, v.ablate)
				// The shape of a one-column pass, so that an ablation is
				// measured against the kernel that ships and not another one.
				shape := GolemShapes()[1]
				pipe, err := d.NewPipelineSpec(spirv, 4, uint32(unsafe.Sizeof(golemPush{})), shape.Spec())
				if err != nil {
					b.Fatal(err)
				}
				defer pipe.Close()

				data := make([]byte, rows*(nn.Matrix{Quant: kind, Cols: cols}).RowBytes())
				rand.New(rand.NewSource(23)).Read(data)
				w, err := d.UploadTail(data, 16)
				if err != nil {
					b.Fatal(err)
				}
				defer w.Close()
				table, err := d.Upload(golemTable())
				if err != nil {
					b.Fatal(err)
				}
				defer table.Close()
				act, err := d.Host(cols*4, bufferUsageStorage)
				if err != nil {
					b.Fatal(err)
				}
				defer act.Close()
				out, err := d.Readback(rows*4, bufferUsageStorage)
				if err != nil {
					b.Fatal(err)
				}
				defer out.Close()
				set, err := pipe.NewSet([]*Buffer{w, table, act, out})
				if err != nil {
					b.Fatal(err)
				}
				defer set.Close()

				const times = 256
				push := golemPush{dim: rows, ffn: cols}
				per := uint32(shape.Threads / 8)
				groups := (uint32(rows) + per - 1) / per
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					if err := set.DispatchTimes(groups, unsafe.Pointer(&push), times); err != nil {
						b.Fatal(err)
					}
				}
				b.StopTimer()
				seconds := b.Elapsed().Seconds() / float64(b.N) / times
				b.ReportMetric(seconds*1e6, "us")
				b.ReportMetric(float64(len(data))/seconds/1e9, "GB/s")
			})
		}
	}
}
