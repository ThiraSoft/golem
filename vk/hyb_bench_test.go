package vk

// Whether a trellis code that decodes two weights a state is faster than T3G's
// one, before any format is built on it.
//
// matvec_hyb.comp reads a row laid out byte for byte as T3G's — the same fifty
// bytes a path, the same steps — so the same random bytes feed both, and what
// differs is only that it cuts a window and reads the table once a pair. The
// shapes are Qwen3.8-27B's two, and each is uploaded several times over and
// cycled through, because one copy of a 36 MB matrix lives in the Infinity
// Cache and a product that reads from there is not the one a token pays for.

import (
	"encoding/binary"
	"fmt"
	"math"
	"math/rand"
	"os"
	"os/exec"
	"path/filepath"
	"testing"
	"time"
	"unsafe"

	"github.com/ThiraSoft/golem/internal/heavy"
	"github.com/ThiraSoft/golem/nn"
)

func buildHybSPIRV(tb testing.TB, mode, columns, half, stepout int, extra ...string) []byte {
	tb.Helper()
	out := filepath.Join(tb.TempDir(), "h.spv")
	cmd := exec.Command("glslc", "-O",
		"-DMODE="+itoa(mode), "-DCOLUMNS="+itoa(columns), "-DHALF="+itoa(half), "-DSTEPOUT="+itoa(stepout),
		"--target-env=vulkan1.1", "-fshader-stage=compute",
		"shaders/matvec_hyb.comp", "-o", out)
	cmd.Args = append(cmd.Args, extra...)
	if b, err := cmd.CombinedOutput(); err != nil {
		tb.Fatalf("glslc: %v\n%s", err, b)
	}
	spirv, err := os.ReadFile(out)
	if err != nil {
		tb.Fatal(err)
	}
	return spirv
}

func buildHybRowsSPIRV(tb testing.TB, columns int, extra ...string) []byte {
	tb.Helper()
	out := filepath.Join(tb.TempDir(), "r.spv")
	cmd := exec.Command("glslc", "-O", "-DCOLUMNS="+itoa(columns),
		"--target-env=vulkan1.1", "-fshader-stage=compute",
		"shaders/matvec_hyb_rows.comp", "-o", out)
	cmd.Args = append(cmd.Args, extra...)
	if b, err := cmd.CombinedOutput(); err != nil {
		tb.Fatalf("glslc: %v\n%s", err, b)
	}
	spirv, err := os.ReadFile(out)
	if err != nil {
		tb.Fatal(err)
	}
	return spirv
}

func buildT3GSPIRV(tb testing.TB, columns int) []byte { return buildT3GAblate(tb, columns, 0) }

func buildT3GAblate(tb testing.TB, columns, ablate int) []byte {
	tb.Helper()
	out := filepath.Join(tb.TempDir(), "t.spv")
	cmd := exec.Command("glslc", "-O", "-DKBITS=3", "-DCOLUMNS="+itoa(columns), "-DABLATE="+itoa(ablate), "-DPHASED="+itoa(map[bool]int{true: 0, false: 1}[ablate != 0]),
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

// hybTable is the step grid followed by a codebook of n pairs. Its contents
// only have to be finite: this measures time.
func hybTable(n int) []byte {
	out := golemTable()
	r := rand.New(rand.NewSource(5))
	for i := 0; i < 2*n; i++ {
		out = binary.LittleEndian.AppendUint32(out, math.Float32bits(float32(r.NormFloat64())))
	}
	return out
}

// t3gTable is the step grid followed by 1MAD's 4096 values.
func t3gTable() []byte {
	out := golemTable()
	for _, v := range nn.T4GTable() {
		out = binary.LittleEndian.AppendUint32(out, math.Float32bits(v))
	}
	return out
}

func TestHybAgainstT3G(t *testing.T) {
	heavy.Skip(t, "times kernels over hundreds of passes of the 27B's shapes")
	shapes := [][2]int{{17408, 5120}, {5120, 17408}}
	const rounds, times = 12, 8
	d := open(t)
	defer d.Close()

	type variant struct {
		name  string
		spirv func(columns int) []byte
		shape func(columns int) GolemShape
		table []byte
		per   int // rows a workgroup answers; zero for threads/8
	}
	def := func(c int) GolemShape { return golemDefaultShapes[c] }
	// What the 27B was measured to want inside a pass: see vk/golem.go.
	big := func(c int) GolemShape {
		switch c {
		case 1:
			return GolemShape{Threads: 256, Table: true, Prefetch: false, Copies: 1}
		case 2:
			return GolemShape{Threads: 256, Table: true, Prefetch: false, Copies: 1}
		}
		return golemDefaultShapes[c]
	}
	hyb := func(pre bool, copies int) func(int) GolemShape {
		return func(c int) GolemShape {
			return GolemShape{Threads: 256, Table: true, Prefetch: pre, Copies: copies}
		}
	}
	variants := []variant{
		{"T3G def", func(c int) []byte { return buildT3GSPIRV(t, c) }, def, golemTable(), 0},
		{"T3G 27b", func(c int) []byte { return buildT3GSPIRV(t, c) }, big, golemTable(), 0},
		{"T3G stream only", func(c int) []byte { return buildT3GAblate(t, c, 4) }, big, golemTable(), 0},
	}
	variants = append(variants, variant{
		name:  "HYB L14 Q10",
		spirv: func(c int) []byte { return buildHybSPIRV(t, 1, c, 1, 1, "-DLBITS=14", "-DQBITS=10") },
		shape: hyb(false, 1),
		table: hybTable(2048),
	})
	for _, v := range []struct{ rows, lanes int }{{2, 16}, {4, 16}, {2, 32}} {
		v := v
		variants = append(variants, variant{
			name: fmt.Sprintf("T3G rows %d lanes %d", v.rows, v.lanes),
			spirv: func(c int) []byte {
				return buildHybRowsSPIRV(t, c, "-DT3G=1", "-DROWS="+itoa(v.rows), "-DLANES="+itoa(v.lanes))
			},
			shape: hyb(false, 1),
			table: t3gTable(),
			per:   256 / v.lanes * v.rows,
		})
	}
	for _, v := range []struct{ rows, lanes int }{{2, 16}, {4, 16}, {2, 32}} {
		v := v
		variants = append(variants, variant{
			name: fmt.Sprintf("HYB rows %d lanes %d", v.rows, v.lanes),
			spirv: func(c int) []byte {
				return buildHybRowsSPIRV(t, c, "-DROWS="+itoa(v.rows), "-DLANES="+itoa(v.lanes))
			},
			shape: hyb(false, 1),
			table: hybTable(2048),
			per:   256 / v.lanes * v.rows,
		})
	}

	r := rand.New(rand.NewSource(11))
	for _, sh := range shapes {
		rows, cols := sh[0], sh[1]
		rowBytes := (nn.Matrix{Quant: nn.T3G, Cols: cols}).RowBytes()
		// Enough copies to be four times the Infinity Cache.
		ncopy := int(math.Ceil(256e6 / float64(rows*rowBytes)))
		var weights []*Buffer
		for i := 0; i < ncopy; i++ {
			data := make([]byte, rows*rowBytes)
			r.Read(data)
			// Steps inside the grid's sane middle, so no product overflows.
			for row := 0; row < rows; row++ {
				for b := 0; b < cols/64; b++ {
					data[row*rowBytes+b] = byte(200 + r.Intn(24))
				}
			}
			buf, err := d.UploadTail(data, golemReadTail)
			if err != nil {
				t.Fatal(err)
			}
			defer buf.Close()
			weights = append(weights, buf)
		}
		act, err := d.Host(8*cols*4, bufferUsageStorage)
		if err != nil {
			t.Fatal(err)
		}
		defer act.Close()
		out, err := d.Readback(8*rows*4, bufferUsageStorage)
		if err != nil {
			t.Fatal(err)
		}
		defer out.Close()

		widths := GolemWidths
		if w := os.Getenv("HYB_W"); w != "" {
			widths = []int{int(w[0] - '0')}
		}
		for _, columns := range widths {
			type entry struct {
				name   string
				sets   []*Set
				groups uint32
				best   time.Duration
			}
			var runs []*entry
			for _, v := range variants {
				shape := v.shape(columns)
				pipe, err := d.NewPipelineSpec(v.spirv(columns), 4, uint32(unsafe.Sizeof(golemPush{})), shape.Spec())
				if err != nil {
					t.Fatal(err)
				}
				defer pipe.Close()
				table, err := d.Upload(v.table)
				if err != nil {
					t.Fatal(err)
				}
				defer table.Close()
				e := &entry{name: v.name, best: time.Hour}
				for _, w := range weights {
					set, err := pipe.NewSet([]*Buffer{w, table, act, out})
					if err != nil {
						t.Fatal(err)
					}
					defer set.Close()
					e.sets = append(e.sets, set)
				}
				per := uint32(shape.Threads / 8)
				if v.per != 0 {
					per = uint32(v.per)
				}
				e.groups = (uint32(rows) + per - 1) / per
				runs = append(runs, e)
			}
			push := golemPush{dim: uint32(rows), ffn: uint32(cols)}
			n := times * len(weights)
			for round := 0; round <= rounds; round++ {
				for i := range runs {
					e := runs[(i+round)%len(runs)]
					start := time.Now()
					err := d.Submit(func(rec *Recorder) {
						for k := 0; k < n; k++ {
							if k > 0 {
								rec.Barrier()
							}
							rec.Dispatch(e.sets[k%len(e.sets)], e.groups, unsafe.Pointer(&push))
						}
					})
					if err != nil {
						t.Fatal(err)
					}
					if took := time.Since(start) / time.Duration(n); round > 0 && took < e.best {
						e.best = took
					}
				}
			}
			base := runs[0].best
			for _, e := range runs {
				if e.best < base && e.name[:3] == "T3G" {
					base = e.best
				}
			}
			for _, e := range runs {
				us := float64(e.best.Nanoseconds()) / 1000
				t.Logf("%5dx%-5d w%d  %-22s %8.2f us  %6.1f GB/s  %.3f of best T3G",
					rows, cols, columns, e.name, us,
					float64(rows*rowBytes)/e.best.Seconds()/1e9, e.best.Seconds()/base.Seconds())
			}
		}
	}
}
