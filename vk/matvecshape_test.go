package vk

// What shape the mat-vec should be.
//
// LANES threads share an output row and OUTS rows go to a workgroup. Eight and
// sixteen is what the kernel has always been built at, and that pair was chosen
// once, for a different kernel, before the packed dot and before the K-quants.
// A token spends two thirds of itself in this kernel and reads the whole model
// through it, so the pair is worth measuring rather than inheriting.
//
// It reads its binaries off disk rather than embedding them: a sweep is an
// experiment and its losers do not belong in the repository. Compile them with
//
//	for L in 4 8 16 32 64; do for O in 4 8 16 32; do
//	  glslc -O -DQ4K -DLANES=${L}u -DOUTS=${O}u --target-env=vulkan1.1 \
//	    -fshader-stage=compute vk/shaders/matvec.comp -o $DIR/l${L}_o${O}.spv
//	done; done
//
// and point GOLEM_MATVEC_SWEEP at $DIR.

import (
	"math/rand"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"testing"
	"time"
	"unsafe"

	"github.com/ThiraSoft/golem/nn"
)

func TestMatVecShapeSweep(t *testing.T) {
	dir := os.Getenv("GOLEM_MATVEC_SWEEP")
	if dir == "" {
		t.Skip("set GOLEM_MATVEC_SWEEP to a directory of l<LANES>_o<OUTS>.spv binaries")
	}
	files, err := filepath.Glob(filepath.Join(dir, "l*_o*.spv"))
	if err != nil || len(files) == 0 {
		t.Skipf("no binaries in %s", dir)
	}
	sort.Strings(files)

	d, err := Open()
	if err != nil {
		t.Skip(err)
	}
	defer d.Close()

	// Qwen3-4B's gate and up are 9728 by 2560, and that matrix is fourteen
	// megabytes: read two hundred times over it never leaves this card's
	// sixty-four megabytes of last-level cache, and the sweep would rank the
	// shapes on a bandwidth generation never sees. The columns are the model's
	// and the rows are as many as it takes to be four hundred megabytes.
	const cols = 2560
	const rows = 262144
	rng := rand.New(rand.NewSource(4))
	data := make([]byte, rows*cols/nn.SuperBlock*144)
	rng.Read(data)
	m := nn.Matrix{Data: data, Quant: nn.Q4_K, Rows: rows, Cols: cols}
	sane(t, m, rng)

	x := make([]float32, cols)
	for i := range x {
		x[i] = rng.Float32()*2 - 1
	}
	q, scales, _ := q80Column(x)

	layout, err := quantLayout(nn.Q4_K, data, rows, cols)
	if err != nil {
		t.Fatal(err)
	}
	w, err := d.Upload(layout)
	if err != nil {
		t.Fatal(err)
	}
	defer w.Close()
	aq, err := d.Upload(unsafe.Slice((*byte)(unsafe.Pointer(&q[0])), len(q)*4))
	if err != nil {
		t.Fatal(err)
	}
	defer aq.Close()
	as, err := d.Upload(asBytes(scales))
	if err != nil {
		t.Fatal(err)
	}
	defer as.Close()
	y, err := d.Readback(rows*4, bufferUsageStorage)
	if err != nil {
		t.Fatal(err)
	}
	defer y.Close()

	// The answer the shape that ships gives, which every other shape has to
	// match: a sweep that reports a throughput without checking the answer
	// finds the binary that skips the most work.
	want := quantProductOnCard(t, d, m, q, scales, rows)

	type result struct {
		name string
		rate float64
	}
	var results []result
	for _, f := range files {
		lanes, outs, ok := shapeOf(filepath.Base(f))
		if !ok {
			continue
		}
		spirv, err := os.ReadFile(f)
		if err != nil {
			t.Fatal(err)
		}
		pipe, err := d.NewPipeline(spirv, 4, uint32(unsafe.Sizeof(moePush{})))
		if err != nil {
			t.Logf("%s: %v", filepath.Base(f), err)
			continue
		}
		set, err := pipe.NewSet([]*Buffer{w, aq, as, y})
		if err != nil {
			pipe.Close()
			t.Fatal(err)
		}
		push := moePush{dim: rows, ffn: cols, used: 1, split: 1}
		groups := uint32((rows + outs - 1) / outs)
		run := func(n int) {
			if err := d.Submit(func(r *Recorder) {
				for i := 0; i < n; i++ {
					r.Dispatch(set, groups, unsafe.Pointer(&push))
					r.Barrier()
				}
			}); err != nil {
				t.Fatal(err)
			}
		}
		run(2) // warm
		got := y.Floats()[:rows]
		for i := range want {
			if diff := got[i] - want[i]; diff > 1e-2 || diff < -1e-2 {
				t.Fatalf("%s: row %d answers %g where the shipped shape says %g",
					filepath.Base(f), i, got[i], want[i])
			}
		}
		const passes = 20
		start := time.Now()
		run(passes)
		took := time.Since(start)
		// The whole matrix is read once a pass, so the rate is what the card
		// managed to pull through this kernel.
		bytes := float64(len(layout)) * passes
		results = append(results, result{
			name: "lanes " + strconv.Itoa(lanes) + " outs " + strconv.Itoa(outs),
			rate: bytes / took.Seconds() / 1e9,
		})
		set.Close()
		pipe.Close()
	}
	sort.Slice(results, func(i, j int) bool { return results[i].rate > results[j].rate })
	for _, r := range results {
		t.Logf("%-22s %6.1f GB/s", r.name, r.rate)
	}
}

func shapeOf(name string) (lanes, outs int, ok bool) {
	name = strings.TrimSuffix(name, ".spv")
	parts := strings.Split(strings.TrimPrefix(name, "l"), "_o")
	if len(parts) != 2 {
		return 0, 0, false
	}
	l, err1 := strconv.Atoi(parts[0])
	o, err2 := strconv.Atoi(parts[1])
	return l, o, err1 == nil && err2 == nil
}
