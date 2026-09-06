package vk

// Sweeping the tiled golem product's geometry.
//
// The tile is preprocessor arithmetic, so a geometry is a binary and not a
// specialization constant: the sweep compiles them outside Go and hands this
// test a directory of them. Names carry the geometry —
// tile_k<bits>_c<columns>_bm<BM>_bn<BN>_tm<TM>_tn<TN>_sw<SPLITW>.spv — because
// the workgroup count is the host's to compute and getting it from anywhere
// but the binary's own name is the trap vk/matmul.go's header names twice.
//
// Interleaved, rotating, and reduced by the fastest round, for the reason
// vk/golemtune.go gives: this card cannot read a difference under a tenth any
// other way.

import (
	"fmt"
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

type tileGeom struct {
	name                   string
	kbits, columns, bm, bn int
	tm, tn, splitw         int
	vec4, ablate           int
	spirv                  []byte
}

func parseTileName(path string) (tileGeom, error) {
	g := tileGeom{name: strings.TrimSuffix(filepath.Base(path), ".spv")}
	fields := map[string]*int{"k": &g.kbits, "c": &g.columns, "bm": &g.bm, "bn": &g.bn, "tm": &g.tm, "tn": &g.tn, "sw": &g.splitw, "v": &g.vec4, "a": &g.ablate}
	for _, part := range strings.Split(g.name, "_")[1:] {
		i := 0
		for i < len(part) && (part[i] < '0' || part[i] > '9') {
			i++
		}
		p, ok := fields[part[:i]]
		if !ok {
			return g, fmt.Errorf("unknown field %q in %s", part[:i], g.name)
		}
		v, err := strconv.Atoi(part[i:])
		if err != nil {
			return g, err
		}
		*p = v
	}
	var err error
	g.spirv, err = os.ReadFile(path)
	return g, err
}

// TestGolemTileSweep times every binary in GOLEM_TILE_DIR against one product.
func TestGolemTileSweep(t *testing.T) {
	dir := os.Getenv("GOLEM_TILE_DIR")
	if dir == "" {
		t.Skip("set GOLEM_TILE_DIR to a directory of tile_*.spv binaries")
	}
	// The 27B's gate-and-up in t3g, which is where four fifths of a .golem
	// prompt is spent. Not vk/golem_bench_test.go's 9728x2560: that one gives a
	// hundred and fifty-two workgroups on a sixty-four unit card, and the first
	// sweep of this kernel ran on it and picked a tile that is six per cent
	// behind in the engine. A tile sweep at a shape the card can fill in one
	// wave is a measurement of the launch.
	rows, cols := 34816, 5120
	if v := os.Getenv("GOLEM_TILE_ROWS"); v != "" {
		rows, _ = strconv.Atoi(v)
	}
	if v := os.Getenv("GOLEM_TILE_COLS"); v != "" {
		cols, _ = strconv.Atoi(v)
	}
	paths, err := filepath.Glob(filepath.Join(dir, "tile_*.spv"))
	if err != nil || len(paths) == 0 {
		t.Fatalf("no tile_*.spv in %s (%v)", dir, err)
	}
	sort.Strings(paths)

	d := open(t)
	defer d.Close()

	// One matrix a tier, uploaded once and shared by every geometry of it.
	widest := 0
	var geoms []tileGeom
	for _, p := range paths {
		g, err := parseTileName(p)
		if err != nil {
			t.Fatal(err)
		}
		widest = max(widest, g.columns)
		geoms = append(geoms, g)
	}

	kindOf := map[int]nn.Quant{3: nn.T3G, 4: nn.T4G, 5: nn.T5G}
	weights := map[int]*Buffer{}
	for _, g := range geoms {
		if weights[g.kbits] != nil {
			continue
		}
		data := make([]byte, rows*(nn.Matrix{Quant: kindOf[g.kbits], Cols: cols}).RowBytes())
		rand.New(rand.NewSource(23)).Read(data)
		b, err := d.UploadTail(data, golemReadTail)
		if err != nil {
			t.Fatal(err)
		}
		defer b.Close()
		weights[g.kbits] = b
	}
	table, err := d.Upload(golemTable())
	if err != nil {
		t.Fatal(err)
	}
	defer table.Close()
	act, err := d.Host(cols*widest*4, bufferUsageStorage)
	if err != nil {
		t.Fatal(err)
	}
	defer act.Close()
	out, err := d.Readback(rows*widest*4, bufferUsageStorage)
	if err != nil {
		t.Fatal(err)
	}
	defer out.Close()
	r := rand.New(rand.NewSource(7))
	for i := range act.Floats() {
		act.Floats()[i] = float32(r.NormFloat64())
	}

	type run struct {
		g      tileGeom
		set    *Set
		groups uint32
		best   time.Duration
	}
	var runs []*run
	for _, g := range geoms {
		pipe, err := d.NewPipeline(g.spirv, 4, golemPushSize)
		if err != nil {
			t.Fatalf("%s: %v", g.name, err)
		}
		defer pipe.Close()
		set, err := pipe.NewSet([]*Buffer{weights[g.kbits], table, act, out})
		if err != nil {
			t.Fatalf("%s: %v", g.name, err)
		}
		defer set.Close()
		// BN is clamped to the width of the pass in the shader, so it has to be
		// clamped here too: left unclamped a BN above COLUMNS divides to zero
		// column groups, the dispatch is empty, and it reports as the fastest
		// shape in the sweep. It did, the first time this ran.
		block := min(g.bn, g.columns)
		groups := uint32((rows+g.bm-1)/g.bm) * uint32(g.columns/block)
		runs = append(runs, &run{g: g, set: set, groups: groups, best: time.Hour})
	}

	push := golemPush{dim: uint32(rows), ffn: uint32(cols), col: 0}
	const rounds, times = 9, 16
	for round := 0; round < rounds; round++ {
		for i := range runs {
			r := runs[(i+round)%len(runs)]
			start := time.Now()
			if err := r.set.DispatchTimes(r.groups, unsafe.Pointer(&push), times); err != nil {
				t.Fatalf("%s: %v", r.g.name, err)
			}
			if d := time.Since(start) / times; d < r.best {
				r.best = d
			}
		}
	}
	sort.Slice(runs, func(i, j int) bool { return runs[i].best < runs[j].best })
	for _, r := range runs {
		per := float64(rows) * float64(cols) * float64(r.g.columns) / r.best.Seconds()
		t.Logf("%-44s %8.1f us  %7.0f Gweight/s  %6.2f us/column",
			r.g.name, float64(r.best.Microseconds()), per/1e9,
			r.best.Seconds()*1e6/float64(r.g.columns))
	}
}
