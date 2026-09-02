package vk

// Finding the shape of the trellis matvec that this card is fastest at.
//
// vk/golem.go carries one answer, measured on one card. This is how another
// card gets its own: build every candidate shape of every pass width against a
// product the size of a real one, time them interleaved, and print what to set
// GOLEM_MATVEC_SHAPE to. cmd/golemtune is the command around it, and
// TestGolemSweep runs it here.
//
// Interleaved, rotating, and reduced by the fastest round, and all three are
// load-bearing. A sweep that runs its cases one after another cannot compare
// them: this card's clock drifts far enough over a few minutes that the same
// two shapes came out 16% apart one way and 3% apart the other. And a sweep
// that interleaves them in a fixed order cannot either: whichever shape sits
// fourth in the loop comes out three to six per cent ahead of whichever sits
// third, the same binary both ways round. Rotating the order between rounds is
// what makes a difference of a few per cent readable at all.

import (
	"fmt"
	"math/rand"
	"sort"
	"strings"
	"time"
	"unsafe"

	"github.com/ThiraSoft/golem/nn"
)

// GolemTuneShapes are the shapes worth timing. They are not every combination:
// a workgroup narrower than 128 leaves the card idle and one wider than 256 was
// measured to gain nothing, and a read-ahead without the table has never won a
// width. Adding one here is free — every shape answers the same numbers.
var GolemTuneShapes = []GolemShape{
	{Threads: 128, Table: false, Prefetch: false},
	{Threads: 256, Table: false, Prefetch: false},
	{Threads: 128, Table: true, Prefetch: false},
	{Threads: 256, Table: true, Prefetch: false},
	{Threads: 128, Table: true, Prefetch: true},
	{Threads: 256, Table: true, Prefetch: true},
	{Threads: 512, Table: true, Prefetch: false},
	{Threads: 512, Table: true, Prefetch: true},
}

// A GolemTuneRow is one shape of one pass width, and the fastest dispatch of it.
type GolemTuneRow struct {
	Columns int
	Shape   GolemShape
	Best    time.Duration
}

// TuneGolem times every shape of every pass width against a rows-by-cols T4G
// product and answers one row each, fastest first within a width.
//
// The weights are random bytes. What a shape costs does not depend on what the
// stream says — the decode is the same arithmetic on every input — and
// TestGolemBuildsAgree is where what a shape answers is held to the processor.
func TuneGolem(d *Device, rows, cols, rounds int) ([]GolemTuneRow, error) {
	if cols%nn.T4GSeq != 0 {
		return nil, fmt.Errorf("vk: a T4G row needs a multiple of %d columns, given %d", nn.T4GSeq, cols)
	}
	data := make([]byte, rows*(nn.Matrix{Quant: nn.T4G, Cols: cols}).RowBytes())
	rand.New(rand.NewSource(23)).Read(data)
	w, err := d.UploadTail(data, golemReadTail)
	if err != nil {
		return nil, err
	}
	defer w.Close()
	table, err := d.Upload(golemTable())
	if err != nil {
		return nil, err
	}
	defer table.Close()

	widest := 0
	for _, columns := range GolemWidths {
		widest = max(widest, columns)
	}
	act, err := d.Host(cols*widest*4, bufferUsageStorage)
	if err != nil {
		return nil, err
	}
	defer act.Close()
	out, err := d.Readback(rows*widest*4, bufferUsageStorage)
	if err != nil {
		return nil, err
	}
	defer out.Close()
	r := rand.New(rand.NewSource(7))
	for i := range act.Floats() {
		act.Floats()[i] = float32(r.NormFloat64())
	}

	type timed struct {
		row    GolemTuneRow
		set    *Set
		groups uint32
	}
	var runs []*timed
	spirv, _ := golemSPIRV(nn.T4G)
	for _, columns := range GolemWidths {
		for _, shape := range GolemTuneShapes {
			pipe, err := d.NewPipelineSpec(spirv[columns], 4, uint32(unsafe.Sizeof(golemPush{})), shape.Spec())
			if err != nil {
				return nil, err
			}
			defer pipe.Close()
			set, err := pipe.NewSet([]*Buffer{w, table, act, out})
			if err != nil {
				return nil, err
			}
			defer set.Close()
			per := uint32(shape.Threads / 8)
			runs = append(runs, &timed{
				row:    GolemTuneRow{Columns: columns, Shape: shape, Best: time.Hour},
				set:    set,
				groups: (uint32(rows) + per - 1) / per,
			})
		}
	}

	// Enough dispatches in one submission that the card raises its clocks: a
	// kernel handed a few microseconds of work and then left alone runs at half
	// its frequency, and that is not the number anyone wants.
	const times = 128 // enough that the card raises its clocks
	push := golemPush{dim: uint32(rows), ffn: uint32(cols), col: 0}
	for round := 0; round <= rounds; round++ {
		order := make([]*timed, len(runs))
		for i := range runs {
			order[i] = runs[(i+round)%len(runs)]
		}
		for _, t := range order {
			start := time.Now()
			if err := t.set.DispatchTimes(t.groups, unsafe.Pointer(&push), times); err != nil {
				return nil, err
			}
			// Round zero is the card waking up, and is not counted.
			if took := time.Since(start) / times; round > 0 && took < t.row.Best {
				t.row.Best = took
			}
		}
	}
	rows2 := make([]GolemTuneRow, 0, len(runs))
	for _, t := range runs {
		rows2 = append(rows2, t.row)
	}
	sort.SliceStable(rows2, func(i, j int) bool {
		if rows2[i].Columns != rows2[j].Columns {
			return rows2[i].Columns < rows2[j].Columns
		}
		return rows2[i].Best < rows2[j].Best
	})
	return rows2, nil
}

// GolemTuneBest is the fastest shape of each width in a set of rows.
func GolemTuneBest(rows []GolemTuneRow) map[int]GolemShape {
	best := map[int]GolemTuneRow{}
	for _, r := range rows {
		if cur, ok := best[r.Columns]; !ok || r.Best < cur.Best {
			best[r.Columns] = r
		}
	}
	out := map[int]GolemShape{}
	for w, r := range best {
		out[w] = r.Shape
	}
	return out
}

// GolemShapeSetting is what to put in GOLEM_MATVEC_SHAPE for those shapes.
func GolemShapeSetting(shapes map[int]GolemShape) string {
	var widths []int
	for w := range shapes {
		widths = append(widths, w)
	}
	sort.Ints(widths)
	var b strings.Builder
	for _, w := range widths {
		if b.Len() > 0 {
			b.WriteByte(' ')
		}
		fmt.Fprintf(&b, "%d:%s", w, shapes[w])
	}
	return b.String()
}
