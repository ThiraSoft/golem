// golemtune finds the shape of the trellis mat-vec this card is fastest at.
//
// The .golem kernel has three knobs that trade against each other with the
// width of a pass — the workgroup, whether the codebook's image is built in
// shared memory or hashed again for every weight, and whether a block's stream
// is read one iteration before it is decoded — and which way they trade is a
// property of the card. This times every combination of them against a product
// the size of a real one, interleaved so that the card's clock drifting between
// cases cannot decide the answer, and prints what to set GOLEM_MATVEC_SHAPE to.
//
//	golemtune                 # the whole table, and the setting at the end
//	golemtune -quiet          # the setting alone, for a shell to export
//	golemtune -rows 4096 -cols 5120 -rounds 9
//
// Every shape answers the same numbers — vk's TestGolemBuildsAgree holds them
// to that — so this is only ever a question of speed, and a setting that turns
// out to be wrong for a machine costs nothing but the speed it was chosen for.
package main

import (
	"flag"
	"fmt"
	"os"

	"github.com/ThiraSoft/golem/vk"
)

func main() {
	// The default shape is one feed-forward matrix of Qwen3-4B, which is the
	// shape this repository's own numbers were taken on.
	rows := flag.Int("rows", 9728, "outputs of the product to time")
	cols := flag.Int("cols", 2560, "inputs of it; a multiple of 128")
	rounds := flag.Int("rounds", 5, "timed passes over every shape; the fastest of each is kept")
	quiet := flag.Bool("quiet", false, "print the setting alone, without the table")
	flag.Parse()

	if err := run(*rows, *cols, *rounds, *quiet); err != nil {
		fmt.Fprintln(os.Stderr, "golemtune:", err)
		os.Exit(1)
	}
}

func run(rows, cols, rounds int, quiet bool) error {
	d, err := vk.Open()
	if err != nil {
		return err
	}
	defer d.Close()

	timings, err := vk.TuneGolem(d, rows, cols, rounds)
	if err != nil {
		return err
	}
	best := vk.GolemTuneBest(timings)
	if !quiet {
		fmt.Printf("%d by %d, the fastest of %d rounds, microseconds a dispatch\n\n", rows, cols, rounds)
		fmt.Printf("%8s %8s %7s %9s %10s %12s\n", "columns", "threads", "table", "prefetch", "us", "us/column")
		width := 0
		for _, t := range timings {
			if t.Columns != width {
				width = t.Columns
			}
			mark := " "
			if best[t.Columns] == t.Shape {
				mark = "*"
			}
			us := float64(t.Best.Nanoseconds()) / 1000
			fmt.Printf("%s%7d %8d %7t %9t %10.2f %12.2f\n",
				mark, t.Columns, t.Shape.Threads, t.Shape.Table, t.Shape.Prefetch, us, us/float64(t.Columns))
		}
		fmt.Println()
	}
	fmt.Printf("GOLEM_MATVEC_SHAPE=%q\n", vk.GolemShapeSetting(best))
	return nil
}
