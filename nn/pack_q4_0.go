package nn

// The packed Q4_0 product, in the parts that do not depend on how a group is
// laid out. The layout itself, and the two functions that know it, live in
// pack_q4_0_layout_x86.go and pack_q4_0_layout_arm64.go.

// maxGroupTile is the tile of the outer loop, in groups of rows. Its
// shape is the one matVecQ4_0Rows uses, for the same reason: a stretch of the
// columns' activations and the slice of weights the tile covers both have to
// stay in the first-level cache while the loop runs.
const maxGroupTile = 8

// matVecPackedQ4_0Rows computes rows [start, end) from the packed weights, on
// the caller's thread. A group whose rows are not all wanted is computed whole
// and written in part: the row range comes from however the section was split,
// and it costs less to recompute a group than to make every caller split on
// them.
func matVecPackedQ4_0Rows(w []byte, b *Batch, inputs int, ys [][]float32, start, end int) {
	groupBytes := inputs / QuantBlock * packedBlockBytes
	first, last := start/PackedRows, (end+PackedRows-1)/PackedRows

	steps := min(kTile, inputs)
	stepBytes := steps / QuantBlock * packedBlockBytes
	tile := min(max(rowTileBytes/stepBytes, 1), maxGroupTile)

	var four [maxGroupTile][4 * PackedRows]float32
	var one [maxGroupTile][PackedRows]float32

	write := func(g, column int, values []float32) {
		row := g * PackedRows
		for r := 0; r < PackedRows; r++ {
			if row+r >= start && row+r < end {
				ys[column][row+r] = values[r]
			}
		}
	}

	for from := first; from < last; from += tile {
		to := min(from+tile, last)
		c := 0
		for ; c+4 <= b.Size; c += 4 {
			for k := 0; k < inputs; k += steps {
				block, n := k/QuantBlock, min(steps, inputs-k)
				mode := Mode(0)
				if k == 0 {
					mode |= Begin
				}
				if k+n >= inputs {
					mode |= Finish
				}
				for g := from; g < to; g++ {
					dotPackedQ4_0x4(w[g*groupBytes+block*packedBlockBytes:], b, block, c, n, four[g-from][:], mode)
				}
			}
			for g := from; g < to; g++ {
				for j := 0; j < 4; j++ {
					write(g, c+j, four[g-from][j*PackedRows:])
				}
			}
		}
		for ; c < b.Size; c++ {
			for k := 0; k < inputs; k += steps {
				block, n := k/QuantBlock, min(steps, inputs-k)
				mode := Mode(0)
				if k == 0 {
					mode |= Begin
				}
				if k+n >= inputs {
					mode |= Finish
				}
				for g := from; g < to; g++ {
					dotPackedQ4_0(w[g*groupBytes+block*packedBlockBytes:], b, block, c, n, one[g-from][:], mode)
				}
			}
			for g := from; g < to; g++ {
				write(g, c, one[g-from][:])
			}
		}
	}
}
