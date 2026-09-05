package vk

import (
	"math/rand"
	"testing"

	"github.com/ThiraSoft/golem/nn"
)

// A K-quant row with an odd number of superblocks, through the tiled product's
// own door.
//
// The packed layouts round a row up to a whole word — two hundred and ten bytes
// a superblock is not one, and neither is a hundred and fourteen — so a matrix
// with an odd superblock count holds two more bytes a row on the card than the
// file gives it. NewMatMulQuant length-checks the caller's data, which is the
// file's, and a check written against the *packed* count rejects a tensor that
// is exactly right.
//
// 3840 columns is fifteen superblocks and it is not a hypothetical width: it is
// Gemma 4 12B's attention. The formats whose row is already a whole number of
// words — Q4_0, Q4_K — cannot show this, which is why it survived.
func TestTiledProductTakesOddSuperblockRows(t *testing.T) {
	d := open(t)
	defer d.Close()
	if !d.Coopmat() {
		t.Skip("no cooperative matrices on this device")
	}

	const rows, cols, columns = 64, 3840, 64
	nsb := cols / nn.SuperBlock
	if nsb%2 == 0 {
		t.Fatalf("%d columns is %d superblocks, an even count, and this test needs an odd one", cols, nsb)
	}
	for _, f := range []struct {
		q     nn.Quant
		bytes int // what one row takes in the file
	}{
		{nn.Q6_K, nsb * 210},
		{nn.Q3_K, nsb * 110},
	} {
		t.Run(f.q.String(), func(t *testing.T) {
			rng := rand.New(rand.NewSource(int64(f.q)))
			data := make([]byte, rows*f.bytes)
			rng.Read(data)
			m := nn.Matrix{Data: data, Quant: f.q, Rows: rows, Cols: cols}
			sane(t, m, rng)

			mm, err := NewMatMulQuant(d, data, rows, cols, columns, true, f.q)
			if err != nil {
				t.Fatalf("a %d by %d %s tensor of %d bytes was refused: %v", rows, cols, f.q, len(data), err)
			}
			mm.Close()
		})
	}
}
