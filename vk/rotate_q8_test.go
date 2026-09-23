package vk

import (
	"math/rand"
	"testing"
	"unsafe"

	"github.com/ThiraSoft/golem/nn"
)

func TestRotateQ8MatchesCPU(t *testing.T) {
	d := open(t)
	defer d.Close()

	testCases := []struct {
		name   string
		n      int
		gather bool
	}{
		{name: "plain-6144", n: 6144, gather: false},
		{name: "gather-6144", n: 6144, gather: true},
		{name: "plain-17408", n: 17408, gather: false},
	}

	const columns = 3

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			r := rand.New(rand.NewSource(int64(tc.n)))
			signs := make([]float32, tc.n)
			for i := range signs {
				if r.Intn(2) == 0 {
					signs[i] = 1.0
				} else {
					signs[i] = -1.0
				}
			}

			var gather []int32
			if tc.gather {
				gather = make([]int32, tc.n)
				// Create permutation: reverse blocks of 128 or random shuffle
				for i := range gather {
					gather[i] = int32(i)
				}
				r.Shuffle(tc.n, func(i, j int) {
					gather[i], gather[j] = gather[j], gather[i]
				})
			}

			x := make([]float32, tc.n*columns)
			for i := range x {
				x[i] = float32(r.NormFloat64())
			}

			// CPU computation
			batch := nn.NewBatch(tc.n, columns)
			for c := 0; c < columns; c++ {
				col := make([]float32, tc.n)
				if tc.gather {
					for i := 0; i < tc.n; i++ {
						col[i] = x[c*tc.n+int(gather[i])]
					}
				} else {
					copy(col, x[c*tc.n:(c+1)*tc.n])
				}
				nn.PrepareGolem(col, signs, RotateQ8Group)
				copy(batch.F[c], col)
			}
			batch.Quantize()

			// GPU computation
			src, err := d.Upload(asBytes(x))
			if err != nil {
				t.Fatal(err)
			}
			defer src.Close()

			aq, err := d.Host(tc.n*columns, bufferUsageStorage)
			if err != nil {
				t.Fatal(err)
			}
			defer aq.Close()

			nb := tc.n / nn.QuantBlock
			as, err := d.Host(2*nb*columns*4, bufferUsageStorage)
			if err != nil {
				t.Fatal(err)
			}
			defer as.Close()

			var rot *RotateQ8
			if tc.gather {
				rot, err = NewRotateQ8Gather(d, src, aq, as, signs, gather)
			} else {
				rot, err = NewRotateQ8(d, src, aq, as, signs)
			}
			if err != nil {
				t.Fatal(err)
			}
			defer rot.Close()

			if err := rot.Run(columns); err != nil {
				t.Fatal(err)
			}

			gotQ := unsafe.Slice((*int8)(unsafe.Pointer(&aq.Bytes()[0])), tc.n*columns)
			gotS := as.Floats()

			diffCount := 0
			scaleMismatches := 0
			corrMismatches := 0

			for c := 0; c < columns; c++ {
				for b := 0; b < nb; b++ {
					idx := b*batch.Stride + c
					wantQ := batch.Q[idx*nn.QuantBlock : (idx+1)*nn.QuantBlock]
					wantScale := batch.Scales[idx]
					wantCorr := batch.Corr[idx]

					gotBlockQ := gotQ[c*tc.n+b*32 : c*tc.n+(b+1)*32]
					gotScale := gotS[c*2*nb+b]
					gotCorr := gotS[c*2*nb+nb+b]

					blockHadDiff := false
					for i := 0; i < 32; i++ {
						dVal := int(gotBlockQ[i]) - int(wantQ[i])
						if dVal != 0 {
							diffCount++
							blockHadDiff = true
							if dVal < -1 || dVal > 1 {
								t.Fatalf("col %d block %d elem %d: got %d, want %d (diff > 1)", c, b, i, gotBlockQ[i], wantQ[i])
							}
						}
					}

					if !blockHadDiff {
						if gotScale != wantScale {
							scaleMismatches++
							t.Errorf("col %d block %d: scale got %v, want %v", c, b, gotScale, wantScale)
						}
						if gotCorr != wantCorr {
							corrMismatches++
							t.Errorf("col %d block %d: corr got %v, want %v", c, b, gotCorr, wantCorr)
						}
					}
				}
			}

			t.Logf("%s: total elements=%d, +/-1 magnitude diffs=%d, scale mismatches=%d, corr mismatches=%d",
				tc.name, tc.n*columns, diffCount, scaleMismatches, corrMismatches)

			// Tolerate at most a handful of +/-1 differences from float summation order
			if diffCount > 50 {
				t.Errorf("too many magnitude differences: %d", diffCount)
			}
		})
	}
}
