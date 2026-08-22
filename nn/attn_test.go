package nn

import (
	"math/rand"
	"testing"
)

// The two halves of one head's attention, against the plain loops they stand
// in for.
func TestScoresAndMixAgreeWithTheLoops(t *testing.T) {
	r := rand.New(rand.NewSource(11))
	for _, hd := range []int{64, 32, 8, 13} {
		for _, n := range []int{1, 3, 4, 7, 64, 129} {
			q := make([]float32, hd)
			k := make([]float32, n*hd)
			v := make([]float32, n*hd)
			w := make([]float32, n)
			for i := range q {
				q[i] = r.Float32()*2 - 1
			}
			for i := range k {
				k[i] = r.Float32()*2 - 1
				v[i] = r.Float32()*2 - 1
			}
			for i := range w {
				w[i] = r.Float32()
			}

			got := make([]float32, n)
			Scores(q, k, hd, n, got)
			for j := 0; j < n; j++ {
				want := DotF32(q, k[j*hd:(j+1)*hd])
				if diff := got[j] - want; diff > 1e-4 || diff < -1e-4 {
					t.Fatalf("hd=%d n=%d: score %d is %g, expected %g", hd, n, j, got[j], want)
				}
			}

			mixed := make([]float32, hd)
			Mix(mixed, v, w, hd, n)
			want := make([]float32, hd)
			for j := 0; j < n; j++ {
				Axpy(want, v[j*hd:(j+1)*hd], w[j])
			}
			for i := range want {
				if diff := mixed[i] - want[i]; diff > 1e-4 || diff < -1e-4 {
					t.Fatalf("hd=%d n=%d: element %d is %g, expected %g", hd, n, i, mixed[i], want[i])
				}
			}
		}
	}
}

// The four-at-a-time forms against the one-at-a-time ones.
func TestScores4AndMix4AgreeWithTheSingles(t *testing.T) {
	r := rand.New(rand.NewSource(23))
	for _, hd := range []int{64, 32, 16} {
		for _, n := range []int{1, 2, 5, 64, 1053} {
			q := make([]float32, 4*hd)
			k := make([]float32, n*hd)
			v := make([]float32, n*hd)
			for i := range q {
				q[i] = r.Float32()*2 - 1
			}
			for i := range k {
				k[i] = r.Float32()*2 - 1
				v[i] = r.Float32()*2 - 1
			}

			got := make([]float32, 4*n)
			if !Scores4(q, k, hd, n, got, n) {
				t.Skip("no kernel on this machine")
			}
			for i := 0; i < 4; i++ {
				want := make([]float32, n)
				Scores(q[i*hd:(i+1)*hd], k, hd, n, want)
				for j := range want {
					if diff := got[i*n+j] - want[j]; diff > 1e-4 || diff < -1e-4 {
						t.Fatalf("hd=%d n=%d: query %d score %d is %g, expected %g",
							hd, n, i, j, got[i*n+j], want[j])
					}
				}
			}

			dst := make([]float32, 4*hd)
			if !Mix4(dst, hd, v, got, n, hd, n) {
				t.Fatalf("hd=%d: Mix4 declined", hd)
			}
			for i := 0; i < 4; i++ {
				want := make([]float32, hd)
				Mix(want, v, got[i*n:(i+1)*n], hd, n)
				for j := range want {
					if diff := dst[i*hd+j] - want[j]; diff > 1e-4 || diff < -1e-4 {
						t.Fatalf("hd=%d n=%d: query %d element %d is %g, expected %g",
							hd, n, i, j, dst[i*hd+j], want[j])
					}
				}
			}
		}
	}
}
