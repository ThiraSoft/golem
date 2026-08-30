package compress

// Weight-compression codecs, and the geometry each one spends its bits on.
//
// Everything here is offline: it turns a float32 matrix into the matrix an
// inference kernel would reconstruct, so the loss can be measured before any
// kernel exists.

import (
	"fmt"
	"math"
	"math/bits"
	"math/rand"
	"runtime"
	"sort"
	"strings"
	"sync"
	"sync/atomic"

	"github.com/ThiraSoft/golem/nn"
)

// ---------- helpers ----------

func Fp16round(x float32) float32 {
	// round-trip through fp16 so stored scales cost what we claim they cost
	f := float64(x)
	if f == 0 {
		return 0
	}
	sign := float32(1)
	if f < 0 {
		sign, f = -1, -f
	}
	e := math.Floor(math.Log2(f))
	if e < -14 {
		e = -14
	}
	step := math.Pow(2, e-10)
	return sign * float32(math.Round(f/step)*step)
}

// StepCodeRound rounds a block's scale onto the eight-bit grid a file stores it
// on — powers of two a sixteenth apart — so that the bench pays for a scale
// what the format pays. Which grid depends on the codec: nn/t4g.go's window
// sits two octaves above nn/d4g.go's, because a lattice step is a fraction of
// its block's RMS and a trellis step is the RMS itself.
//
// The grid is 4.4 % wide, against fp16's 0.05 %, and half a bit a block cheaper
// at ScaleBlock 64. Whether that trade is free is not a question squared error
// can answer: compress/README.md records a step grid an eighth apart costing a
// whole point of perplexity for 0.03 dB. It is settled by perplexity and
// divergence, and the answer is written down beside the format — on Qwen3-0.6B
// at k=4, an fp16 scale reads 29.78 and KL 0.0658 where the step code reads
// 29.87 and 0.0702, which is a twentieth of a point for three percent of a file.
func StepCodeRound(x float32, trellis bool) float32 {
	if !(x > 0) {
		return 0
	}
	if trellis {
		return nn.T4GStep(nn.T4GStepCode(x))
	}
	c := nn.D4StepCode(x)
	if c == 0 || c == 255 {
		atomic.AddInt64(&stepClipped, 1)
	}
	return nn.D4Step(c)
}

// stepClipped counts the blocks whose scale landed on an end of the lattice's
// grid. The trellis keeps its own count, in nn, because the file's encoder
// needs it too and a converter says it out loud.
var stepClipped int64

// StepClipped is that count, and resets it.
func StepClipped() int64 { return atomic.SwapInt64(&stepClipped, 0) + nn.T4GStepClipped.Swap(0) }

func Parallel(n int, fn func(lo, hi int)) {
	// GOMAXPROCS and not NumCPU: they are the same until somebody sets the
	// first, and somebody setting it means they wanted fewer cores busy —
	// a laptop that throttles, a machine doing something else. NumCPU would
	// spawn the goroutines anyway and let the scheduler sort it out, which
	// works and says the wrong thing.
	w := runtime.GOMAXPROCS(0)
	if n < w {
		w = n
	}
	if w < 1 {
		w = 1
	}
	var wg sync.WaitGroup
	chunk := (n + w - 1) / w
	for i := 0; i < n; i += chunk {
		hi := i + chunk
		if hi > n {
			hi = n
		}
		wg.Add(1)
		go func(lo, hi int) { defer wg.Done(); fn(lo, hi) }(i, hi)
	}
	wg.Wait()
}

// ---------- Hadamard rotation ----------

// hadamard applies, in place and per group of g columns, a fixed random sign
// flip followed by a normalised fast Walsh-Hadamard transform. The transform is
// orthogonal, so at inference the same rotation is applied to the activations
// instead: y = Wx = (W Rᵀ)(R x).
func Hadamard(w []float32, rows, cols, g int, signs []float32) {
	if g <= 1 {
		return
	}
	inv := float32(1 / math.Sqrt(float64(g)))
	Parallel(rows, func(lo, hi int) {
		for r := lo; r < hi; r++ {
			row := w[r*cols : (r+1)*cols]
			for base := 0; base+g <= cols; base += g {
				blk := row[base : base+g]
				for i := range blk {
					blk[i] *= signs[base+i]
				}
				for l := 1; l < g; l <<= 1 {
					for i := 0; i < g; i += l << 1 {
						for j := i; j < i+l; j++ {
							a, b := blk[j], blk[j+l]
							blk[j], blk[j+l] = a+b, a-b
						}
					}
				}
				for i := range blk {
					blk[i] *= inv
				}
			}
		}
	})
}

func RandomSigns(n int, seed int64) []float32 {
	rg := rand.New(rand.NewSource(seed))
	s := make([]float32, n)
	for i := range s {
		if rg.Intn(2) == 0 {
			s[i] = -1
		} else {
			s[i] = 1
		}
	}
	return s
}

// ---------- scalar baselines ----------

// q40 is llama.cpp's Q4_0: one fp16 scale and 32 nibbles, 4.5 bits per weight.
func Q40(w []float32, rows, cols int) []float32 {
	out := make([]float32, len(w))
	Parallel(rows, func(lo, hi int) {
		for r := lo; r < hi; r++ {
			for b := r * cols; b < (r+1)*cols; b += 32 {
				var amax float32
				var mx float32
				for i := b; i < b+32; i++ {
					if v := float32(math.Abs(float64(w[i]))); v > amax {
						amax, mx = v, w[i]
					}
				}
				d := Fp16round(mx / -8)
				id := float32(0)
				if d != 0 {
					id = 1 / d
				}
				for i := b; i < b+32; i++ {
					q := math.Round(float64(w[i]*id)) + 8
					q = math.Max(0, math.Min(15, q))
					out[i] = (float32(q) - 8) * d
				}
			}
		}
	})
	return out
}

// qNsym is a symmetric n-bit scalar quantiser with one fp16 scale per block: a
// stand-in for "just use fewer bits, scalar".
func QNsym(w []float32, rows, cols, nbits, block int) []float32 {
	out := make([]float32, len(w))
	lim := float32(int(1)<<(nbits-1)) - 1
	Parallel(rows, func(lo, hi int) {
		for r := lo; r < hi; r++ {
			for b := r * cols; b < (r+1)*cols; b += block {
				var amax float32
				for i := b; i < b+block; i++ {
					if v := float32(math.Abs(float64(w[i]))); v > amax {
						amax = v
					}
				}
				d := Fp16round(amax / lim)
				id := float32(0)
				if d != 0 {
					id = 1 / d
				}
				for i := b; i < b+block; i++ {
					q := math.Round(float64(w[i] * id))
					q = math.Max(float64(-lim-1), math.Min(float64(lim), q))
					out[i] = float32(q) * d
				}
			}
		}
	})
	return out
}

// ---------- vector quantisation ----------

type Opts struct {
	Dim        int   // subvector dimension
	Stages     []int // codebook size per residual stage
	ScaleBlock int   // weights sharing one fp16 scale; 0 means one scale per row
	HadGroup   int   // 0 disables the rotation
	TrainMax   int   // subvectors sampled to fit the codebooks
	Iters      int

	// A lattice replaces the trained codebooks entirely: no search, no
	// dictionary, and a rate set by the shell radius rather than by k.
	UseLattice bool
	Lat        Lattice
	UseBox     bool // D4 confined to a box, so a decoder needs no table
	MaxNorm2   float32
	Beta       float64 // how far the normalised subvector is scaled up before rounding

	// A trellis replaces the lattice: the sequence, not the subvector, is the
	// unit that gets coded, and the state's value is computed rather than
	// stored. UseTrellis wins over UseLattice when both are set.
	UseTrellis bool
	Tr         TrellisOpts

	// Step8 stores each block's scale as an eight-bit step code rather than an
	// fp16. It is what the file does; the bench does not have to, so the two
	// can be measured against each other.
	Step8 bool

	// SearchScale trades encoding time for accuracy: instead of taking the
	// block's RMS as its scale, it tries a few multiples of it and keeps the
	// one the lattice actually reconstructs best. The RMS is the scale that
	// makes the block unit-variance, not the scale that minimises the error.
	SearchScale bool
}

// scaleBits is what one stored scale costs.
func (o Opts) scaleBits() float64 {
	if o.Step8 {
		return 8
	}
	return 16
}

// roundScale is how a stored scale is rounded: onto the eight-bit grid when the
// file will store it there, through fp16 otherwise.
func (o Opts) roundScale(x float32) float32 {
	if o.Step8 {
		return StepCodeRound(x, o.UseTrellis)
	}
	return Fp16round(x)
}

func (o Opts) BPW() float64 {
	b := 0.0
	if o.UseTrellis {
		b = o.Tr.BPW()
		if o.ScaleBlock > 0 {
			b += o.scaleBits() / float64(o.ScaleBlock)
		}
		return b
	}
	if o.UseLattice {
		n := shellSize(o.Lat, o.MaxNorm2)
		if o.UseBox {
			n = boxSize()
		}
		b = math.Log2(float64(n)) / float64(o.Lat.Dim())
		if o.ScaleBlock > 0 {
			b += o.scaleBits() / float64(o.ScaleBlock)
		}
		return b
	}
	for _, k := range o.Stages {
		b += math.Log2(float64(k)) / float64(o.Dim)
	}
	if o.ScaleBlock > 0 {
		b += o.scaleBits() / float64(o.ScaleBlock)
	}
	return b
}

func (o Opts) Name() string {
	var sb strings.Builder
	if o.HadGroup > 0 {
		fmt.Fprintf(&sb, "HAD%d+", o.HadGroup)
	}
	if o.UseTrellis {
		fmt.Fprintf(&sb, "TCQ(k%d,L%d,T%d,g%.2f)", o.Tr.K, o.Tr.L, o.Tr.Seq, o.Tr.gain())
		if o.ScaleBlock > 0 {
			fmt.Fprintf(&sb, "/s%d", o.ScaleBlock)
			if o.Step8 {
				sb.WriteString("e8")
			}
		} else {
			sb.WriteString("/srow")
		}
		return sb.String()
	}
	if o.UseLattice {
		if o.UseBox {
			fmt.Fprintf(&sb, "BOX(b%.2f)", o.Beta)
		} else {
			fmt.Fprintf(&sb, "%s(r%g,b%.2f)", o.Lat, o.MaxNorm2, o.Beta)
		}
		if o.ScaleBlock > 0 {
			fmt.Fprintf(&sb, "/s%d", o.ScaleBlock)
		} else {
			sb.WriteString("/srow")
		}
		return sb.String()
	}
	sb.WriteString("VQ")
	for i, k := range o.Stages {
		if i > 0 {
			sb.WriteString("+")
		}
		fmt.Fprintf(&sb, "%d", k)
	}
	fmt.Fprintf(&sb, "d%d", o.Dim)
	if o.ScaleBlock > 0 {
		fmt.Fprintf(&sb, "/s%d", o.ScaleBlock)
	} else {
		sb.WriteString("/srow")
	}
	return sb.String()
}

// kmeans fits k centroids of dimension d to the sampled points by Lloyd's
// algorithm, reseeding any cluster that empties onto the worst-fit point.
func Kmeans(pts []float32, n, d, k, iters int, seed int64) []float32 {
	rg := rand.New(rand.NewSource(seed))
	cent := make([]float32, k*d)
	perm := rg.Perm(n)
	for i := 0; i < k; i++ {
		copy(cent[i*d:(i+1)*d], pts[perm[i%n]*d:(perm[i%n]+1)*d])
	}
	assign := make([]int32, n)
	dist := make([]float32, n)
	for it := 0; it < iters; it++ {
		cn := make([]float32, k) // ||c||^2
		for i := 0; i < k; i++ {
			var s float32
			for j := 0; j < d; j++ {
				s += cent[i*d+j] * cent[i*d+j]
			}
			cn[i] = s
		}
		Parallel(n, func(lo, hi int) {
			for p := lo; p < hi; p++ {
				x := pts[p*d : (p+1)*d]
				best, bestv := int32(0), float32(math.MaxFloat32)
				for i := 0; i < k; i++ {
					c := cent[i*d : (i+1)*d]
					v := cn[i]
					for j := 0; j < d; j++ {
						v -= 2 * x[j] * c[j]
					}
					if v < bestv {
						best, bestv = int32(i), v
					}
				}
				assign[p], dist[p] = best, bestv
			}
		})
		if it == iters-1 {
			break
		}
		sum := make([]float64, k*d)
		cnt := make([]int, k)
		for p := 0; p < n; p++ {
			a := int(assign[p])
			cnt[a]++
			for j := 0; j < d; j++ {
				sum[a*d+j] += float64(pts[p*d+j])
			}
		}
		type pd struct {
			i int
			v float32
		}
		var worst []pd
		for i := 0; i < k; i++ {
			if cnt[i] == 0 {
				worst = append(worst, pd{i, 0})
			}
		}
		if len(worst) > 0 {
			order := make([]int, n)
			for i := range order {
				order[i] = i
			}
			sort.Slice(order, func(a, b int) bool { return dist[order[a]] > dist[order[b]] })
			for wi := range worst {
				src := order[wi%n]
				copy(cent[worst[wi].i*d:(worst[wi].i+1)*d], pts[src*d:(src+1)*d])
				cnt[worst[wi].i] = -1
			}
		}
		for i := 0; i < k; i++ {
			if cnt[i] > 0 {
				for j := 0; j < d; j++ {
					cent[i*d+j] = float32(sum[i*d+j] / float64(cnt[i]))
				}
			}
		}
	}
	return cent
}

// encode replaces each subvector by its nearest centroid, accumulating the
// residual so later stages can correct it.
func Encode(pts []float32, n, d int, cent []float32, k int) {
	cn := make([]float32, k)
	for i := 0; i < k; i++ {
		var s float32
		for j := 0; j < d; j++ {
			s += cent[i*d+j] * cent[i*d+j]
		}
		cn[i] = s
	}
	Parallel(n, func(lo, hi int) {
		for p := lo; p < hi; p++ {
			x := pts[p*d : (p+1)*d]
			best, bestv := 0, float32(math.MaxFloat32)
			for i := 0; i < k; i++ {
				c := cent[i*d : (i+1)*d]
				v := cn[i]
				for j := 0; j < d; j++ {
					v -= 2 * x[j] * c[j]
				}
				if v < bestv {
					best, bestv = i, v
				}
			}
			c := cent[best*d : (best+1)*d]
			for j := 0; j < d; j++ {
				x[j] -= c[j]
			}
		}
	})
}

// vq runs the whole scheme and returns the reconstruction.
func VQ(w []float32, rows, cols int, o Opts, seed int64) []float32 {
	src := make([]float32, len(w))
	copy(src, w)
	if o.HadGroup > 0 {
		if bits.OnesCount(uint(o.HadGroup)) != 1 || cols%o.HadGroup != 0 {
			return nil
		}
		Hadamard(src, rows, cols, o.HadGroup, RandomSigns(cols, seed+7))
	}

	// per-block RMS scales
	sb := o.ScaleBlock
	if sb == 0 {
		sb = cols
	}
	scales := make([]float32, rows*cols/sb)
	norm := make([]float32, len(src))
	Parallel(rows, func(lo, hi int) {
		for r := lo; r < hi; r++ {
			for b := 0; b < cols; b += sb {
				var s float64
				for i := r*cols + b; i < r*cols+b+sb; i++ {
					s += float64(src[i]) * float64(src[i])
				}
				sc := float32(math.Sqrt(s / float64(sb)))
				// The lattice stores this number, so it is rounded to what
				// the file can hold. A trellis does not: its step is fitted
				// to the path by least squares below and rounded there, and
				// this one only normalises the block on the way in. Rounding
				// it here would move the path — by four percent on the
				// eight-bit grid, which is a gain error the codebook was
				// measured not to want — and buy nothing at all.
				if !o.UseTrellis {
					sc = o.roundScale(sc)
				}
				scales[(r*cols+b)/sb] = sc
				inv := float32(0)
				if sc != 0 {
					inv = 1 / sc
				}
				for i := r*cols + b; i < r*cols+b+sb; i++ {
					norm[i] = src[i] * inv
				}
			}
		}
	})

	// The reconstruction both vector quantizers share: put the scales back,
	// then undo the rotation the way the activation side would.
	rebuild := func() []float32 {
		out := make([]float32, len(w))
		Parallel(rows, func(lo, hi int) {
			for r := lo; r < hi; r++ {
				for b := 0; b < cols; b += sb {
					sc := scales[(r*cols+b)/sb]
					for i := r*cols + b; i < r*cols+b+sb; i++ {
						out[i] = norm[i] * sc
					}
				}
			}
		})
		if o.HadGroup > 0 {
			signs := RandomSigns(cols, seed+7)
			Hadamard(out, rows, cols, o.HadGroup, Make1(cols))
			Parallel(rows, func(lo, hi int) {
				for r := lo; r < hi; r++ {
					for i := 0; i < cols; i++ {
						out[r*cols+i] *= signs[i]
					}
				}
			})
		}
		return out
	}

	// A trellis codes whole sequences, so it wants the row length to hold a
	// whole number of them; every matrix in these models does.
	if o.UseTrellis {
		if o.Tr.Seq <= 0 || len(norm)%o.Tr.Seq != 0 || o.Tr.L <= o.Tr.K || o.Tr.K > 8 {
			return nil
		}
		// The step is chosen *after* the path, not before it. The block's RMS
		// is the scale that makes it unit-variance, which is not the scale that
		// reconstructs it best; the lattice buys that difference with a grid of
		// seven multipliers per block, and a trellis cannot, because one path
		// spans many blocks. Least squares gets it exactly and for nothing: the
		// step code the format already stores per 32 weights absorbs it.
		src := append([]float32(nil), norm...)
		quantizeTrellis(norm, o.Tr, TrellisTable(o.Tr.Code, o.Tr.L), nil)
		for b := 0; b < len(norm)/sb; b++ {
			var num, den float64
			for i := b * sb; i < (b+1)*sb; i++ {
				num += float64(src[i]) * float64(norm[i])
				den += float64(norm[i]) * float64(norm[i])
			}
			if den > 0 {
				scales[b] = o.roundScale(scales[b] * float32(num/den))
			}
		}
		return rebuild()
	}

	if o.UseLattice {
		d := o.Lat.Dim()
		beta := float32(o.Beta)
		mults := []float32{1}
		if o.SearchScale {
			mults = []float32{0.82, 0.88, 0.94, 1, 1.06, 1.13, 1.22}
		}
		nblk := rows * cols / sb
		Parallel(nblk, func(lo, hi int) {
			pt := make([]float32, d)
			tmp := make([]float32, d)
			buf := make([]float32, d)
			best := make([]float32, sb)
			cand := make([]float32, sb)
			for b := lo; b < hi; b++ {
				blk := norm[b*sb : (b+1)*sb]
				bestErr := float32(math.MaxFloat32)
				for _, mu := range mults {
					copy(cand, blk)
					var err float32
					for p := 0; p*d < sb; p++ {
						x := cand[p*d : (p+1)*d]
						for i := range x {
							buf[i] = x[i] * beta / mu
						}
						if o.UseBox {
							quantizeBox(buf, pt)
						} else {
							quantizeLattice(o.Lat, buf, pt, tmp, o.MaxNorm2)
						}
						for i := range x {
							q := buf[i] * mu / beta
							e := x[i] - q
							err += e * e
							x[i] = q
						}
					}
					if err < bestErr {
						bestErr = err
						copy(best, cand)
					}
				}
				copy(blk, best)
			}
		})
		return rebuild()
	}

	d := o.Dim
	n := len(norm) / d
	resid := make([]float32, len(norm))
	copy(resid, norm)

	rg := rand.New(rand.NewSource(seed))
	for si, k := range o.Stages {
		m := o.TrainMax
		if m > n {
			m = n
		}
		sample := make([]float32, m*d)
		for i := 0; i < m; i++ {
			p := rg.Intn(n)
			copy(sample[i*d:(i+1)*d], resid[p*d:(p+1)*d])
		}
		cent := Kmeans(sample, m, d, k, o.Iters, seed+int64(si))
		Encode(resid, n, d, cent, k)
	}

	// reconstruction = normalised original minus what the stages failed to code
	out := make([]float32, len(w))
	Parallel(rows, func(lo, hi int) {
		for r := lo; r < hi; r++ {
			for b := 0; b < cols; b += sb {
				sc := scales[(r*cols+b)/sb]
				for i := r*cols + b; i < r*cols+b+sb; i++ {
					out[i] = (norm[i] - resid[i]) * sc
				}
			}
		}
	})
	if o.HadGroup > 0 {
		// the rotation is its own inverse, up to the sign flips applied first
		signs := RandomSigns(cols, seed+7)
		Hadamard(out, rows, cols, o.HadGroup, Make1(cols))
		Parallel(rows, func(lo, hi int) {
			for r := lo; r < hi; r++ {
				for i := 0; i < cols; i++ {
					out[r*cols+i] *= signs[i]
				}
			}
		})
	}
	return out
}

func Make1(n int) []float32 {
	s := make([]float32, n)
	for i := range s {
		s[i] = 1
	}
	return s
}
