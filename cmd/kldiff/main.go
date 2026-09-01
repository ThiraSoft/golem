package main

// kldiff: the distance between two models' opinions, position by position.
//
// KL(P‖Q) with P the original weighs every token the original thought possible;
// top-1 agreement asks the blunter question of whether the same word wins.

import (
	"encoding/binary"
	"flag"
	"fmt"
	"math"
	"os"
)

func main() {
	a := flag.String("a", "", "reference logits")
	b := flag.String("b", "", "compressed logits")
	vocab := flag.Int("vocab", 151936, "vocabulary")
	flag.Parse()

	fa, err := os.Open(*a)
	must(err)
	fb, err := os.Open(*b)
	must(err)

	// A dump is rows of the model's vocabulary and nothing else — no header, no
	// count — so the width has to be given, and a wrong one is not an error
	// anywhere downstream: the read succeeds, every row straddles two real ones,
	// and the answer comes back in the range a right answer lives in. That
	// happened. Qwen3.8's vocabulary is 248320 against the 151936 default, and
	// two files of 4088 positions were compared as 6681 of them for 0.0945 nats,
	// 83.2 % top-1 and 98.5 % top-5 — all plausible, all meaningless.
	//
	// The width divides the file or it is the wrong width. That is the whole
	// check, and it is enough: a dump whose size is not a whole number of rows
	// is being read as something it is not.
	row := int64(*vocab) * 4
	for _, f := range []*os.File{fa, fb} {
		st, err := f.Stat()
		must(err)
		if st.Size()%row != 0 {
			must(fmt.Errorf("kldiff: %s is %d bytes, which is %.2f rows of %d — pass the -vocab this dump was written with",
				f.Name(), st.Size(), float64(st.Size())/float64(row), *vocab))
		}
	}
	if sa, sb := size(fa), size(fb); sa != sb {
		must(fmt.Errorf("kldiff: %s holds %d positions and %s holds %d", *a, sa/row, *b, sb/row))
	}

	pa := make([]float32, *vocab)
	pb := make([]float32, *vocab)

	var kl, top1, top5, n float64
	for {
		if err := binary.Read(fa, binary.LittleEndian, pa); err != nil {
			break
		}
		must(binary.Read(fb, binary.LittleEndian, pb))
		la := logSoftmax(pa)
		lb := logSoftmax(pb)
		var d float64
		for i := range la {
			p := math.Exp(la[i])
			if p > 1e-12 {
				d += p * (la[i] - lb[i])
			}
		}
		kl += d
		ia, ib := argmax(pa), argmax(pb)
		if ia == ib {
			top1++
		}
		if inTop(pa, ib, 5) {
			top5++
		}
		n++
	}
	fmt.Printf("%d positions\n  mean KL(orig‖compressed) = %.4f nats\n"+
		"  top-1 agreement          = %.1f%%\n"+
		"  compressed pick in orig's top-5 = %.1f%%\n",
		int(n), kl/n, 100*top1/n, 100*top5/n)
}

func logSoftmax(l []float32) []float64 {
	out := make([]float64, len(l))
	mx := float64(l[0])
	for _, v := range l {
		if float64(v) > mx {
			mx = float64(v)
		}
	}
	var s float64
	for _, v := range l {
		s += math.Exp(float64(v) - mx)
	}
	ls := mx + math.Log(s)
	for i, v := range l {
		out[i] = float64(v) - ls
	}
	return out
}

func argmax(l []float32) int {
	bi, bv := 0, l[0]
	for i, v := range l {
		if v > bv {
			bi, bv = i, v
		}
	}
	return bi
}

func inTop(l []float32, id, k int) bool {
	better := 0
	for _, v := range l {
		if v > l[id] {
			better++
			if better >= k {
				return false
			}
		}
	}
	return true
}

func must(err error) {
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

// size is a file's length, for a caller that has already established it can be
// stat'd.
func size(f *os.File) int64 {
	st, err := f.Stat()
	must(err)
	return st.Size()
}
