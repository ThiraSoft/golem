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
