package main

// vqeval: how much of a weight matrix's *output* a codec loses.
//
// Frobenius error on the weights is the wrong scoreboard: a coefficient that
// meets a large activation matters more than one that meets noise. This runs a
// calibration text through the model, keeps the activations that feed each
// matrix, and scores every scheme by ||(W-Ŵ)X|| / ||WX||.

import (
	"flag"
	"fmt"
	"math"
	"os"
	"strings"
	"time"

	"github.com/ThiraSoft/golem/compress"
	"github.com/ThiraSoft/golem/qwen"
	"github.com/ThiraSoft/golem/tensors"
	"github.com/ThiraSoft/golem/token/bytebpe"
)

// site names the matrices each tap feeds.
var sites = map[string][]string{
	"qkv":    {"attn_q", "attn_k", "attn_v"},
	"o":      {"attn_output"},
	"gateup": {"ffn_gate", "ffn_up"},
	"down":   {"ffn_down"},
}

func main() {
	model := flag.String("model", "", "BF16 GGUF")
	ntok := flag.Int("tokens", 128, "calibration tokens")
	blocks := flag.String("blocks", "0,17,35", "blocks to score")
	only := flag.String("only", "attn_k,ffn_down", "matrices to score")
	flag.Parse()

	m, err := qwen.Open(*model, 4096)
	must(err)
	defer m.Close()
	v, err := bytebpe.Load(m.File())
	must(err)

	ids := v.Encode(calibText, true, false)
	if len(ids) > *ntok {
		ids = ids[:*ntok]
	}

	want := map[int]bool{}
	for _, b := range strings.Split(*blocks, ",") {
		var n int
		fmt.Sscan(b, &n)
		want[n] = true
	}

	// Collect the activations that feed the matrices we care about.
	acts := map[string][][]float32{}
	qwen.Calib = func(block int, site string, rows [][]float32) {
		if !want[block] {
			return
		}
		key := fmt.Sprintf("%d/%s", block, site)
		if _, seen := acts[key]; seen {
			return
		}
		cp := make([][]float32, len(rows))
		for i, r := range rows {
			cp[i] = append([]float32(nil), r...)
		}
		acts[key] = cp
	}
	t0 := time.Now()
	m.ForwardBatch(ids, 0)
	qwen.Calib = nil
	fmt.Printf("%d tokens through the model in %s; %d activation sets kept\n\n",
		len(ids), time.Since(t0).Round(time.Millisecond), len(acts))

	g := m.File()
	schemes := buildSchemes()
	wanted := strings.Split(*only, ",")

	for blk := range want {
		for site, mats := range sites {
			X := acts[fmt.Sprintf("%d/%s", blk, site)]
			if X == nil {
				continue
			}
			sal := salience(X)
			for _, mat := range mats {
				if !contains(wanted, mat) {
					continue
				}
				name := fmt.Sprintf("blk.%d.%s.weight", blk, mat)
				t, err := g.Get(name)
				if err != nil {
					continue
				}
				w, err := t.F32()
				must(err)
				cols := t.Shape[0]
				rows := t.Elems() / cols
				base := outputs(w, X, rows, cols)
				fmt.Printf("== %s  %dx%d\n", name, rows, cols)
				fmt.Printf("   %-34s %6s  %8s  %8s\n", "scheme", "bpw", "out.err", "wt.err")
				report := func(label string, bpw float64, rec []float32) {
					fmt.Printf("   %-34s %6.2f  %8.5f  %8.5f\n",
						label, bpw, outErr(base, rec, X, rows, cols), relErr(w, rec))
				}
				report("Q4_0 (llama.cpp baseline)", 4.5, compress.Q40(w, rows, cols))
				for _, s := range schemes {
					rec := s.Apply(w, rows, cols, sal, 1234)
					if rec == nil {
						continue
					}
					report(s.Name(), s.BPW(cols), rec)
				}
				fmt.Println()
			}
		}
	}
}

func buildSchemes() []compress.Scheme {
	lat := func(l compress.Lattice, r float32, beta float64) compress.Scheme {
		return compress.Scheme{Alpha: 0.5, Outliers: 32, VQ: compress.Opts{
			UseLattice: true, Lat: l, MaxNorm2: r, Beta: beta,
			ScaleBlock: 64, HadGroup: 128, SearchScale: true}}
	}
	box := func(beta float64) compress.Scheme {
		return compress.Scheme{Alpha: 0.5, Outliers: 32, VQ: compress.Opts{
			UseLattice: true, Lat: compress.LatD4, UseBox: true, Beta: beta,
			ScaleBlock: 64, HadGroup: 128, SearchScale: true}}
	}
	var out []compress.Scheme
	out = append(out, lat(compress.LatD4, 40, 8))   // the shell, 12 bits, 31 KiB table
	out = append(out, lat(compress.LatD4, 20, 5.7)) // a shell of 11 bits, to match the box
	for _, b := range []float64{1.5, 2.2, 3.0, 4.0, 5.5} {
		out = append(out, box(b))
	}
	return out
}

// salience is the per-column RMS of the activations feeding a matrix.
func salience(X [][]float32) []float32 {
	cols := len(X[0])
	s := make([]float64, cols)
	for _, r := range X {
		for j, v := range r {
			s[j] += float64(v) * float64(v)
		}
	}
	out := make([]float32, cols)
	for j := range out {
		out[j] = float32(math.Sqrt(s[j] / float64(len(X))))
	}
	return out
}

// outputs computes W xᵀ for every calibration activation.
func outputs(w []float32, X [][]float32, rows, cols int) [][]float32 {
	out := make([][]float32, len(X))
	for i := range out {
		out[i] = make([]float32, rows)
	}
	compress.Parallel(rows, func(lo, hi int) {
		for r := lo; r < hi; r++ {
			row := w[r*cols : (r+1)*cols]
			for t, x := range X {
				var s float32
				for j, xv := range x {
					s += row[j] * xv
				}
				out[t][r] = s
			}
		}
	})
	return out
}

func outErr(base [][]float32, rec []float32, X [][]float32, rows, cols int) float64 {
	got := outputs(rec, X, rows, cols)
	var num, den float64
	for t := range base {
		for r := range base[t] {
			d := float64(base[t][r] - got[t][r])
			num += d * d
			den += float64(base[t][r]) * float64(base[t][r])
		}
	}
	return math.Sqrt(num / den)
}

func relErr(a, b []float32) float64 {
	var num, den float64
	for i := range a {
		d := float64(a[i] - b[i])
		num += d * d
		den += float64(a[i]) * float64(a[i])
	}
	return math.Sqrt(num / den)
}

func contains(s []string, v string) bool {
	for _, x := range s {
		if x == v {
			return true
		}
	}
	return false
}

func must(err error) {
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

const calibText = `The quick brown fox jumps over the lazy dog. ` +
	`In computing, quantization is the process of constraining values from a ` +
	`continuous set to a relatively small discrete set. Neural network weights ` +
	`are commonly stored as 16-bit floating point numbers, and inference is ` +
	`limited by memory bandwidth rather than arithmetic throughput, so reducing ` +
	`the number of bits per weight makes generation faster as well as smaller. ` +
	`La compression des poids d'un modèle de langage repose sur l'idée que la ` +
	`distribution des coefficients est fortement redondante.`

var _ = tensors.Tensor{}
