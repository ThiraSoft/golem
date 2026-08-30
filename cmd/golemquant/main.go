package main

// golemquant: a checkpoint in, a .golem out.
//
// The file is a GGUF — the same container, so the vocabulary, the rope base and
// the chat template travel unchanged — carrying tensors of a type llama.cpp
// does not know. What is new besides the weights is one vector a block: the
// per-column scale the activations of each site must meet, with the rotation's
// sign flips folded into it.

import (
	"bufio"
	"encoding/binary"
	"flag"
	"fmt"
	"io"
	"math"
	"os"
	"sort"
	"strings"
	"time"

	"github.com/ThiraSoft/golem/compress"
	"github.com/ThiraSoft/golem/nn"
	"github.com/ThiraSoft/golem/qwen"
	"github.com/ThiraSoft/golem/qwen35"
	"github.com/ThiraSoft/golem/tensors"
	"github.com/ThiraSoft/golem/token/bytebpe"
	"github.com/ThiraSoft/golem/vk"
)

// headKey is the calibration site of the tied output head, which belongs to no
// block and so is filed under -1.
const headKey = "-1/head"

func main() {
	src := flag.String("model", "", "the BF16 checkpoint to convert")
	dst := flag.String("out", "", "the .golem file to write")
	alpha := flag.Float64("alpha", 0.5, "salience exponent; 0 leaves the columns alone")
	clamp := flag.Float64("clamp", 24, "largest factor the salience may scale a column by, either way; 0 lets it run")
	hadGroup := flag.Int("hadamard", 128, "rotation group; 0 leaves the weights unrotated")
	beta := flag.Float64("beta", 2, "how far a block is scaled up before rounding")
	codebook := flag.String("codebook", "d4", "d4 for the lattice in a table, lloyd for eight levels in registers")
	headBits := flag.Int("head", 4, "trellis: bits a weight for the logit head, 4 or 5. Five is what llama.cpp's K-quant mixes do in spirit — Qwen3-4B's Q4_K_M spends 6.56 bits there and 4.95 on the rest — and on Qwen3-0.6B it takes about three fifths of what an unquantized head is worth, for six percent of the file rather than seventy-three. Four is the default because the smallest file is the point")
	codec := flag.String("codec", "lattice", "lattice or trellis; the trellis has no decode table at all, and reaches four bits where the lattice's shell stops fitting a workgroup")
	codeBits := flag.Int("bits", 12, "code width: 12 for the ordinary tier, 16 for the wide one")
	scaleBlk := flag.Int("scale", 32, "weights sharing one step code; 32 is what the format stores")
	ntok := flag.Int("tokens", 8192, "calibration tokens")
	ctx := flag.Int("ctx", 512, "calibration window")
	calibFile := flag.String("calib", "", "text to calibrate on; a built-in paragraph when empty")
	embd := flag.String("embd", "rot", "how to store token_embd: rot, plain or bf16")
	search := flag.Bool("search", false, "choose the salience of each site by the output error it leaves")
	sample := flag.Int("sample", 128, "rows a site's salience is chosen on")
	swin := flag.Int("swin", 256, "how many columns the search's Hessian keeps together")
	report := flag.Bool("report", false, "print what each matrix's codes cost it")
	blind := flag.Bool("blind", true, "rotate matrices no calibration site names, with signs alone")
	keep := flag.String("keep", "", "comma-separated tensor name fragments left in BF16")
	window := flag.Int("gptq", 0, "error-compensation window in columns; 0 turns it off")
	damp := flag.Float64("damp", 0.01, "ridge on the Hessian diagonal, as a fraction of its mean")
	vulkan := flag.Bool("vulkan", true, "encode the matrices on a Vulkan device when there is one")
	calibSrc := flag.String("calib-model", "", "the checkpoint to calibrate on, when it is not the one being converted")
	salFile := flag.String("salience", "", "read the sites from this file, or write them to it after measuring; the salience does not depend on -alpha, -clamp or the codec, and measuring it again for each of them is most of a sweep's wall clock")
	flag.Parse()

	trellis := *codec == "trellis"
	if !trellis && *codec != "lattice" {
		must(fmt.Errorf("golemquant: %q is not a codec", *codec))
	}
	if trellis && *headBits != nn.T4GK && *headBits != nn.T5GK {
		must(fmt.Errorf("golemquant: the head is %d or %d bits, not %d", nn.T4GK, nn.T5GK, *headBits))
	}
	if trellis {
		// The step is one per sixty-four weights and the format says so; the
		// flag is the lattice's and there is nothing here to choose.
		*scaleBlk = nn.T4GBlock
	}

	text := calibText
	if *calibFile != "" {
		b, err := os.ReadFile(*calibFile)
		must(err)
		text = string(b)
	}
	win := *window
	if *search && win < *swin {
		win = *swin
	}
	// What a site is fed does not depend on which of llama.cpp's four-bit
	// forms the weights that fed it were stored in, so a calibration may read
	// a different build of the same model from the one being converted — which
	// is what -calib-model is for, and what lets the pass run on a card when
	// the converter's input is a type no kernel here reads.
	calibFrom := *src
	if *calibSrc != "" {
		calibFrom = *calibSrc
	}
	// The salience is a property of the model and the corpus alone: what the
	// activations put through each column. Everything that turns it into a
	// scale — the exponent, the bound — happens below, and a sweep over those
	// re-measures nothing. On Qwen3-4B the measurement is twelve minutes and
	// the conversion is ten, so a sweep of four settings goes from ninety
	// minutes to fifty.
	var salience map[string][]float32
	var accs map[string]*compress.Acc
	if *salFile != "" && win == 0 {
		if s, err := readSalience(*salFile); err == nil {
			fmt.Printf("%d sites read from %s\n", len(s), *salFile)
			salience = s
		}
	}
	if salience == nil {
		salience, accs = calibrate(calibFrom, text, *ntok, *ctx, win, *scaleBlk, *vulkan)
		if *salFile != "" && win == 0 && len(salience) > 0 {
			if err := writeSalience(*salFile, salience); err != nil {
				fmt.Printf("the sites were not kept (%v)\n", err)
			} else {
				fmt.Printf("%d sites written to %s\n", len(salience), *salFile)
			}
		}
	}
	if len(salience) == 0 && !*blind {
		must(fmt.Errorf("golemquant: nothing calibrated and -blind is off, so nothing would be rotated"))
	}

	g, err := tensors.OpenGGUF(*src)
	must(err)
	defer g.Close()

	// The two halves of a conversion are not alike. The calibration runs the
	// model, so it needs the engine and stays on the processor. The encoding
	// needs nothing but the matrix — eight hundred million scale blocks, each
	// searching its own step over forty candidates, none of them looking at
	// another — and on this machine that was sixty-eight minutes of eight
	// cores against under two of a card. The kernel writes the same bytes:
	// vk.TestEncodeD4GMatchesCPU holds it to this encoder byte for byte,
	// because a file quantized half one way and half the other would be two
	// formats sharing a name.
	var gpu *vk.D4GEncoder
	if *vulkan {
		if d, err := vk.Open(); err != nil {
			fmt.Printf("no device (%v); the matrices are encoded on the processor\n", err)
		} else {
			defer d.Close()
			if trellis {
				// The Viterbi is 2^L operations a weight against the lattice's
				// eight — five hundred times the work, and the only half of a
				// conversion that cares where it runs. Sixteen million weights
				// a pass is a quarter of a gigabyte of buffers.
				if enc, err := vk.NewTrellisEncoder(d, 1<<24); err != nil {
					fmt.Printf("no trellis encoder on the device (%v); the processor then\n", err)
				} else {
					defer enc.Close()
					var onCard, offCard int64
					compress.TrellisPathAccel = func(norm []float32, o compress.TrellisOpts, states []uint16) bool {
						// The kernels are compiled for two rates and one
						// shape. Anything else falls back rather than quietly
						// answering a different question.
						if !vk.TrellisGPUHasK(o.K) || o.L != vk.TrellisGPUL || o.Seq != vk.TrellisGPUSeq {
							offCard += int64(len(norm))
							return false
						}
						if err := enc.UseK(o.K); err != nil {
							offCard += int64(len(norm))
							return false
						}
						if err := enc.QuantizePath(norm, float32(o.Gain), states); err != nil {
							fmt.Printf("  the card refused a matrix (%v); the processor takes it\n", err)
							offCard += int64(len(norm))
							return false
						}
						onCard += int64(len(norm))
						return true
					}
					defer func() {
						fmt.Printf("%d M weights through the card, %d M through the processor\n",
							onCard/1e6, offCard/1e6)
					}()
				}
			} else if gpu, err = vk.NewD4GEncoder(d, *codeBits, 32<<20); err != nil {
				// Nine and a quarter bytes a weight across the six buffers, so
				// thirty-two million of them is three hundred megabytes and the
				// widest row in any checkpoint fits a pass many times over.
				fmt.Printf("no encoder on the device (%v); the matrices are encoded on the processor\n", err)
			} else if gpu != nil {
				defer gpu.Close()
			}
		}
	}

	names := make([]string, 0, len(g.Tensors))
	for n := range g.Tensors {
		names = append(names, n)
	}
	sort.Strings(names)

	params := compress.D4Params{Beta: *beta, ScaleBlock: *scaleBlk,
		HadGroup: *hadGroup, Bits: *codeBits, SearchScale: true}
	dtype := "D4G"
	if *codeBits == nn.D4Bits16 {
		dtype = "D4G16"
	}
	// A trellis settles all of this: the sequence, the rate and the state width
	// are what a workgroup's shared memory holds, the step is one per
	// sixty-four weights, and the codebook has no parameter at all. So the
	// lattice's flags are simply not read.
	if trellis {
		dtype = "T4G"
		params = compress.D4Params{ScaleBlock: nn.T4GBlock, HadGroup: *hadGroup}
	}
	lloyd := !trellis && *codebook == "lloyd"
	// What a row has to be a multiple of.
	unit := nn.D4Block
	if trellis {
		unit = nn.T4GSeq
	}
	if lloyd {
		dtype = "L8G"
		if !flagWasSet("beta") {
			// The levels are a unit Gaussian's, so a block's step is its RMS
			// rather than a fraction of it. The lattice wants the block scaled
			// up into its shell; this wants it left where it is.
			params.Beta = 1
		}
	} else if !trellis && *codebook != "d4" {
		must(fmt.Errorf("golemquant: %q is not a codebook", *codebook))
	}

	// One vector a site: the sign flips of the rotation over the salience
	// scale. The weights are multiplied by it, the activations by its
	// reciprocal, and nn.PrepareD4G is both.
	signs := map[int][]float32{}
	pre := map[string][]float32{}    // what the activations meet
	weight := map[string][]float32{} // its reciprocal, what the weights meet
	build := func(key string, alpha, clamp float64) ([]float32, []float32) {
		sal := salience[key]
		cols := len(sal)
		if signs[cols] == nil {
			signs[cols] = compress.RandomSigns(cols, int64(cols)*7919)
		}
		sc := saliencyScale(sal, alpha, clamp)
		p := make([]float32, cols)
		q := make([]float32, cols)
		for j := range sc {
			q[j] = signs[cols][j] * sc[j]
			p[j] = 1 / q[j]
		}
		return p, q
	}
	for key := range salience {
		pre[key], weight[key] = build(key, *alpha, *clamp)
	}

	// A site's salience is a guess at how to move error away from the columns
	// the activations use most, and a guess can be checked. With -search each
	// site tries a few exponents and bounds on a sample of one of its matrices
	// and keeps the one that leaves the product closest — measured against the
	// activations themselves, because weight error cannot see the trade the
	// scaling is making.
	//
	// It is off, because it does not work. Over six settings of the sample and
	// the window — 32, 128 and 512 rows against Hessians 256 and 1024 columns
	// wide — Qwen3-0.6B reads 39.20, 40.23, 39.64, 39.75, 39.94 and 39.64, a
	// mean of 39.73 either side of the 39.80 that one bound chosen for the
	// whole model gives. The spread is the choosing, not the choice: a metric
	// that ranks candidates by a windowed Hessian over a sample of rows is
	// noisier than the differences between the candidates, so the search picks
	// a different winner each time and lands where it started. The first run
	// read 39.20 and it would have been easy to keep only that one.
	//
	// What survives is the machinery — Acc.Energy and EnergyD4G measure what a
	// matrix costs the product rather than what it costs the weights, and that
	// is the right question whatever asks it next.
	type cand struct{ alpha, clamp float64 }
	// A narrow grid on purpose. Widening it to eighteen candidates reaching an
	// exponent of 1 and bounds of 4 and 96 makes the answer worse — 40.42
	// against 39.20 on Qwen3-0.6B, and 41.30 with three times the sample, so
	// it is not the sample. The windowed Hessian cannot see what an aggressive
	// scaling moves beyond its own window, so it ranks the extremes too well
	// and the search believes it. The grid is kept to the range the metric can
	// be trusted over.
	cands := []cand{{0.5, 0}, {0.35, 24}, {0.5, 12}, {0.5, 24}, {0.5, 48}, {0.65, 12}, {0.65, 24}, {0.8, 8}}
	searched := map[string]bool{}
	pickSalience := func(key string, w []float32, rows, cols int) {
		if !*search || searched[key] || accs[key] == nil {
			return
		}
		searched[key] = true
		n := *sample
		if n > rows {
			n = rows
		}
		p := params
		best, bestAt := math.Inf(1), cand{*alpha, *clamp}
		for _, c := range cands {
			pv, qv := build(key, c.alpha, c.clamp)
			data := compress.EncodeD4G(w[:n*cols], n, cols, qv, p, nil)
			num, den := compress.EnergyD4G(w[:n*cols], n, cols, qv, pv, p, data, accs[key])
			if e := num / den; e < best {
				best, bestAt = e, c
			}
		}
		pre[key], weight[key] = build(key, bestAt.alpha, bestAt.clamp)
		fmt.Printf("  salience %-10s alpha %.2f bound %4.0f, output error %.4f\n",
			key, bestAt.alpha, bestAt.clamp, math.Sqrt(best))
	}

	// The second pass. The Hessian of a site is what says how to spend the
	// columns not yet quantized on the error of the ones already are, and it
	// has to be taken in the basis the weights were rotated into — so it can
	// only be built once the vectors above exist, which is why this is a pass
	// of its own and not a tally kept during the first.
	comps := map[string]*compress.Comp{}
	if *window > 0 {
		comps = hessians(calibFrom, text, *ntok, *ctx, pre, *hadGroup, *embd,
			*window, *scaleBlk, *damp)
	}

	// What a matrix with no calibration site was rotated by, one vector each,
	// written beside it.
	blindPre := map[string][]float32{}
	signsFor := func(cols int, name string) ([]float32, []float32) {
		// The seed is the name, so that a converter run twice writes the same
		// file and two tensors of the same width do not share a rotation.
		var h int64 = 1469598103934665603
		for _, c := range []byte(name) {
			h = (h ^ int64(c)) * 1099511628211
		}
		if h < 0 {
			h = -h
		}
		q := compress.RandomSigns(cols, h)
		p := make([]float32, cols)
		for j, v := range q {
			p[j] = 1 / v
		}
		return q, p
	}

	// plan is what a tensor will be, decided before any of it is encoded.
	type plan struct {
		name     string
		t        tensors.Tensor
		passthru bool
		dtype    string
		cols     int
		rows     int // rows of one matrix; a stack has that many each
		stack    int
		key      string // the site this matrix is filed under, or nothing
		params   compress.D4Params
		size     int
	}

	// encodeInto writes one matrix, a run of rows at a time.
	//
	// A run and not the whole of it, because expanding a matrix to floats is
	// four bytes a weight: a vocabulary table of 248320 rows by 5120 is five
	// gigabytes of them, and there are two such tensors in a model with an
	// untied head. The rows of a matrix are independent — the rotation is a
	// row's own business and so is its step search — so a chunk is the same
	// bytes as the whole, produced in a bounded amount of memory.
	encodeInto := func(w io.Writer, pl plan) error {
		// The vector this matrix is quantized against, taken from the very map
		// that is written out — pre for a site, blindPre for a matrix that has
		// none. There is no second copy to drift from it.
		//
		// It is read here rather than settled when the tensor was planned
		// because -search may still replace a measured site's vector, and the
		// file gets whatever the map holds at the end.
		var av []float32
		if pl.key != "" {
			av = pre[pl.key]
		} else {
			av = blindPre[pl.name]
		}
		q := reciprocal(av)
		comp := comps[pl.key]
		kind, _ := nn.QuantOf(pl.dtype)
		stride := rowBytes(pl.cols, pl.dtype)
		perChunk := max(1, expandBudget/(pl.cols*4))
		var relerr float64
		for e := 0; e < pl.stack; e++ {
			for at := 0; at < pl.rows; at += perChunk {
				n := min(perChunk, pl.rows-at)
				rows, err := expandRows(pl.t, e*pl.rows+at, e*pl.rows+at+n)
				if err != nil {
					return err
				}
				if e == 0 && at == 0 {
					pickSalience(pl.key, rows, n, pl.cols)
					if pl.key != "" {
						q = reciprocal(pre[pl.key])
					}
				}
				var data []byte
				switch {
				case trellis:
					data = compress.EncodeT4GAs(rows, n, pl.cols, q, pl.params, kind)
				case lloyd:
					data = compress.EncodeL8G(rows, n, pl.cols, q, pl.params)
				case onDevice(gpu, pl.params, comp, q):
					data = encodeOnDevice(gpu, rows, n, pl.cols, q, pl.params)
				default:
					data = compress.EncodeD4G(rows, n, pl.cols, q, pl.params, comp)
				}
				if e == 0 && at == 0 {
					// What the codes cost in the basis they were written in.
					// The theoretical floor for a memoryless Gaussian at this
					// rate is about seventeen decibels, so this says how much
					// of the gap is the quantizer's own and how much is
					// everything else.
					//
					// It is measured on every matrix and not only under
					// -report, because it is also the only thing that notices
					// a matrix quantized against a vector that is not the one
					// written beside it. That mistake changes no shape and no
					// name — the file loads, every tensor is the size it
					// should be — and it takes this number from 0.15 to 1.4,
					// which is a matrix with nothing left of the one it stands
					// for. Twice now: a hybrid's linear-attention projections,
					// and the token table of a model whose engine taps no head
					// site.
					//
					// It is measured against the vector the FILE holds, not
					// the one the encoder happened to hold, and that is the
					// whole of its value. A guard that asks the encoder to
					// check its own arithmetic cannot see the mistake this
					// format keeps making, which is a matrix quantized against
					// one vector and read back through another: both halves
					// agree with themselves and disagree with each other.
					//
					// A sample of rows, because the whole of a 248320-row
					// table would cost more than the encoding did, and the
					// error is the same in every row.
					sample := min(n, 64)
					relerr = compress.RelErr(rows[:sample*pl.cols], sample, pl.cols, q, pl.params, data[:sample*stride], kind)
					if relerr > relErrCeiling {
						return fmt.Errorf("golemquant: %s came back at %.4f of its own size, so it was not quantized against the vector written beside it",
							pl.name, relerr)
					}
				}
				if _, err := w.Write(data); err != nil {
					return err
				}
			}
		}
		note := ""
		if *report {
			note = fmt.Sprintf("  rel %.4f  %.2f dB", relerr, -20*math.Log10(relerr))
		}
		what := fmt.Sprintf("%6dx%-6d", pl.rows, pl.cols)
		if pl.stack > 1 {
			what = fmt.Sprintf("%3dx%5dx%-6d", pl.stack, pl.rows, pl.cols)
		}
		fmt.Printf("  %-32s %s %s%s\n", pl.name, what, sizeOf(pl.size), note)
		return nil
	}

	// A conversion is planned before it is written. Every tensor's name, shape,
	// type and size are known before a single code is chosen — and they have
	// to be, because a GGUF's table carries offsets into the data and the
	// table is written first.
	//
	// The alternative is what this did: encode all of it, then write it. Ten
	// gigabytes of output held beside the sixteen gigabyte checkpoint it was
	// read from, on a machine with thirty-one. The kernel ended that run.
	// The vector each encoded matrix was rotated by, under the name it is
	// written as. checkVectors reads it back the way the loader will.
	rotatedBy := map[string]string{}
	// A site nothing was measured at still has one vector, shared by every
	// matrix that reads it, written under the site's own name. sitePre is what
	// the activations meet — the same thing pre holds for a measured site —
	// and blindSite is the half of it that has to be written out.
	sitePre := map[string][]float32{}
	blindSite := map[string][]float32{}
	plans := make([]plan, 0, len(names))
	var bits, count float64
	for _, name := range names {
		t := g.Tensors[name]
		// A tensor this format has nothing to say about, one a probe is
		// leaving alone, or the table of a probe that wants it in bf16: all
		// three travel unchanged.
		if !encodable(t, unit) || kept(name, *keep) || (name == "token_embd.weight" && *embd == "bf16") {
			plans = append(plans, plan{name: name, t: t, passthru: true,
				dtype: t.DType, size: len(t.Raw)})
			bits += float64(len(t.Raw)) * 8
			count += float64(t.Elems())
			continue
		}
		pl := plan{name: name, t: t, dtype: dtype, cols: t.Shape[0], stack: 1, params: params}
		// The logit head, which is the table when the two are tied. It is not
		// a hidden layer whose error the layers after it absorb; it is the
		// thing that makes the logits, and a bit a weight over a tenth of the
		// model is a tenth of a bit over the file.
		//
		// On Qwen3-0.6B, where the head is a quarter of the weights: four bits
		// reads 30.82 and KL 0.0812, five reads 30.52 and 0.0718, and bf16 —
		// the ceiling, at seventy-three percent more file — reads 30.30 and
		// 0.0662. Five bits takes three fifths of the way there for six
		// percent of the file.
		if trellis && *headBits == nn.T5GK &&
			(name == "output.weight" || (name == "token_embd.weight" && *embd != "bf16")) {
			pl.dtype = "T5G"
		}
		if len(t.Shape) == 3 {
			// A stack of experts: ne2 of them, each ne1 rows of ne0. They are
			// one tensor in the file and one matrix each here, because a
			// lattice code spans four weights of a row and a row belongs to
			// one expert.
			pl.stack = t.Shape[2]
		}
		pl.rows = t.Elems() / pl.cols / pl.stack

		var q []float32
		if blk, mat, ok := parse(name); ok {
			// Which site's activations reach this matrix, asked of nn, which
			// is also where the reader asks what to file the answer under. A
			// copy of the table here is how a hybrid's four input projections
			// came to be rotated blind while the reader bound them to the
			// site vector the three of a full attention share — a file that
			// loads and answers nonsense.
			if site, known := nn.D4GSite(mat); known {
				pl.key = fmt.Sprintf("%d/%s", blk, site)
				q = weight[pl.key]
			}
		} else if name == "output.weight" || (name == "token_embd.weight" && *embd == "rot") {
			pl.key = headKey
			// The table is also the logit head, and the head is a site like any
			// other: it has activations, so it has a salience and a rotation.
			// What the input path pays for that is one transform of the model's
			// width per token, which is nothing beside reading the row.
			q = weight[headKey]
		}
		if q == nil && *blind && pl.cols%*hadGroup == 0 {
			// No statistics for this matrix. It still gets the rotation,
			// because incoherence is most of what the rotation is for and it
			// needs none: signs alone, and a vector written beside it so the
			// reader can undo them.
			//
			// Where that vector goes is the whole of what went wrong twice.
			// The reader asks nn.D4GVectorNames, which offers a site's name
			// before the tensor's own — so a matrix that HAS a site must be
			// rotated by the site's vector whether or not anything was
			// measured there, and the matrices of one site must all get the
			// same one. A prediction block lies past the trunk a corpus walks,
			// so nothing measures it; its attention was rotated with three
			// different vectors and read back with one, and it drafted a token
			// the model agreed with zero times in twenty-four.
			if pl.key != "" {
				at := siteTensor(pl.key, *embd)
				if sitePre[pl.key] == nil {
					_, blindSite[pl.key] = signsFor(pl.cols, at)
					sitePre[pl.key] = blindSite[pl.key]
				} else if len(sitePre[pl.key]) != pl.cols {
					must(fmt.Errorf("golemquant: %s reads %d columns at a site whose vector is %d wide",
						name, pl.cols, len(sitePre[pl.key])))
				}
			} else {
				_, blindPre[name] = signsFor(pl.cols, name)
			}
		} else if q == nil {
			// Neither a site nor a rotation: the matrix is quantized as it
			// stands, and the reader must find no vector for it at all.
			pl.key = ""
		}
		switch {
		case pl.key != "":
			rotatedBy[name] = siteTensor(pl.key, *embd)
		case blindPre[name] != nil:
			rotatedBy[name] = name + ".pre"
		default:
			pl.params.HadGroup = 0
			rotatedBy[name] = ""
		}
		pl.size = pl.stack * pl.rows * rowBytes(pl.cols, pl.dtype)
		plans = append(plans, pl)
		bits += float64(pl.size) * 8
		count += float64(t.Elems())
	}

	var out []tensors.OutStream
	for i := range plans {
		pl := plans[i]
		if pl.passthru {
			raw := pl.t.Raw
			out = append(out, tensors.OutStream{Name: pl.name, Shape: pl.t.Shape,
				DType: pl.dtype, Size: len(raw),
				Write: func(w io.Writer) error { _, err := w.Write(raw); return err }})
			continue
		}
		out = append(out, tensors.OutStream{Name: pl.name, Shape: pl.t.Shape,
			DType: pl.dtype, Size: pl.size,
			Write: func(w io.Writer) error { return encodeInto(w, pl) }})
	}

	// A vector for every matrix that had no site, under its own name.
	blindNames := make([]string, 0, len(blindPre))
	for k := range blindPre {
		blindNames = append(blindNames, k)
	}
	sort.Strings(blindNames)
	for _, k := range blindNames {
		raw := f32Bytes(blindPre[k])
		out = append(out, tensors.OutStream{Name: k + ".pre", Shape: []int{len(blindPre[k])},
			DType: "F32", Size: len(raw),
			Write: func(w io.Writer) error { _, err := w.Write(raw); return err }})
		bits += float64(len(raw)) * 8
	}

	// The vectors, one a site, as plain F32 tensors the loader binds by name.
	preNames := make([]string, 0, len(pre))
	for k := range pre {
		preNames = append(preNames, k)
	}
	for k, v := range blindSite {
		if pre[k] != nil {
			must(fmt.Errorf("golemquant: site %s has both a measured vector and a blind one", k))
		}
		pre[k] = v
	}
	preNames = preNames[:0]
	for k := range pre {
		preNames = append(preNames, k)
	}
	sort.Strings(preNames)
	for _, k := range preNames {
		name := siteTensor(k, *embd)
		if name == "" {
			continue
		}
		raw := f32Bytes(pre[k])
		out = append(out, tensors.OutStream{Name: name, Shape: []int{len(pre[k])},
			DType: "F32", Size: len(raw),
			Write: func(w io.Writer) error { _, err := w.Write(raw); return err }})
		bits += float64(len(raw)) * 8
	}

	t0 := time.Now()
	meta := map[string]any{}
	for k, v := range g.Meta {
		meta[k] = v
	}
	meta["golem.d4.hadamard_group"] = uint32(*hadGroup)
	meta["golem.d4.radius"] = uint32(nn.D4Radius)
	meta["golem.d4.scale_block"] = uint32(*scaleBlk)
	meta["golem.d4.code_bits"] = uint32(*codeBits)
	meta["general.file_type"] = uint32(1000)
	if trellis {
		meta["general.file_type"] = uint32(1003)
		meta["golem.trellis.seq"] = uint32(nn.T4GSeq)
		meta["golem.trellis.bits"] = uint32(nn.T4GK)
		meta["golem.trellis.state"] = uint32(nn.T4GL)
	}

	must(checkVectors(out, rotatedBy))

	must(tensors.WriteGGUFStream(*dst, meta, out))
	if n := nn.T4GStepClipped.Load(); n > 0 {
		fmt.Printf("%d blocks of the %.0f M weights landed on an end of the step grid\n", n, count/1e6)
	}
	fmt.Printf("\n%.0f M weights at %.3f bits each — %s\n",
		count/1e6, bits/count, sizeOf(int(bits/8)))
	fmt.Printf("written to %s in %s\n", *dst, time.Since(t0).Round(time.Second))
}

// onDevice says whether the card's encoder answers this matrix. It writes the
// same bytes as the sweep for the format as the converter's defaults produce
// it, and nothing else: the compensation is a sequential pass over the columns
// of a row and belongs on a processor, and the kernel is built for the scale
// block and the rotation the format stores.
func onDevice(gpu *vk.D4GEncoder, p compress.D4Params, comp *compress.Comp, q []float32) bool {
	if gpu == nil || comp != nil {
		return false
	}
	if p.ScaleBlock != nn.D4SubBlock {
		return false
	}
	return q == nil || p.HadGroup == 0 || p.HadGroup == vk.PrepareD4GGroup
}

// encodeOnDevice is EncodeD4G with the search on the card: the codes and the
// steps come back wide, and the packing into the file's twelve-bit planes is
// compress.PackD4G, which is the same call the sweep makes at the end of a row.
func encodeOnDevice(gpu *vk.D4GEncoder, w []float32, rows, cols int, q []float32, p compress.D4Params) []byte {
	codes := make([]uint16, rows*cols/4)
	steps := make([]byte, rows*cols/nn.D4SubBlock)
	lo, hi := compress.D4SearchSpan(p)
	must(gpu.Encode(w, rows, cols, q, vk.D4GEncodeParams{
		HadGroup: p.HadGroup, Beta: float32(p.Beta), SpanLo: lo, SpanHi: hi,
	}, codes, steps))
	return compress.PackD4G(codes, steps, rows, cols, p.Width())
}

// relErrCeiling is what a matrix may differ from its original by and still be
// this format doing its job. The quantizer reads 0.15 on every tensor of every
// model it has been pointed at, ±0.005; a matrix given the wrong vector reads
// 1.4, which is √2 — an encoding with nothing in common with what it stands
// for. Nothing lands between, so the line can sit anywhere between them.
const relErrCeiling = 0.5

// rowBytes is what one row of a matrix takes in the format named, so that a
// sample of rows can be cut out of the bytes without decoding them.
func rowBytes(cols int, dtype string) int {
	kind, _ := nn.QuantOf(dtype)
	return nn.Matrix{Quant: kind, Cols: cols}.RowBytes()
}

// siteTensor names the F32 tensor a site's vector is written as.
//
// The head's site is written whenever anything is filed under it. A model with
// an untied head files its output matrix there whatever -embd says; -embd only
// decides whether the table joins it, because a table stored plain is not
// rotated at all.
func siteTensor(key, embd string) string {
	if key == headKey {
		return "output.pre"
	}
	parts := strings.SplitN(key, "/", 2)
	return fmt.Sprintf("blk.%s.%s.pre", parts[0], parts[1])
}

// checkVectors reads the file back the way the loader will and insists that
// every matrix finds the vector it was rotated by.
//
// The loader takes the first name nn.D4GVectorNames offers that the file has, a
// site's before a tensor's own. So a matrix quantized blind, under its own
// name, is silently given the site's vector instead whenever that site exists —
// which is what happened to a hybrid's four input projections when this
// converter kept its own table of sites and did not know theirs. The file
// loaded, every shape agreed, and the model answered nonsense. Nothing here
// notices that by arithmetic, so it is asked outright.
func checkVectors(out []tensors.OutStream, rotatedBy map[string]string) error {
	have := map[string]int{}
	for _, t := range out {
		if t.DType == "F32" && strings.HasSuffix(t.Name, ".pre") {
			have[t.Name] = t.Shape[0]
		}
	}
	for _, t := range out {
		want, encoded := rotatedBy[t.Name]
		if !encoded {
			continue
		}
		got := ""
		for _, at := range nn.D4GVectorNames(t.Name) {
			// The loader also refuses a vector that is not the width of the
			// row, so this asks the same question it does.
			if n, ok := have[at]; ok && n == t.Shape[0] {
				got = at
				break
			}
		}
		if got != want {
			return fmt.Errorf("golemquant: %s was rotated by %q and the loader would read %q",
				t.Name, want, got)
		}
	}
	return nil
}

// runCalib walks a text through the model in windows, with the tap set. The
// cache is forgotten between windows: each is its own context, which is what
// makes one long text into many independent samples.
func runCalib(path, text string, ntok, ctx int, hook func(int, string, [][]float32)) int {
	// Whichever engine claims the checkpoint. The salience is worth twenty
	// points of perplexity on Qwen3-0.6B — 39.80 against 60.01 with the
	// columns left alone — so a conversion that cannot calibrate is not a
	// conversion worth doing, and it says so rather than quietly writing one.
	var run func(func(int, string, [][]float32)) (int, error)
	if m, err := qwen.Open(path, ctx); err == nil {
		run = func(h func(int, string, [][]float32)) (int, error) {
			defer m.Close()
			qwen.Calib = h
			defer func() { qwen.Calib = nil }()
			return sweep(m.File(), text, ntok, ctx, m.Reset, m.ForwardBatch)
		}
	} else if m2, err2 := qwen35.Open(path, ctx); err2 == nil {
		run = func(h func(int, string, [][]float32)) (int, error) {
			defer m2.Close()
			qwen35.Calib = h
			defer func() { qwen35.Calib = nil }()
			return sweep(m2.File(), text, ntok, ctx, m2.Reset, m2.ForwardBatch)
		}
	} else {
		fmt.Printf("no calibration: %v; %v\n  every matrix will be rotated with signs alone, and none scaled —\n  which is worth twenty points of perplexity, so do not ship this\n", err, err2)
		return 0
	}
	n, err := run(hook)
	must(err)
	return n
}

// sweep walks the text through a model in windows, forgetting the cache between
// them so each is its own context and one long text becomes many samples.
func sweep(g *tensors.GGUF, text string, ntok, ctx int,
	reset func(), forward func([]int32, int) [][]float32) (int, error) {
	v, err := bytebpe.Load(g)
	if err != nil {
		return 0, err
	}
	ids := v.Encode(text, true, false)
	if len(ids) > ntok {
		ids = ids[:ntok]
	}
	n := 0
	for start := 0; start+ctx <= len(ids); start += ctx {
		reset()
		forward(ids[start:start+ctx], 0)
		n += ctx
	}
	return n, nil
}

// calibrate keeps, for each site, the per-column power of the activations that
// reach it. That is all the salience scaling needs, and it is what the second
// pass has to know before it can build anything.
// calibrateVulkan is the same measurement with the model on a card: the
// accumulators live beside the activations and the host never sees a row.
//
// It answers only the salience, which is the per-column power of a site. A
// compensation pass wants the whole Hessian of a site, and that is rows —
// there is no summary of them — so -gptq keeps to the processor.
func calibrateVulkan(path, text string, ntok, ctx int) (map[string][]float32, bool) {
	m, err := qwen35.Open(path, ctx)
	if err != nil {
		return nil, false
	}
	defer m.Close()
	if err := m.UseVulkanStack(); err != nil {
		fmt.Printf("no calibration on the card (%v); the processor then\n", err)
		return nil, false
	}
	if err := m.StartVulkanCalibration(); err != nil {
		fmt.Printf("no calibration on the card (%v); the processor then\n", err)
		return nil, false
	}
	t0 := time.Now()
	n, err := sweep(m.File(), text, ntok, ctx, m.Reset, func(ids []int32, at int) [][]float32 {
		out := m.ForwardBatch(ids, at)
		m.CountVulkanCalibration(len(ids))
		return out
	})
	must(err)
	sums, rows, err := m.VulkanCalibrationSums()
	must(err)
	if rows == 0 {
		return nil, false
	}
	out := make(map[string][]float32, len(sums))
	for k, v := range sums {
		u := make([]float32, len(v))
		for j, x := range v {
			u[j] = float32(math.Sqrt(float64(x) / float64(rows)))
		}
		out[k] = u
	}
	fmt.Printf("calibrated on %d tokens in %s on the card, %d sites\n",
		n, time.Since(t0).Round(time.Millisecond), len(out))
	return out, true
}

func calibrate(path, text string, ntok, ctx, win, block int, useVulkan bool) (map[string][]float32, map[string]*compress.Acc) {
	if useVulkan && win == 0 {
		if sal, ok := calibrateVulkan(path, text, ntok, ctx); ok {
			return sal, nil
		}
	}
	sums := map[string][]float64{}
	seen := map[string]int{}
	accs := map[string]*compress.Acc{}
	t0 := time.Now()
	n := runCalib(path, text, ntok, ctx, func(blk int, site string, rows [][]float32) {
		key := fmt.Sprintf("%d/%s", blk, site)
		s := sums[key]
		if s == nil {
			s = make([]float64, len(rows[0]))
			sums[key] = s
		}
		for _, r := range rows {
			for j, x := range r {
				s[j] += float64(x) * float64(x)
			}
		}
		seen[key] += len(rows)
		if win > 0 {
			// The Hessian in the basis the activations arrive in, which is the
			// one an error has to be brought back to before it is weighed.
			a := accs[key]
			if a == nil {
				a = compress.NewAcc(len(rows[0]), fitWindow(len(rows[0]), win, block))
				accs[key] = a
			}
			a.AddRows(rows)
		}
	})
	out := map[string][]float32{}
	for k, s := range sums {
		v := make([]float32, len(s))
		for j := range s {
			v[j] = float32(math.Sqrt(s[j] / float64(seen[k])))
		}
		out[k] = v
	}
	fmt.Printf("calibrated on %d tokens in %s, %d sites\n",
		n, time.Since(t0).Round(time.Millisecond), len(out))
	return out, accs
}

// hessians is the second pass: the same text again, with each activation put
// through the site's own vector and rotation before it is counted, so that what
// comes out is the Hessian the quantizer will actually meet.
func hessians(path, text string, ntok, ctx int, pre map[string][]float32,
	group int, embd string, want, block int, damp float64) map[string]*compress.Comp {
	accs := map[string]*compress.Acc{}
	buf := [][]float32{}
	t0 := time.Now()
	runCalib(path, text, ntok, ctx, func(blk int, site string, rows [][]float32) {
		key := fmt.Sprintf("%d/%s", blk, site)
		cols := len(rows[0])
		a := accs[key]
		if a == nil {
			a = compress.NewAcc(cols, fitWindow(cols, want, block))
			accs[key] = a
		}
		for len(buf) < len(rows) {
			buf = append(buf, make([]float32, cols))
		}
		use := buf[:len(rows)]
		for i, r := range rows {
			if len(use[i]) != cols {
				use[i] = make([]float32, cols)
			}
			copy(use[i], r)
			nn.PrepareD4G(use[i], pre[key], hadamardOf(key, group, embd))
		}
		a.AddRows(use)
	})
	out := map[string]*compress.Comp{}
	for k, a := range accs {
		out[k] = a.Comp(damp)
	}
	fmt.Printf("factored %d sites in %s\n", len(out), time.Since(t0).Round(time.Second))
	return out
}

// fitWindow shrinks a requested window until it divides the row and holds a
// whole number of scale blocks: a ragged window would either straddle a block
// or leave one uncompensated, and neither is worth the special case.
func fitWindow(cols, want, block int) int {
	if want > cols {
		want = cols
	}
	want -= want % block
	for w := want; w >= block; w -= block {
		if cols%w == 0 {
			return w
		}
	}
	return block
}

// hadamardOf says how wide the rotation is at a site, which is nothing for a
// table stored plain.
func hadamardOf(key string, group int, embd string) int {
	if key == headKey && embd != "rot" {
		return 0
	}
	return group
}

// The sites, kept between runs. Not a GGUF: it is one map of one shape, it is
// read by nothing but this command, and a format nobody else parses is a format
// nobody else can misread. A magic word so that a stale or foreign file is an
// error rather than a model quantized against noise.
const salienceMagic = "golemsal1"

func writeSalience(path string, sal map[string][]float32) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	w := bufio.NewWriter(f)
	if _, err := w.WriteString(salienceMagic); err != nil {
		f.Close()
		return err
	}
	keys := make([]string, 0, len(sal))
	for k := range sal {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	put := func(n uint32) error { return binary.Write(w, binary.LittleEndian, n) }
	if err := put(uint32(len(keys))); err != nil {
		f.Close()
		return err
	}
	for _, k := range keys {
		v := sal[k]
		if err := put(uint32(len(k))); err != nil {
			f.Close()
			return err
		}
		if _, err := w.WriteString(k); err != nil {
			f.Close()
			return err
		}
		if err := put(uint32(len(v))); err != nil {
			f.Close()
			return err
		}
		if err := binary.Write(w, binary.LittleEndian, v); err != nil {
			f.Close()
			return err
		}
	}
	if err := w.Flush(); err != nil {
		f.Close()
		return err
	}
	return f.Close()
}

func readSalience(path string) (map[string][]float32, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	r := bufio.NewReader(f)
	magic := make([]byte, len(salienceMagic))
	if _, err := io.ReadFull(r, magic); err != nil || string(magic) != salienceMagic {
		return nil, fmt.Errorf("golemquant: %s is not a salience file", path)
	}
	get := func() (uint32, error) {
		var n uint32
		err := binary.Read(r, binary.LittleEndian, &n)
		return n, err
	}
	n, err := get()
	if err != nil {
		return nil, err
	}
	out := make(map[string][]float32, n)
	for i := uint32(0); i < n; i++ {
		kn, err := get()
		if err != nil || kn > 1<<10 {
			return nil, fmt.Errorf("golemquant: %s is truncated", path)
		}
		key := make([]byte, kn)
		if _, err := io.ReadFull(r, key); err != nil {
			return nil, err
		}
		vn, err := get()
		if err != nil || vn > 1<<22 {
			return nil, fmt.Errorf("golemquant: %s is truncated", path)
		}
		v := make([]float32, vn)
		if err := binary.Read(r, binary.LittleEndian, v); err != nil {
			return nil, err
		}
		out[string(key)] = v
	}
	return out, nil
}

// saliencyScale turns per-column activation power into the factor the weights
// are multiplied by, normalised so the matrix keeps its overall size.
//
// clamp bounds how far it may go, and it is not a detail: the scale is applied
// before the rotation, and the rotation mixes a hundred and twenty-eight
// columns into each other. A column shrunk by two thousand is mixed with one
// left alone, quantized as if it were the second, and then multiplied back by
// two thousand on the activation side — so its error comes back two thousand
// times larger. At an exponent of 0.75 the span reaches eighteen thousand and
// the model reads at a perplexity of 246 rather than 40. Salience and
// incoherence do not compose freely, and this is where they are made to.
func saliencyScale(sal []float32, alpha, clamp float64) []float32 {
	out := make([]float32, len(sal))
	if alpha == 0 {
		for j := range out {
			out[j] = 1
		}
		return out
	}
	var geo float64
	for _, v := range sal {
		geo += math.Log(math.Max(float64(v), 1e-8))
	}
	geo = math.Exp(geo / float64(len(sal)))
	for j, v := range sal {
		s := math.Pow(math.Max(float64(v), 1e-8)/geo, alpha)
		if clamp > 1 {
			s = math.Min(math.Max(s, 1/clamp), clamp)
		}
		out[j] = float32(s)
	}
	return out
}

// encodable says whether a tensor is one this format has anything to say about:
// a matrix, or a stack of them, of a type that can be read back as floats. The
// norms, the vectors and the scalars are none of those and travel unchanged.
func encodable(t tensors.Tensor, block int) bool {
	if len(t.Shape) != 2 && len(t.Shape) != 3 {
		return false
	}
	// A row has to hold a whole number of whatever the codec's unit is — 64
	// weights for a lattice block, 128 for a trellis sequence. Refuse rather
	// than pad: a padded row is bits nobody reads and an offset nobody expects.
	if t.Shape[0]%block != 0 {
		return false
	}
	if t.DType == "BF16" {
		return true
	}
	if t.DType == "F32" {
		// Never. A float tensor in one of these files is a norm's gain, a
		// scale, or the router's own matrix, and the router is the one thing
		// in a mixture that must not be approximated: a model that picks the
		// wrong experts answers fluently and wrongly, which gemma/moe.go says
		// at more length.
		return false
	}
	_, ok := nn.QuantOf(t.DType)
	return ok
}

// expandBudget is how many bytes of floats one chunk of a matrix may take.
// Bounded, because the number that matters is not the tensor's size but the
// machine's: 248320 rows of 5120 is five gigabytes expanded, and a converter
// that asks for that beside the checkpoint it is reading is a converter the
// kernel stops.
const expandBudget = 64 << 20

// expandRows reads a run of a tensor's rows back as floats, whatever the
// tensor is stored as. from and to count rows of the whole tensor, a stack of
// experts included, because that is how its bytes are laid out.
func expandRows(t tensors.Tensor, from, to int) ([]float32, error) {
	cols := t.Shape[0]
	n := to - from
	switch t.DType {
	case "F32":
		out := make([]float32, n*cols)
		for i := range out {
			out[i] = math.Float32frombits(binary.LittleEndian.Uint32(t.Raw[(from*cols+i)*4:]))
		}
		return out, nil
	case "BF16":
		// A brain float is the top half of a float, so widening is a shift.
		// Done here rather than through Tensor.F32 because that expands the
		// whole tensor, which for a vocabulary table is the five gigabytes
		// this function exists to avoid.
		out := make([]float32, n*cols)
		for i := range out {
			out[i] = math.Float32frombits(uint32(binary.LittleEndian.Uint16(t.Raw[(from*cols+i)*2:])) << 16)
		}
		return out, nil
	}
	q, ok := nn.QuantOf(t.DType)
	if !ok {
		return nil, fmt.Errorf("golemquant: %s is a type this cannot read", t.DType)
	}
	rows := t.Elems() / cols
	m := nn.Matrix{Data: t.Raw, Quant: q, Rows: rows, Cols: cols}
	if want := rows * m.RowBytes(); len(t.Raw) != want {
		return nil, fmt.Errorf("golemquant: a %s tensor of %dx%d wants %d bytes, the file has %d",
			t.DType, rows, cols, want, len(t.Raw))
	}
	out := make([]float32, n*cols)
	compress.Parallel(n, func(lo, hi int) {
		for r := lo; r < hi; r++ {
			m.Row(from+r, out[r*cols:(r+1)*cols])
		}
	})
	return out, nil
}

// reciprocal is the weights' half of a vector given the activations', which is
// what nn.PrepareD4G undoes. The two are elementwise reciprocal by definition
// of the scheme; compress/encode_d4g.go says why.
func reciprocal(v []float32) []float32 {
	if v == nil {
		return nil
	}
	out := make([]float32, len(v))
	for i, x := range v {
		out[i] = 1 / x
	}
	return out
}

// f32Bytes is a vector as the file holds it.
func f32Bytes(v []float32) []byte {
	raw := make([]byte, len(v)*4)
	for i, x := range v {
		binary.LittleEndian.PutUint32(raw[4*i:], math.Float32bits(x))
	}
	return raw
}

// expand reads a tensor back as floats whatever it is stored as. A checkpoint
// that arrives already quantized can be converted — the arithmetic works — but
// what comes out is a quantization of a quantization, and the second one cannot
// undo what the first threw away. The right input is bf16.
func expand(t tensors.Tensor) ([]float32, error) {
	if t.DType == "BF16" || t.DType == "F32" {
		return t.F32()
	}
	q, ok := nn.QuantOf(t.DType)
	if !ok {
		return nil, fmt.Errorf("golemquant: %s is a type this cannot read", t.DType)
	}
	cols := t.Shape[0]
	rows := t.Elems() / cols
	m := nn.Matrix{Data: t.Raw, Quant: q, Rows: rows, Cols: cols}
	if want := rows * m.RowBytes(); len(t.Raw) != want {
		return nil, fmt.Errorf("golemquant: a %s tensor of %dx%d wants %d bytes, the file has %d",
			t.DType, rows, cols, want, len(t.Raw))
	}
	out := make([]float32, t.Elems())
	compress.Parallel(rows, func(lo, hi int) {
		for r := lo; r < hi; r++ {
			m.Row(r, out[r*cols:(r+1)*cols])
		}
	})
	return out, nil
}

// flagWasSet says whether the command line named a flag, so that a default
// chosen for one codebook is not imposed on a caller who chose another.
func flagWasSet(name string) bool {
	found := false
	flag.Visit(func(f *flag.Flag) {
		if f.Name == name {
			found = true
		}
	})
	return found
}

// kept says whether a tensor is one of those a probe is leaving alone, so that
// what the others cost can be read off on its own.
//
// A fragment with a dot in it names a tensor and matches whole fields from the
// end: output.weight is the head and not blk.3.attn_output.weight, which a
// plain substring test made it — seventeen attention projections left in Q4_K
// beside sixty-five blocks of codes, and a stack that refuses the file rather
// than reading half of it one way and half the other. A fragment without a dot
// is a substring, which is how ssm_alpha names forty-eight of them.
func kept(name, list string) bool {
	if list == "" {
		return false
	}
	fields := strings.Split(name, ".")
	for _, frag := range strings.Split(list, ",") {
		if frag == "" {
			continue
		}
		if !strings.Contains(frag, ".") {
			if strings.Contains(name, frag) {
				return true
			}
			continue
		}
		want := strings.Split(frag, ".")
		if len(want) > len(fields) {
			continue
		}
		if strings.Join(fields[len(fields)-len(want):], ".") == frag {
			return true
		}
	}
	return false
}

func parse(name string) (int, string, bool) {
	if !strings.HasPrefix(name, "blk.") {
		return 0, "", false
	}
	parts := strings.Split(name, ".")
	if len(parts) != 4 {
		return 0, "", false
	}
	var n int
	fmt.Sscan(parts[1], &n)
	return n, parts[2], true
}

func sizeOf(n int) string {
	switch {
	case n > 1<<30:
		return fmt.Sprintf("%.2f GiB", float64(n)/(1<<30))
	case n > 1<<20:
		return fmt.Sprintf("%.1f MiB", float64(n)/(1<<20))
	}
	return fmt.Sprintf("%.1f KiB", float64(n)/1024)
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
	`limited by memory bandwidth rather than arithmetic throughput. ` +
	`La compression des poids d'un modèle de langage repose sur l'idée que la ` +
	`distribution des coefficients est fortement redondante. ` +
	`def fibonacci(n):\n    if n < 2:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)\n` +
	`The capital of France is Paris, and the capital of Japan is Tokyo. ` +
	`Water boils at 100 degrees Celsius at sea level, and freezes at zero. ` +
	`Un modèle de langage prédit le prochain jeton à partir de ceux qui précèdent.`
