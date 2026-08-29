package main

// vqdiff: what a compressed checkpoint costs in answers rather than in bits.
//
// Two numbers and one sample. The perplexity of a held-out text says whether
// the model still predicts language; the greedy continuation of a prompt says
// whether it still says the same thing. The texts are not the calibration text.

import (
	"encoding/binary"
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"os"
	"time"

	"github.com/ThiraSoft/golem/qwen"
	"github.com/ThiraSoft/golem/token/bytebpe"
)

type report struct {
	Model      string    `json:"model"`
	Perplexity float64   `json:"perplexity"`
	Tokens     int       `json:"tokens"`
	LogProbs   []float64 `json:"logprobs"`
	Greedy     []int32   `json:"greedy"`
	Text       string    `json:"text"`
	Seconds    float64   `json:"seconds"`
}

func main() {
	model := flag.String("model", "", "GGUF")
	out := flag.String("out", "", "where to write the report")
	dump := flag.String("dump", "", "where to write the raw logits of the held-out text")
	prompt := flag.String("prompt", "The three laws of thermodynamics are", "greedy prompt")
	corpus := flag.String("corpus", "", "a text file to measure perplexity over, in windows")
	ctx := flag.Int("ctx", 512, "window size for the corpus")
	limit := flag.Int("limit", 4096, "how many corpus tokens to read")
	steps := flag.Int("steps", 40, "greedy steps")
	flag.Parse()

	m, err := qwen.Open(*model, 4096)
	must(err)
	defer m.Close()
	v, err := bytebpe.Load(m.File())
	must(err)
	t0 := time.Now()

	if *corpus != "" {
		text, err := os.ReadFile(*corpus)
		must(err)
		ids := v.Encode(string(text), true, false)
		if len(ids) > *limit {
			ids = ids[:*limit]
		}
		logits := make([]float32, m.Cfg.Vocab)
		// The corpus is the evaluation set, so it is the one whose opinions are
		// worth comparing between two models. Perplexity says whether a model
		// still predicts language; these say whether it still says the same
		// thing as the model it was made from, which is a different question
		// and the one a compression has to answer.
		var cf *os.File
		if *dump != "" {
			cf, err = os.Create(*dump)
			must(err)
			defer cf.Close()
		}
		var sum float64
		var n int
		for start := 0; start+*ctx <= len(ids); start += *ctx {
			m.Reset()
			window := ids[start : start+*ctx]
			h := m.ForwardBatch(window, 0)
			// The first token of a window has nothing before it, so it is not
			// predicted and does not count.
			for i := 0; i < len(window)-1; i++ {
				m.Logits(h[i], logits)
				if cf != nil {
					must(binary.Write(cf, binary.LittleEndian, logits))
				}
				sum += logSoftmaxAt(logits, window[i+1])
				n++
			}
			fmt.Printf("  window %d: running perplexity %.4f over %d tokens\n",
				start / *ctx, math.Exp(-sum/float64(n)), n)
		}
		fmt.Printf("%s\n  corpus perplexity %.4f over %d tokens in %s\n",
			*model, math.Exp(-sum/float64(n)), n, time.Since(t0).Round(time.Second))
		return
	}

	// Teacher-forced perplexity on the held-out text.
	ids := v.Encode(heldOut, true, false)
	h := m.ForwardBatch(ids, 0)
	logits := make([]float32, m.Cfg.Vocab)
	var df *os.File
	if *dump != "" {
		df, err = os.Create(*dump)
		must(err)
	}
	var sum float64
	lp := make([]float64, 0, len(ids)-1)
	for i := 0; i < len(ids)-1; i++ {
		m.Logits(h[i], logits)
		if df != nil {
			must(binary.Write(df, binary.LittleEndian, logits))
		}
		p := logSoftmaxAt(logits, ids[i+1])
		lp = append(lp, p)
		sum += p
	}
	ppl := math.Exp(-sum / float64(len(lp)))

	if df != nil {
		must(df.Close())
	}

	// Greedy continuation, from a clean cache.
	m.Reset()
	pids := v.Encode(*prompt, true, false)
	h = m.ForwardBatch(pids, 0)
	m.Logits(h[len(h)-1], logits)
	next := argmax(logits)
	var gen []int32
	for s := 0; s < *steps; s++ {
		gen = append(gen, next)
		h = m.ForwardBatch([]int32{next}, len(pids)+s)
		m.Logits(h[0], logits)
		next = argmax(logits)
	}

	r := report{Model: *model, Perplexity: ppl, Tokens: len(lp), LogProbs: lp,
		Greedy: gen, Text: v.Decode(gen, false), Seconds: time.Since(t0).Seconds()}
	fmt.Printf("%s\n  perplexity %.4f over %d tokens\n  continuation: %q\n",
		*model, ppl, len(lp), r.Text)
	if *out != "" {
		f, err := os.Create(*out)
		must(err)
		must(json.NewEncoder(f).Encode(r))
		must(f.Close())
	}
}

func logSoftmaxAt(l []float32, id int32) float64 {
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
	return float64(l[id]) - mx - math.Log(s)
}

func argmax(l []float32) int32 {
	best, bi := l[0], int32(0)
	for i, v := range l {
		if v > best {
			best, bi = v, int32(i)
		}
	}
	return bi
}

func must(err error) {
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

const heldOut = `Attention mechanisms let a model weigh every earlier position ` +
	`when it forms the representation of the current one. A transformer stacks ` +
	`such layers, each followed by a position-wise feed-forward network, and the ` +
	`residual connections around them keep the gradient path short. ` +
	`Le soleil se levait à peine sur la vallée lorsque les premiers voyageurs ` +
	`atteignirent le col, épuisés mais soulagés d'avoir franchi la nuit sans encombre. ` +
	`In 1687 Newton published the Principia, which set out the three laws of motion ` +
	`and the law of universal gravitation. The result was a single framework that ` +
	`accounted for the fall of an apple and the orbit of the moon alike.`
