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
	"github.com/ThiraSoft/golem/qwen35"
	"github.com/ThiraSoft/golem/tensors"
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
	vulkan := flag.Bool("vulkan", false, "run the model on a Vulkan device")
	flag.Parse()

	// The window, not a fixed four thousand. A context is the cache a model
	// keeps and the width of every attention dispatch that reads it: on a
	// twenty-seven billion parameter model four thousand positions is a kernel
	// long enough for the driver's watchdog to call the ring hung, and it
	// resets the card mid-measurement. Nothing here ever reads past one window.
	ctxSize := *ctx
	if *steps > 0 {
		ctxSize += *steps + len(heldOut)/2
	}
	m, err := open(*model, ctxSize, *vulkan)
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
		logits := make([]float32, m.Vocab())
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
	logits := make([]float32, m.Vocab())
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

// A model is as much of one as this tool reads, so that a checkpoint of either
// architecture can be measured by the same instrument. The two packages have
// the same shape here and no interface of their own; naming one is cheaper than
// a second copy of the loop.
type model interface {
	Close() error
	File() *tensors.GGUF
	Vocab() int
	Reset()
	ForwardBatch(tokens []int32, startPos int) [][]float32
	Logits(hidden, out []float32)
}

type qwenModel struct{ *qwen.Model }

func (m qwenModel) Vocab() int { return m.Cfg.Vocab }

type qwen35Model struct{ *qwen35.Model }

func (m qwen35Model) Vocab() int { return m.Cfg.Vocab }

// vulkanModel is a model that can put itself on a device. Both engines can;
// naming it here keeps open's return type the narrow one above.
type vulkanModel interface{ UseVulkan() error }

// open reads a checkpoint with whichever engine claims it. qwen refuses an
// architecture it does not know, which is how the second gets its turn.
//
// onDevice is not an optimisation here, it is what makes the measurement
// possible. A perplexity over four thousand positions of a twenty-seven
// billion parameter model is an hour of eight cores and a few seconds of a
// card, and a quality number nobody runs is a quality number nobody has — which
// is how a format came to be judged by how well each of its matrices agreed
// with its own codes.
func open(path string, ctx int, onDevice bool) (model, error) {
	var m model
	if a, err := qwen.Open(path, ctx); err == nil {
		m = qwenModel{a}
	} else if b, err2 := qwen35.Open(path, ctx); err2 == nil {
		m = qwen35Model{b}
	} else {
		return nil, fmt.Errorf("neither engine reads it: %v; %v", err, err2)
	}
	if onDevice {
		v, ok := m.(vulkanModel)
		if !ok {
			return nil, fmt.Errorf("this engine has nothing that moves to a device")
		}
		if err := v.UseVulkan(); err != nil {
			m.Close()
			return nil, fmt.Errorf("on the device: %w", err)
		}
	}
	return m, nil
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
