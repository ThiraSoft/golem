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

	"github.com/ThiraSoft/golem/gemma"
	"github.com/ThiraSoft/golem/qwen"
	"github.com/ThiraSoft/golem/qwen35"
	"github.com/ThiraSoft/golem/tensors"
	"github.com/ThiraSoft/golem/token/bpe"
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
	prefix := flag.String("prefix", "", "text every corpus window opens with, special tokens parsed, and not scored: a chat turn, for an instruct model that reads raw text as noise")
	stream := flag.Bool("stream", false, "carry the checkpoint past the card a window of blocks at a time, widened to floats. For one the resident stack cannot take: too large for the card, or a form no kernel here reads. The processor is the alternative and it reads every weight for every token. Not for a .golem: this streams the body and leaves the head on the processor, and a trellis head is 1.27 billion weights decoded per position — 4088 positions of Qwen3.8-27B did not finish in half an hour, where -vulkan does the whole corpus in under three minutes")
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
	v, err := vocabulary(m)
	must(err)
	t0 := time.Now()

	if *corpus != "" {
		text, err := os.ReadFile(*corpus)
		must(err)
		ids := v.Encode(string(text), false, false)
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
		// The windows, settled before any of them is run, because the streamed
		// path wants them all at once: it carries the model past the card a
		// window of blocks at a time and every position has to meet a block
		// while that block is resident.
		//
		// Every window opens with the same head: the BOS, when the vocabulary
		// wants one, then -prefix. The BOS on every window and not only the
		// first, because Gemma reads a window that starts mid-sentence with
		// none as noise, 338 against 33 on the E2B. The prefix because an
		// instruct Gemma 4 12B reads raw text as noise too — llama.cpp
		// continues "The three laws of thermodynamics are" with a column of
		// underscores just as this does — and a perplexity of noise compares
		// nothing; inside a model's turn the same text is prose. Only the
		// text's own tokens are scored. Qwen wants neither and is cut as it
		// always was.
		var head []int32
		if v.AddBOS() {
			head = append(head, v.BOS())
		}
		if *prefix != "" {
			head = append(head, v.Encode(*prefix, false, true)...)
		}
		windows := cutWindows(ids, head, *ctx)
		skip := max(len(head)-1, 0)

		// A checkpoint the resident stack cannot take, for either of the two
		// reasons it cannot: too large for the card, or a form no kernel here
		// reads. Both are answered the same way — the model goes past the card
		// a window of blocks at a time, widened to floats on the way up by the
		// same nn.Matrix.Row every other reader of a checkpoint uses.
		//
		// It is what makes some of these measurable at all. The processor reads
		// every weight for every token — qwen35's ForwardBatch steps one at a
		// time — which for the 54 GB BF16 is twenty-three minutes for
		// twenty-four positions and about thirty hours for this corpus. The same
		// corpus on the card is nine minutes, and the two answer the same logits
		// to two parts in a million.
		//
		// Note what it does to a comparison: a streamed row is read with float
		// activations where a resident row rounds them to Q8_0 for some
		// projections. So a table mixing the two flatters whichever rows were
		// streamed. Say which is which, or stream all of them.
		var streamed [][][]float32
		if *stream {
			sm, ok := m.(streamedModel)
			if !ok {
				must(fmt.Errorf("vqdiff: this engine has nothing that streams"))
			}
			streamed, err = sm.ForwardStreamed(windows, 0, *ctx)
			must(err)
		}
		var sum float64
		var n int
		for w, window := range windows {
			var h [][]float32
			if streamed != nil {
				h = streamed[w]
			} else {
				m.Reset()
				h = m.ForwardBatch(window, 0)
			}
			// The first token of a window has nothing before it, so it is not
			// predicted and does not count; nor does the rest of the head.
			for i := skip; i < len(window)-1; i++ {
				m.Logits(h[i], logits)
				if cf != nil {
					must(binary.Write(cf, binary.LittleEndian, logits))
				}
				sum += logSoftmaxAt(logits, window[i+1])
				n++
			}
			fmt.Printf("  window %d: running perplexity %.4f over %d tokens\n",
				w, math.Exp(-sum/float64(n)), n)
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

// cutWindows makes windows of ctx positions out of a text, each the head and
// then as much of the text as fills it, the text's tokens never repeated.
func cutWindows(ids, head []int32, ctx int) [][]int32 {
	step := ctx - len(head)
	if step <= 0 {
		return nil
	}
	var windows [][]int32
	for start := 0; start+step <= len(ids); start += step {
		w := make([]int32, 0, ctx)
		w = append(append(w, head...), ids[start:start+step]...)
		windows = append(windows, w)
	}
	return windows
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

type gemmaModel struct{ *gemma.Model }

func (m gemmaModel) Vocab() int { return m.Cfg.Vocab }

// UseVulkan is what engine.UseVulkan does for Gemma, and in the same order: the
// head first, because a head uploaded after the blocks is the one that does not
// fit and lands in host memory without a word.
func (m gemmaModel) UseVulkan() error {
	if err := m.UseVulkanHead(); err != nil {
		return err
	}
	return m.UseVulkanStack()
}

// vocab is the part of a tokenizer this tool reads. Gemma's is token/bpe and
// Qwen's token/bytebpe; each reads the other's file without complaint and
// encodes nonsense, so the choice follows the engine and not the file.
type vocab interface {
	Encode(text string, addBOS, parseSpecial bool) []int32
	Decode(ids []int32, special bool) string
	AddBOS() bool
	BOS() int32
}

func vocabulary(m model) (vocab, error) {
	if _, ok := m.(gemmaModel); ok {
		return bpe.Load(m.File())
	}
	return bytebpe.Load(m.File())
}

// vulkanModel is a model that can put itself on a device. Both engines can;
// naming it here keeps open's return type the narrow one above.
type vulkanModel interface{ UseVulkan() error }

// streamedModel is a model that can carry itself past the card a window of
// blocks at a time. Only the hybrid engine has one, and only a checkpoint whose
// weights no kernel reads — a BF16 — has any reason to use it.
type streamedModel interface {
	ForwardStreamed(runs [][]int32, window, ctx int) ([][][]float32, error)
}

// open reads a checkpoint with whichever engine claims it. qwen refuses an
// architecture it does not know, which is how the others get their turn.
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
	} else if c, err3 := gemma.Open(path, ctx); err3 == nil {
		m = gemmaModel{c}
	} else {
		return nil, fmt.Errorf("no engine reads it: %v; %v; %v", err, err2, err3)
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
