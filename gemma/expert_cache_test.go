package gemma

// How much of a token's expert reads a cache of a given size would already
// hold.
//
// This is the number the whole streamed-mixture design rests on, and it is
// measured here rather than assumed because it can be: the router is
// arithmetic, so simulating a cache against a real routing costs a map and no
// card time at all. Nothing on the device is touched and nothing the model
// computes changes — RouteWatch reads the identifiers the processor's router
// has already chosen.
//
// What the table means. A miss is 3.3 MB across a bus this machine measures at
// 6.7 GB/s (vk/upload_test.go), so 240 misses — a token of this model routing
// to eight experts over thirty blocks with nothing resident — is 120 ms, and a
// resident token is about 10. The tokens/s column is that arithmetic and not a
// measurement of anything that has been built.
//
// **The traps, and why the shape of this test is what it is.** Prefill routes
// almost every expert of every block and would drag the figure to a floor that
// decoding never sees, so only decode is counted. A cold cache would drag it
// the other way, so the first tokens of decoding are recorded and then thrown
// away. And a short or repetitive continuation concentrates the routing on a
// handful of experts and reports a hit rate nobody will ever get, which is why
// this wants a few hundred tokens and says how many it took.

import (
	"container/list"
	"os"
	"testing"

	"github.com/ThiraSoft/golem/nn"
	"github.com/ThiraSoft/golem/token/bpe"
)

// defaultCachePrompt asks for something long and varied enough that the
// continuation moves between subjects, because a routing that stays on one
// subject stays on a handful of experts and reports a hit rate that no
// conversation will reproduce.
const defaultCachePrompt = "Write a detailed technical essay comparing three " +
	"very different subjects, one after another and at length: how a modern GPU " +
	"schedules compute work across its shader cores; how sourdough fermentation " +
	"changes the gluten structure of bread over eighteen hours; and how the " +
	"Portuguese language diverged from Galician between the twelfth and the " +
	"sixteenth centuries. Take each in turn, in full paragraphs.\n\n"

// A routeLog is one decoded token's routing: the experts each mixture block
// chose, in block order.
type routeLog [][]int32

// TestExpertCacheHitRate decodes a continuation and reports what a cache of
// each size would have held.
func TestExpertCacheHitRate(t *testing.T) {
	if testing.Short() {
		t.Skip("decodes several hundred tokens on the processor")
	}
	path := model26BPath(t)
	m, err := Open(path, 2048)
	if err != nil {
		t.Skipf("open: %v", err)
	}
	defer m.Close()
	cfg := m.Cfg
	if cfg.Experts == 0 {
		t.Skipf("%s is not a mixture", path)
	}

	// One expert's three matrices, in the form the card reads. It is the same
	// arithmetic vk/mixture.go does over the whole stack, on one expert.
	var expertBytes int
	for i := range cfg.Blocks {
		if !cfg.Blocks[i].MoE {
			continue
		}
		bw := &m.W.Blocks[i]
		for _, e := range []ExpertStack{bw.GateUpExps, bw.DownExps} {
			one := nn.Matrix{Quant: e.Quant, Rows: e.Rows, Cols: e.Cols}
			expertBytes += one.RowBytes() * e.Rows
		}
		break
	}

	// The routing, block by block, for every position the model is given.
	var log []routeLog
	var current routeLog
	RouteWatch = func(ids [][]int32) {
		// One row a position. Decoding is a batch of one; a prefill batch is
		// several, and only the last position of it continues the sequence.
		row := ids[len(ids)-1]
		current = append(current, append([]int32(nil), row...))
	}
	defer func() { RouteWatch = nil }()

	// A prompt with some substance to it, and then the model's own
	// continuation. What matters is that the routing is the model's own and
	// varied: a loop over one token routes to the same eight experts every time
	// and would report a hit rate near one. GOLEM_CACHE_PROMPT replaces the text
	// so the figure can be taken over something else and compared.
	vocab, err := bpe.Load(m.File())
	if err != nil {
		t.Skipf("no tokenizer: %v", err)
	}
	text := os.Getenv("GOLEM_CACHE_PROMPT")
	if text == "" {
		text = defaultCachePrompt
	}
	prompt := vocab.Encode(text, true, false)
	t.Logf("prompt of %d tokens", len(prompt))
	pos := 0
	var hidden []float32
	for _, tok := range prompt {
		current = nil
		hidden = m.Forward(tok, pos)
		pos++
	}
	// Nothing above was recorded: prefill routes far more widely than decoding
	// does — near enough every expert of every block — and mixing the two
	// answers a question nobody asked.

	const decode = 320
	logits := make([]float32, cfg.Vocab)
	for i := 0; i < decode; i++ {
		m.Logits(hidden, logits)
		tok := Argmax(logits)
		current = nil
		hidden = m.Forward(tok, pos)
		pos++
		log = append(log, current)
	}
	if len(log) == 0 || len(log[0]) == 0 {
		t.Fatal("the routing was never watched: no mixture block ran")
	}
	blocks := len(log[0])
	perToken := blocks * cfg.ExpertsUsed
	pool := blocks * cfg.Experts
	t.Logf("%d mixture blocks, %d experts each, %d used: %d reads a token out of a pool of %d",
		blocks, cfg.Experts, cfg.ExpertsUsed, perToken, pool)
	t.Logf("one expert is %.2f MB, the pool is %.2f GB, a token reads %.1f MB",
		float64(expertBytes)/1e6, float64(pool)*float64(expertBytes)/1e9,
		float64(perToken)*float64(expertBytes)/1e6)

	// The first tokens of decoding fill an empty cache and would be counted as
	// misses that a warm cache never pays.
	const warm = 64
	if len(log) <= warm {
		t.Fatalf("only %d tokens decoded, which is not past the warm-up", len(log))
	}

	t.Logf("%-9s %-8s %8s %10s %9s %9s", "resident", "slots", "VRAM", "hit rate", "misses/tok", "tokens/s")
	for _, share := range []float64{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9} {
		slots := int(float64(pool) * share)
		if slots < 1 {
			continue
		}
		hits, misses := simulateLRU(log, slots, warm)
		rate := float64(hits) / float64(hits+misses)
		perTok := float64(misses) / float64(len(log)-warm)
		// 6.7 GB/s is what vk/upload_test.go measures for the crossing alone,
		// and 10 ms is roughly what a resident token of this model costs.
		seconds := 0.010 + perTok*float64(expertBytes)/6.7e9
		t.Logf("%-9.0f%% %-8d %6.2f GB %9.1f%% %10.1f %9.1f",
			share*100, slots, float64(slots)*float64(expertBytes)/1e9,
			rate*100, perTok, 1/seconds)
	}

	// The same capacity split evenly between the blocks rather than shared.
	// FreeToken's claim is that one pool over every (block, expert) beats a
	// per-block split, and this is that claim on this checkpoint.
	t.Log("the same capacity, but each block holding its own share:")
	for _, share := range []float64{0.3, 0.5, 0.7} {
		slots := int(float64(pool) * share)
		hits, misses := simulatePerBlock(log, slots/blocks, warm)
		rate := float64(hits) / float64(hits+misses)
		t.Logf("%-9.0f%% %-8d %9.1f%%", share*100, slots/blocks, rate*100)
	}
}

// simulateLRU counts hits and misses against one cache shared by every block,
// keyed by the pair of a block and an expert. Everything before `warm` tokens is
// played through the cache and not counted, so the figure is a warm one.
//
// container/list rather than a slice: the shared cache holds a few thousand
// slots and a token touches two hundred and forty of them, so promoting by
// shifting a slice would be a billion moves over a run of this length. It was
// written that way first and would have made the sweep the slow part of a test
// whose point is that it costs nothing.
func simulateLRU(log []routeLog, slots, warm int) (hits, misses int) {
	c := newLRU(slots)
	for tok, one := range log {
		for b, ids := range one {
			for _, e := range ids {
				hit := c.touch(lruKey{int32(b), e})
				switch {
				case tok < warm:
				case hit:
					hits++
				default:
					misses++
				}
			}
		}
	}
	return hits, misses
}

// simulatePerBlock is the same with one cache of `each` slots per block, which
// is what a residency chosen from the checkpoint rather than from the routing
// amounts to.
func simulatePerBlock(log []routeLog, each, warm int) (hits, misses int) {
	if each < 1 {
		return 0, 1
	}
	caches := make([]*lru, len(log[0]))
	for i := range caches {
		caches[i] = newLRU(each)
	}
	for tok, one := range log {
		for b, ids := range one {
			for _, e := range ids {
				hit := caches[b].touch(lruKey{int32(b), e})
				switch {
				case tok < warm:
				case hit:
					hits++
				default:
					misses++
				}
			}
		}
	}
	return hits, misses
}

// An lruKey names one expert of one block. The block is part of the key even in
// the per-block caches, so that the two simulations differ only in how the
// capacity is divided.
type lruKey struct{ block, expert int32 }

// An lru is a fixed number of slots with a recency order.
type lru struct {
	slots int
	at    map[lruKey]*list.Element
	order *list.List // least recently used at the front
}

func newLRU(slots int) *lru {
	return &lru{slots: slots, at: make(map[lruKey]*list.Element, slots), order: list.New()}
}

// touch says whether the key was already held, and leaves it held and most
// recent either way.
func (c *lru) touch(k lruKey) bool {
	if e, ok := c.at[k]; ok {
		c.order.MoveToBack(e)
		return true
	}
	if c.order.Len() == c.slots {
		oldest := c.order.Front()
		delete(c.at, oldest.Value.(lruKey))
		c.order.Remove(oldest)
	}
	c.at[k] = c.order.PushBack(k)
	return false
}
