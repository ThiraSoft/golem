package gemma

// The assistant: a drafter Google publishes beside a Gemma 4 checkpoint, which
// guesses the token after the one the model has just chosen.
//
// It is a small model of its own — the 12B's has four blocks of width 1024 —
// that owns no cache. Every block reads the target's: a window block the
// target's last window block, a global block its last global block, which is
// what llama.cpp's shared cache hands it (llama-model.cpp, the share callback of
// LLM_ARCH_GEMMA4_ASSISTANT). And it is fed the target, not a token alone: the
// target's embedding of the token beside the target's normed state from the
// position before, projected down to its own width.
//
// It never writes: the target's cache holds every position before the one being
// drafted from, and nothing at it. So a block sees one position short of its
// own query, which is BlockConfig.Behind.

import (
	"fmt"

	"github.com/ThiraSoft/golem/nn"
	"github.com/ThiraSoft/golem/tensors"
	"github.com/ThiraSoft/golem/vk"
)

// AssistantWeights is what the assistant's file holds. The blocks are ordinary
// Gemma blocks with neither a key nor a value projection.
type AssistantWeights struct {
	// PreProj takes the target's embedding and state, concatenated, to the
	// assistant's width.
	PreProj nn.Matrix
	// PostProj takes the assistant's normed state back to the target's width,
	// which is what a second guess is fed in place of the target's state.
	PostProj   nn.Matrix
	Blocks     []BlockWeights
	OutputNorm []float32
	// TokenEmbd is only ever read as the head: the input embedding is the
	// target's.
	TokenEmbd nn.Matrix
	RoPEFreqs []float32
}

type Assistant struct {
	Cfg *Config
	W   *AssistantWeights

	target  *Model
	file    *tensors.GGUF
	scratch *Scratch
	// view is the target's active cache as the assistant's blocks index it:
	// entry i is the target's cache that block i reads.
	view    *Cache
	xh      *nn.Batch
	xs      [][]float32
	pre     []float32   // the projection, for the tests
	outputs [][]float32 // one per block, for the tests
	hidden  []float32
	head    *nn.Batch
	post    *nn.Batch // the normed state, for the post-projection
	own     []float32 // what the post-projection made of it

	// The card, when the model's blocks are on one: gemma/assistant_vulkan.go.
	stack *vk.Stack
	vhead vk.Head
}

// OpenAssistant binds an assistant file to the model it drafts for. The model
// is the one whose cache it reads; it has to be the checkpoint the assistant
// was trained beside, and what can be checked of that is checked.
func OpenAssistant(path string, target *Model) (*Assistant, error) {
	g, err := tensors.OpenGGUF(path)
	if err != nil {
		return nil, err
	}
	a, err := newAssistant(g, target)
	if err != nil {
		g.Close()
		return nil, err
	}
	return a, nil
}

func newAssistant(g *tensors.GGUF, target *Model) (*Assistant, error) {
	cfg, err := loadAssistantConfig(g, target.Cfg)
	if err != nil {
		return nil, err
	}
	w, err := loadAssistantWeights(g, cfg, target.Cfg)
	if err != nil {
		return nil, err
	}
	a := &Assistant{
		Cfg:     cfg,
		W:       w,
		target:  target,
		file:    g,
		scratch: NewScratch(cfg),
		view:    &Cache{Layers: make([]*LayerCache, len(cfg.Blocks))},
		xh:      nn.NewBatch(2*target.Cfg.Dim, 1),
		xs:      rows(1, cfg.Dim),
		pre:     make([]float32, cfg.Dim),
		outputs: rows(len(cfg.Blocks), cfg.Dim),
		hidden:  make([]float32, cfg.Dim),
		head:    nn.NewBatch(cfg.Dim, 1),
		post:    nn.NewBatch(cfg.Dim, 1),
		own:     make([]float32, target.Cfg.Dim),
	}
	return a, nil
}

func (a *Assistant) Close() error {
	a.closeVulkan()
	return a.file.Close()
}

// Draft writes into out the assistant's scores for the token after `token`.
//
// token is the one the target has just chosen, and sits at pos; hidden is the
// target's state after its final norm at pos-1, the one token was drawn from.
// The target's cache must hold every position before pos, which it does
// between two steps of a generation.
func (a *Assistant) Draft(token int32, hidden []float32, pos int, out []float32) {
	if len(out) != a.Cfg.Vocab {
		panic(fmt.Sprintf("gemma: the assistant scores %d tokens, given room for %d", a.Cfg.Vocab, len(out)))
	}
	a.project(token, hidden)
	if a.stack != nil {
		a.draftOnCard(pos, out)
		return
	}
	for i := range a.Cfg.Blocks {
		a.block(i, pos)
	}
	a.score(out)
}

// DraftNext writes the scores for the token after guess, the one the last Draft
// chose. It is fed the assistant's own state projected back to the target's
// width, at the same position: nothing has been written at pos, so the cache it
// reads is the one the first guess read. llama.cpp's draft-mtp drafts a second
// token the same way (common/speculative.cpp, the is_mem_shared branch).
func (a *Assistant) DraftNext(guess int32, pos int, out []float32) {
	a.ownState()
	a.Draft(guess, a.own, pos, out)
}

// ownState projects the last draft's normed state to the target's width.
func (a *Assistant) ownState() {
	copy(a.post.F[0], a.hidden)
	a.post.QuantizeColumnRange(0, 0, a.Cfg.Dim)
	a.W.PostProj.MatVec(a.post, a.own)
}

// project seeds the stream: the target's embedding of token and its state,
// concatenated and taken to the assistant's width.
func (a *Assistant) project(token int32, hidden []float32) {
	tcfg := a.target.Cfg
	if len(hidden) != tcfg.Dim {
		panic(fmt.Sprintf("gemma: the assistant takes a state of %d, given %d", tcfg.Dim, len(hidden)))
	}
	// The embedding first, the state second: llama.cpp concatenates them in
	// that order.
	x := a.xh.F[0]
	Embed(tcfg, a.target.W, token, x[:tcfg.Dim])
	copy(x[tcfg.Dim:], hidden)
	a.xh.QuantizeColumnRange(0, 0, 2*tcfg.Dim)
	a.W.PreProj.MatVecBatch(a.xh, a.xs)
	copy(a.pre, a.xs[0])
}

// block carries the stream through block i, attending over the target's cache
// as it stands: every position before pos.
func (a *Assistant) block(i, pos int) {
	bc := a.Cfg.Blocks[i]
	// The target's active cache, looked up each time: which conversation that
	// is changes with the slot.
	a.view.Layers[i] = a.target.cache.Layers[bc.KVSource]
	at := []Place{{Cache: a.view, Pos: pos, Until: pos}}
	freqs := a.W.RoPEFreqs
	if bc.Window {
		freqs = nil // the frequency factors belong to the global blocks
	}
	ropes := a.scratch.RoPE(bc, at, freqs)
	Block(a.Cfg, bc, &a.W.Blocks[i], ropes, at, a.scratch, a.xs, nil)
	copy(a.outputs[i], a.xs[0])
}

// score norms the stream and reads it against the assistant's own table, with
// no softcap: the graph has none.
func (a *Assistant) score(out []float32) {
	copy(a.hidden, a.xs[0])
	nn.RMSNormPlain(a.hidden, a.W.OutputNorm, a.Cfg.Eps)
	copy(a.head.F[0], a.hidden)
	a.W.TokenEmbd.MatVec(a.head, out)
}

// loadAssistantConfig reads the assistant's geometry and decides which of the
// target's caches each block reads.
func loadAssistantConfig(g *tensors.GGUF, target *Config) (*Config, error) {
	arch, err := g.String("general.architecture")
	if err != nil {
		return nil, err
	}
	if arch != "gemma4-assistant" {
		return nil, fmt.Errorf("architecture %q is not gemma4-assistant", arch)
	}
	key := func(suffix string) string { return arch + "." + suffix }
	u32 := func(suffix string) (int, error) {
		v, err := g.Uint32(key(suffix))
		return int(v), err
	}

	blockCount, err := u32("block_count")
	if err != nil {
		return nil, err
	}
	dim, err := u32("embedding_length")
	if err != nil {
		return nil, err
	}
	// The width of the state it is fed, which has to be the target's.
	out, err := u32("embedding_length_out")
	if err != nil {
		return nil, err
	}
	if out != target.Dim {
		return nil, fmt.Errorf("the assistant takes a state of %d and the model's is %d", out, target.Dim)
	}
	eps, err := g.Float32(key("attention.layer_norm_rms_epsilon"))
	if err != nil {
		return nil, err
	}
	heads, err := g.Uint32Slice(key("attention.head_count"))
	if err != nil {
		return nil, err
	}
	kvHeads, err := g.Uint32Slice(key("attention.head_count_kv"))
	if err != nil {
		return nil, err
	}
	ffn, err := g.Uint32Slice(key("feed_forward_length"))
	if err != nil {
		return nil, err
	}
	pattern, err := g.BoolSlice(key("attention.sliding_window_pattern"))
	if err != nil {
		return nil, err
	}
	if len(pattern) != blockCount {
		return nil, fmt.Errorf("the window pattern has %d entries for %d blocks", len(pattern), blockCount)
	}
	window, err := u32("attention.sliding_window")
	if err != nil {
		return nil, err
	}
	headGlobal, err := u32("attention.key_length")
	if err != nil {
		return nil, err
	}
	headWindow, err := u32("attention.key_length_swa")
	if err != nil {
		return nil, err
	}
	ropeGlobal, err := g.Float32(key("rope.freq_base"))
	if err != nil {
		return nil, err
	}
	ropeWindow, err := g.Float32(key("rope.freq_base_swa"))
	if err != nil {
		return nil, err
	}
	ropeDimsGlobal, err := u32("rope.dimension_count")
	if err != nil {
		return nil, err
	}
	ropeDimsWindow, err := u32("rope.dimension_count_swa")
	if err != nil {
		return nil, err
	}

	embd, ok := g.Tensors["token_embd.weight"]
	if !ok || len(embd.Shape) != 2 {
		return nil, fmt.Errorf("token_embd.weight is absent or not two-dimensional")
	}
	if embd.Shape[1] != target.Vocab {
		return nil, fmt.Errorf("the assistant scores %d tokens and the model %d", embd.Shape[1], target.Vocab)
	}

	// llama.cpp does not search: a window block reads the target's
	// second-to-last block and a global block its last. Whether those are of
	// the right kind is checked rather than assumed.
	last := len(target.Blocks) - 1
	sources := map[bool]int{false: last, true: last - 1}

	cfg := &Config{
		Arch:       arch,
		Dim:        dim,
		Vocab:      embd.Shape[1],
		Eps:        eps,
		MaxContext: target.MaxContext,
		Blocks:     make([]BlockConfig, blockCount),
	}
	for i := range cfg.Blocks {
		b := BlockConfig{
			Index:    i,
			Window:   pattern[i],
			Heads:    perBlock(heads, i),
			KVHeads:  perBlock(kvHeads, i),
			FFN:      perBlock(ffn, i),
			KVSource: sources[pattern[i]],
			Behind:   true,
		}
		if b.Window {
			b.HeadDim, b.RoPEBase, b.RoPEDims, b.WindowSize = headWindow, float64(ropeWindow), ropeDimsWindow, window
		} else {
			b.HeadDim, b.RoPEBase, b.RoPEDims = headGlobal, float64(ropeGlobal), ropeDimsGlobal
		}
		src := target.Blocks[b.KVSource]
		if src.Window != b.Window || src.WindowSize != b.WindowSize {
			return nil, fmt.Errorf("assistant block %d would read the model's block %d, which is of another kind", i, b.KVSource)
		}
		if src.KVHeads != b.KVHeads || src.HeadDim != b.HeadDim {
			return nil, fmt.Errorf("assistant block %d reads %d heads of %d, and the model's block %d holds %d of %d",
				i, b.KVHeads, b.HeadDim, b.KVSource, src.KVHeads, src.HeadDim)
		}
		if b.Heads%b.KVHeads != 0 {
			return nil, fmt.Errorf("assistant block %d: %d query heads do not divide among %d key-value heads", i, b.Heads, b.KVHeads)
		}
		cfg.Blocks[i] = b
	}
	return cfg, nil
}

func loadAssistantWeights(g *tensors.GGUF, cfg *Config, target *Config) (*AssistantWeights, error) {
	w := &AssistantWeights{Blocks: make([]BlockWeights, len(cfg.Blocks))}
	var err error
	if w.PreProj, err = matrix(g, "nextn.pre_projection.weight"); err != nil {
		return nil, err
	}
	if w.PreProj.Rows != cfg.Dim || w.PreProj.Cols != 2*target.Dim {
		return nil, fmt.Errorf("the pre-projection is %dx%d, expected %dx%d", w.PreProj.Rows, w.PreProj.Cols, cfg.Dim, 2*target.Dim)
	}
	if w.TokenEmbd, err = matrix(g, "token_embd.weight"); err != nil {
		return nil, err
	}
	if w.OutputNorm, err = floats(g, "output_norm.weight"); err != nil {
		return nil, err
	}
	if w.RoPEFreqs, err = floats(g, "rope_freqs.weight"); err != nil {
		return nil, err
	}
	if w.PostProj, err = matrix(g, "nextn.post_projection.weight"); err != nil {
		return nil, err
	}
	if w.PostProj.Rows != target.Dim || w.PostProj.Cols != cfg.Dim {
		return nil, fmt.Errorf("the post-projection is %dx%d, expected %dx%d", w.PostProj.Rows, w.PostProj.Cols, target.Dim, cfg.Dim)
	}

	for i, bc := range cfg.Blocks {
		p := fmt.Sprintf("blk.%d.", i)
		b := BlockWeights{OutScale: 1}
		for _, f := range []struct {
			into *[]float32
			name string
		}{
			{&b.AttnNorm, "attn_norm"}, {&b.QNorm, "attn_q_norm"},
			{&b.PostAttnNorm, "post_attention_norm"}, {&b.FFNNorm, "ffn_norm"},
			{&b.PostFFWNorm, "post_ffw_norm"},
		} {
			if *f.into, err = floats(g, p+f.name+".weight"); err != nil {
				return nil, err
			}
		}
		for _, m := range []struct {
			into *nn.Matrix
			name string
		}{
			{&b.Q, "attn_q"}, {&b.O, "attn_output"},
			{&b.Gate, "ffn_gate"}, {&b.Up, "ffn_up"}, {&b.Down, "ffn_down"},
		} {
			if *m.into, err = matrix(g, p+m.name+".weight"); err != nil {
				return nil, err
			}
		}
		if s, err := floats(g, p+"layer_output_scale.weight"); err == nil && len(s) == 1 {
			b.OutScale = s[0]
		}
		if b.Q.Rows != bc.Heads*bc.HeadDim || b.Q.Cols != cfg.Dim {
			return nil, fmt.Errorf("assistant block %d: query is %dx%d, expected %dx%d",
				i, b.Q.Rows, b.Q.Cols, bc.Heads*bc.HeadDim, cfg.Dim)
		}
		if b.O.Rows != cfg.Dim || b.O.Cols != bc.Heads*bc.HeadDim {
			return nil, fmt.Errorf("assistant block %d: output projection is %dx%d", i, b.O.Rows, b.O.Cols)
		}
		if b.Gate.Rows != bc.FFN || b.Down.Cols != bc.FFN {
			return nil, fmt.Errorf("assistant block %d: feed forward is %d wide, expected %d", i, b.Gate.Rows, bc.FFN)
		}
		if len(b.QNorm) != bc.HeadDim {
			return nil, fmt.Errorf("assistant block %d: query norm has %d entries for a head of %d", i, len(b.QNorm), bc.HeadDim)
		}
		w.Blocks[i] = b
	}

	all := []*nn.Matrix{&w.PreProj, &w.PostProj}
	for i := range w.Blocks {
		b := &w.Blocks[i]
		all = append(all, &b.Q, &b.O, &b.Gate, &b.Up, &b.Down)
	}
	nn.InParallel(len(all), 1<<30, func(first, last int) {
		for i := first; i < last; i++ {
			all[i].Repack()
		}
	})
	return w, nil
}
