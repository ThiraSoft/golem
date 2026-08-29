package qwen35

import (
	"github.com/ThiraSoft/golem/nn"
)

// The multi-token-prediction block, which the checkpoint carries as one extra
// decoder layer past the trunk (blk.64 here, `qwen35.nextn_predict_layers = 1`).
//
// It is an ordinary full-attention block with a two-input front: the embedding
// of the token just decided and the trunk's hidden state for the token before
// it, each under its own norm, concatenated and projected back down to one
// hidden state. From there it is a Qwen3.8 block and the model's own logit
// head. llama.cpp's models/qwen35.cpp, graph_mtp, is the other copy of this.
//
// The hidden state it takes is the one **under the model's output norm** —
// llama.cpp names it `h_nextn` and takes it after `build_norm(cur,
// output_norm)`, not before. Feeding it the raw residual instead is a quiet
// wrong answer, not a crash: the draft still reads like language and simply
// agrees with the model far less often.

// HasMTP reports whether the checkpoint carries a prediction block.
func (m *Model) HasMTP() bool {
	return m.W != nil && m.W.MTP.HasMTP && m.Cfg.NextN > 0 && len(m.Cfg.Blocks) > m.trunk()
}

// MTPScratch is the block's own working memory, kept apart from the trunk's so
// that a draft cannot tread on a pass that is still being read.
type MTPScratch struct {
	e      []float32
	eh     []float32
	x      []float32
	h      []float32
	batch  *nn.Batch
	inner  *Scratch
	cache  *BlockCache
	logits []float32
}

func (m *Model) newMTPScratch() *MTPScratch {
	dim := m.Cfg.Dim
	return &MTPScratch{
		e:      make([]float32, dim),
		eh:     make([]float32, dim*2),
		x:      make([]float32, dim),
		h:      make([]float32, dim),
		batch:  nn.NewBatch(dim*2, 1),
		inner:  NewScratch(m.Cfg),
		cache:  newBlockCache(m.Cfg, m.Cfg.Blocks[m.trunk()]),
		logits: make([]float32, m.Cfg.Vocab),
	}
}

// ResetMTP clears the prediction block's own key-value cache. A draft writes
// into it at the position it drafts for, and a rejected draft leaves a key
// there that the next one at that position overwrites — so the only thing that
// has to be cleared is a new conversation.
func (m *Model) ResetMTP() {
	if m.mtp != nil {
		clear(m.mtp.cache.Keys)
		clear(m.mtp.cache.Values)
	}
	if m.gpuPipe != nil && m.gpuPipe.HasMTP() {
		if err := m.gpuPipe.ResetMTPCache(); err != nil {
			panic("qwen35: cannot clear the prediction block's cache: " + err.Error())
		}
	}
}

// ForwardMTP drafts the token after next.
//
// token is the token just decided, which sits at pos; hidden is the trunk's
// output-normed state for the token before it. draftLogits receives the
// distribution over the token that would follow `token`.
func (m *Model) ForwardMTP(token int32, hidden []float32, pos int, draftLogits []float32) []float32 {
	return m.ForwardMTPAt(token, hidden, Place{Slot: m.slot, Pos: pos, T: pos, H: pos, W: pos}, draftLogits)
}

// ForwardMTPAt is ForwardMTP for a draft whose axes do not follow the cache
// index — a token drafted straight after an image, whose h and w carry the
// grid's last patch rather than its own position.
func (m *Model) ForwardMTPAt(token int32, hidden []float32, at Place, draftLogits []float32) []float32 {
	if !m.HasMTP() {
		return nil
	}
	if m.mtp == nil {
		m.mtp = m.newMTPScratch()
	}
	s := m.mtp
	cfg, dim := m.Cfg, m.Cfg.Dim
	il := m.trunk()

	// [ norm(embed(token)) | norm(hidden) ] -> eh_proj
	m.W.TokenEmbd.Row(int(token), s.e)
	nn.RMSNormPlain(s.e, m.W.MTP.ENorm, cfg.Eps)
	copy(s.eh[:dim], s.e)
	copy(s.eh[dim:], hidden)
	nn.RMSNormPlain(s.eh[dim:], m.W.MTP.HNorm, cfg.Eps)

	if m.gpuPipe != nil && m.gpuPipe.HasMTP() {
		h, err := m.gpuPipe.DraftMTPAt(s.eh, at.gpu())
		if err != nil {
			panic("qwen35: the prediction block failed on the card: " + err.Error())
		}
		copy(s.h, h)
		m.Logits(s.h, draftLogits)
		return s.h
	}

	copy(s.batch.F[0], s.eh)
	s.batch.QuantizeColumnRange(0, 0, dim*2)
	if s.batch.QK != nil {
		s.batch.QuantizeK()
	}
	m.W.MTP.EHProj.MatVec(s.batch, s.x)

	// The prediction block rotates by the same rule the trunk does, at the
	// place it drafts for.
	Block(cfg, cfg.Blocks[il], &m.W.Blocks[il], s.cache, m.rope, at, s.x, s.inner)

	copy(s.h, s.x)
	norm := m.W.MTP.SharedHeadNorm
	if len(norm) == 0 {
		norm = m.W.OutputNorm
	}
	nn.RMSNormPlain(s.h, norm, cfg.Eps)

	m.Logits(s.h, draftLogits)
	return s.h
}
