package gemma

// The 12B's assistant against llama.cpp's draft-mtp, waypoint by waypoint and
// then over a greedy run.
//
// The fixtures are dump_assistant's (ref/dump_assistant.cpp): one draft's
// intermediate activations, the target's normed state it was fed, and the
// input of the two target blocks whose caches the assistant reads. The
// waypoint test fills those two caches from the reference's own input, so
// that what it measures is the assistant and not forty-six blocks of the
// target's rounding in front of it.

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/ThiraSoft/golem/nn"
)

func assistant12BPath(tb testing.TB) string {
	tb.Helper()
	path := os.Getenv("GOLEM_ASSISTANT_12B")
	if path == "" {
		tb.Skip("set GOLEM_ASSISTANT_12B to the 12B's gemma4-assistant GGUF to run this test")
	}
	if _, err := os.Stat(path); err != nil {
		tb.Skipf("GOLEM_ASSISTANT_12B names %s, which is not there", path)
	}
	return path
}

// draftFixture is a dump_assistant recording: a fixture, and the assistant's
// guess at every step of the greedy run beside the target's choice.
type draftFixture struct {
	*fixture
	Assistant string  `json:"assistant"`
	Drafts    []int32 `json:"drafts"`
	// Drafts2 is the second guess of each step: the first fed back with the
	// assistant's own projected state, at the same position.
	Drafts2 []int32 `json:"drafts2"`
}

func loadDraftFixture(t *testing.T, name string) *draftFixture {
	t.Helper()
	f := &draftFixture{fixture: loadFixture(t, name)}
	raw, err := os.ReadFile(filepath.Join(f.dir, "index.json"))
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(raw, f); err != nil {
		t.Fatalf("%s/index.json: %v", name, err)
	}
	if len(f.Drafts) != len(f.Greedy) {
		t.Fatalf("%s: %d drafts for %d greedy steps", name, len(f.Drafts), len(f.Greedy))
	}
	return f
}

func openAssistant12B(t *testing.T) (*Model, *Assistant) {
	t.Helper()
	path := assistant12BPath(t)
	m := open12BEngine(t)
	// Through the model, which is what follows it onto a card and closes it.
	if err := m.OpenAssistant(path); err != nil {
		t.Fatal(err)
	}
	return m, m.assistant
}

// fillFromReference runs the target's blocks 46 and 47 over the prompt from the
// reference's own input, which leaves in the model's cache what llama.cpp's
// held when the assistant read it.
func fillFromReference(t *testing.T, f *draftFixture, m *Model) {
	t.Helper()
	s := NewScratch(m.Cfg)
	xs := rows(1, m.Cfg.Dim)
	for _, il := range []int{46, 47} {
		bc := m.Cfg.Blocks[il]
		freqs := m.W.RoPEFreqs
		if bc.Window {
			freqs = nil
		}
		for pos := range f.Tokens {
			copy(xs[0], f.column(t, "l_out-"+itoa(il-1), pos))
			at := Run(m.cache, pos, 1)
			Block(m.Cfg, bc, &m.W.Blocks[il], s.RoPE(bc, at, freqs), at, s, xs, nil)
		}
	}
}

// What fillFromReference leaves in the two caches, against the keys and values
// llama.cpp stored: the assistant's attention is only as right as these.
func TestAssistantTargetCaches12B(t *testing.T) {
	f := loadDraftFixture(t, "assistant12")
	m := open12BEngine(t)
	fillFromReference(t, f, m)
	for _, il := range []int{46, 47} {
		bc := m.Cfg.Blocks[il]
		lc := m.cache.Layers[il]
		got := make([]float32, bc.KVHeads*bc.HeadDim)
		widen := func(half []uint16, into []float32) {
			clear(into)
			nn.AxpyHalf(into, half, 1)
		}
		for pos := range f.Tokens {
			for h := 0; h < bc.KVHeads; h++ {
				widen(lc.Key(pos, h), got[h*bc.HeadDim:(h+1)*bc.HeadDim])
			}
			compareRelative(t, "Kcur_pos-"+itoa(il)+" at "+itoa(pos), got, f.heads(t, "Kcur_pos-"+itoa(il), pos), 2e-3)
			for h := 0; h < bc.KVHeads; h++ {
				widen(lc.Value(pos, h), got[h*bc.HeadDim:(h+1)*bc.HeadDim])
			}
			compareRelative(t, "Vcur_normed-"+itoa(il)+" at "+itoa(pos), got, f.heads(t, "Vcur_normed-"+itoa(il), pos), 2e-3)
		}
	}
}

func TestAssistantConfig12B(t *testing.T) {
	_, a := openAssistant12B(t)
	if a.Cfg.Dim != 1024 || len(a.Cfg.Blocks) != 4 {
		t.Fatalf("%d blocks of %d, expected 4 of 1024", len(a.Cfg.Blocks), a.Cfg.Dim)
	}
	for i, b := range a.Cfg.Blocks {
		want := 46
		if !b.Window {
			want = 47
		}
		if b.Window != (i < 3) || b.KVSource != want || !b.Behind || b.OwnsKV {
			t.Fatalf("block %d: window %v, reads %d, behind %v, owns %v", i, b.Window, b.KVSource, b.Behind, b.OwnsKV)
		}
	}
}

// One draft, every waypoint llama.cpp names, each block driven by the
// reference's own input.
func TestAssistantWaypoints12B(t *testing.T) {
	for _, name := range []string{"assistant12", "assistant12chat"} {
		t.Run(name, func(t *testing.T) {
			f := loadDraftFixture(t, name)
			m, a := openAssistant12B(t)
			fillFromReference(t, f, m)

			token, pos := f.Greedy[0], len(f.Tokens)
			a.project(token, f.tensor(t, "target_h"))
			compareRelative(t, "inp_xh", a.xh.F[0], f.tensor(t, "inp_xh"), 1e-6)
			compareRelative(t, "pre_proj", a.pre, f.tensor(t, "pre_proj"), 2e-3)

			for i, bc := range a.Cfg.Blocks {
				in := "pre_proj"
				if i > 0 {
					in = "out_scaled-" + itoa(i-1)
				}
				suffix := "-" + itoa(i)
				bw := &a.W.Blocks[i]

				// The attention alone first. Block reuses the scratch batch
				// the heads are mixed into — the global block's heads and the
				// feed forward are both 8192 wide — so kqv_out can only be
				// read before the feed forward runs.
				normed := nn.NewBatch(a.Cfg.Dim, 1)
				copy(normed.F[0], f.tensor(t, in))
				nn.RMSNormPlain(normed.F[0], bw.AttnNorm, a.Cfg.Eps)
				compareRelative(t, "attn_norm"+suffix, normed.F[0], f.tensor(t, "attn_norm"+suffix), 2e-3)
				normed.QuantizeColumnRange(0, 0, a.Cfg.Dim)
				a.view.Layers[i] = m.cache.Layers[bc.KVSource]
				at := []Place{{Cache: a.view, Pos: pos, Until: pos}}
				freqs := a.W.RoPEFreqs
				if bc.Window {
					freqs = nil
				}
				Attention(a.Cfg, bc, bw, a.scratch.RoPE(bc, at, freqs), at, a.scratch, normed, rows(1, a.Cfg.Dim))
				compareRelative(t, "Qcur_pos"+suffix, a.scratch.Q(bc), f.heads(t, "Qcur_pos"+suffix, 0), 2e-3)
				compareRelative(t, "kqv_out"+suffix, a.scratch.Heads(bc), f.tensor(t, "kqv_out"+suffix), 2e-3)
				// The scores, which ggml records as [cells, tokens, heads]:
				// the cells of a fresh cache are its positions.
				kq := f.tensor(t, "kq"+suffix)
				cells := int(f.Tensors["kq"+suffix].NE[0])
				lc := a.view.Layers[i]
				perKV := bc.Heads / bc.KVHeads
				var got, want []float32
				for h := 0; h < bc.Heads; h++ {
					query := a.scratch.qh[0][h*bc.HeadDim : (h+1)*bc.HeadDim]
					for p := 0; p < pos; p++ {
						got = append(got, nn.DotF32Half(query, lc.Key(p, h/perKV)))
						want = append(want, kq[h*cells+p])
					}
				}
				compareRelative(t, "kq"+suffix, got, want, 2e-3)

				// Then the whole block, as Draft runs it.
				copy(a.xs[0], f.tensor(t, in))
				a.block(i, pos)
				compareRelative(t, "attn_out"+suffix, a.scratch.AttnOut(), f.tensor(t, "attn_out"+suffix), 2e-3)
				compareRelative(t, "out_scaled"+suffix, a.xs[0], f.tensor(t, "out_scaled"+suffix), 2e-3)
			}

			copy(a.xs[0], f.tensor(t, "out_scaled-"+itoa(len(a.Cfg.Blocks)-1)))
			logits := make([]float32, a.Cfg.Vocab)
			a.score(logits)
			compareRelative(t, "result_norm", a.hidden, f.tensor(t, "result_norm"), 2e-3)
			compareRelative(t, "result_output", logits, f.tensor(t, "result_output"), 2e-3)
			if got := Argmax(logits); got != f.Argmax {
				t.Fatalf("the assistant guesses %d, the reference guessed %d", got, f.Argmax)
			}
			// The state a second guess is fed: the normed state taken back to
			// the target's width.
			a.ownState()
			compareRelative(t, "h_nextn", a.own, f.tensor(t, "h_nextn"), 2e-3)
		})
	}
}

// The greedy run, the target's cache now the engine's own: at every step the
// assistant's guess against the reference's. The reference's token is fed back
// either way, so one tie does not derail the rest.
func TestAssistantDrafts12B(t *testing.T) {
	for _, name := range []string{"assistant12", "assistant12chat"} {
		t.Run(name, func(t *testing.T) {
			f := loadDraftFixture(t, name)
			m, a := openAssistant12B(t)
			replayDrafts(t, f, m, a)
		})
	}
}

// The same on the card: the model's blocks and caches there, and the
// assistant's stack reading them where they are.
func TestAssistantVulkanDrafts12B(t *testing.T) {
	for _, name := range []string{"assistant12", "assistant12chat"} {
		t.Run(name, func(t *testing.T) {
			f := loadDraftFixture(t, name)
			m, a := openAssistant12B(t)
			if err := m.UseVulkanHead(); err != nil {
				t.Skipf("no Vulkan head: %v", err)
			}
			if err := m.UseVulkanStack(); err != nil {
				t.Skipf("no Vulkan stack: %v", err)
			}
			if a.stack == nil || !m.Speculate() {
				t.Fatal("the assistant did not follow the model onto the card")
			}
			replayDrafts(t, f, m, a)
		})
	}
}

func replayDrafts(t *testing.T, f *draftFixture, m *Model, a *Assistant) {
	t.Helper()
	hidden := append([]float32(nil), m.ForwardBatch(f.Tokens, 0)[len(f.Tokens)-1]...)
	pos := len(f.Tokens)
	// A guess inside this margin of the reference's is a tie, the way
	// TestGreedy12BMatchesTheReference counts one.
	const tie = 1.0
	logits := make([]float32, a.Cfg.Vocab)
	kept := 0
	for step, token := range f.Greedy {
		a.Draft(token, hidden, pos, logits)
		want := f.Drafts[step]
		if got := Argmax(logits); got != want {
			if margin := logits[got] - logits[want]; margin > tie {
				t.Fatalf("step %d: guessed %d over the reference's %d by %v, past a tie", step, got, want, margin)
			} else {
				t.Logf("step %d: guessed %d over %d by %v, a tie", step, got, want, margin)
			}
		}
		if step+1 < len(f.Greedy) && want == f.Greedy[step+1] {
			kept++
		}
		// The second guess, from the reference's first so that one tie does
		// not carry into the next comparison.
		a.DraftNext(want, pos, logits)
		if want2 := f.Drafts2[step]; Argmax(logits) != want2 {
			got := Argmax(logits)
			if margin := logits[got] - logits[want2]; margin > tie {
				t.Fatalf("step %d: second guess %d over the reference's %d by %v, past a tie", step, got, want2, margin)
			} else {
				t.Logf("step %d: second guess %d over %d by %v, a tie", step, got, want2, margin)
			}
		}
		copy(hidden, m.Forward(token, pos))
		pos++
	}
	t.Logf("the reference's drafts would have been kept %d times in %d", kept, len(f.Greedy)-1)
}
