package qwen35

import (
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"
	"time"

	"github.com/ThiraSoft/golem/chat"
	"github.com/ThiraSoft/golem/nn"
	"github.com/ThiraSoft/golem/tensors"
	"github.com/ThiraSoft/golem/token/bytebpe"
)

// golemPath is the .golem checkpoint these tests read, and they say nothing
// without one: the file is eleven gigabytes and is made by cmd/golemquant, so
// it is not something a checkout carries.
func golemPath(t *testing.T) string {
	t.Helper()
	path := os.Getenv("GOLEM_MODEL_QWEN35_GOLEM")
	if path == "" {
		t.Skip("GOLEM_MODEL_QWEN35_GOLEM unset")
	}
	return path
}

// The whole trunk of a .golem model on the card, held to the same model on the
// processor.
//
// Every projection in this architecture changes form at once — forty-eight
// delta nets with five each, seventeen attentions with four, sixty-five feed
// forwards with three, and the head — and what sits between them does not
// change at all. So this asserts the two things that say the swap was done
// right: the hidden state stays inside the gap two arithmetics leave, and both
// paths name the same token.
func TestVulkanD4GMatchesCPU(t *testing.T) {
	path := golemPath(t)
	m, err := Open(path, 512)
	if err != nil {
		t.Skipf("open: %v", err)
	}
	defer m.Close()
	if m.W.Blocks[0].Down.Quant.D4Width() == 0 {
		t.Fatalf("%s is not a .golem checkpoint: its blocks are %s", path, m.W.Blocks[0].Down.Quant)
	}

	toks := []int32{100, 200, 300, 400}
	cpu := make([][]float32, len(toks))
	cpuTop := make([]int32, len(toks))
	logits := make([]float32, m.Cfg.Vocab)
	for i, tok := range toks {
		cpu[i] = append([]float32(nil), m.Forward(tok, i)...)
		m.Logits(cpu[i], logits)
		cpuTop[i] = argmax(logits)
	}

	t0 := time.Now()
	if err := m.UseVulkanStack(); err != nil {
		t.Fatalf("vulkan stack: %v", err)
	}
	if err := m.UseVulkanHead(); err != nil {
		t.Fatalf("vulkan head: %v", err)
	}
	fmt.Printf("uploaded in %v\n", time.Since(t0).Round(time.Millisecond))
	m.Reset()

	for i, tok := range toks {
		g := m.Forward(tok, i)
		maxAbs, rel := diff(cpu[i], g)
		m.Logits(g, logits)
		top := argmax(logits)
		fmt.Printf("pos %d: max|d|=%.4g  rel=%.4g  token cpu=%d gpu=%d\n", i, maxAbs, rel, cpuTop[i], top)
		if hasNaN(g) >= 0 {
			t.Fatalf("position %d has a hidden state that is not a number", i)
		}
		if rel > 0.1 {
			t.Errorf("position %d diverges far past the quantisation gap: relative error %.4g", i, rel)
		}
		if top != cpuTop[i] {
			t.Errorf("position %d: the card draws %d where the processor draws %d", i, top, cpuTop[i])
		}
		_ = tok
	}
}

// Block by block, which is what says where a divergence begins rather than
// that there is one. Probe stops the card after n blocks and hands back the
// stream; the processor is run to the same point.
func TestVulkanD4GWaypoints(t *testing.T) {
	path := golemPath(t)
	m, err := Open(path, 512)
	if err != nil {
		t.Skipf("open: %v", err)
	}
	defer m.Close()

	const tok = 3000
	emb := make([]float32, m.Cfg.Dim)
	m.W.TokenEmbd.Row(tok, emb)

	// Where a block's mixer lands, for the first of each kind and a few after
	// them: a delta net and a full attention answer through different
	// projections, and this says which of the two went wrong.
	kinds := []int{}
	for i, bc := range m.Cfg.Blocks[:m.trunk()] {
		if len(kinds) < 6 || bc.Type == BlockFullAttn && len(kinds) < 8 {
			kinds = append(kinds, i)
		}
	}
	wantMix := map[int][]float32{}
	for _, b := range kinds {
		m.Reset()
		_, mix := m.cpuMixer(tok, 0, b)
		wantMix[b] = mix
	}
	want := map[int][]float32{}
	for _, n := range []int{1, 2, 3, 4, 8, 16, m.trunk()} {
		m.Reset()
		want[n] = m.cpuTrunk(tok, 0, n)
	}

	if err := m.UseVulkanStack(); err != nil {
		t.Fatalf("vulkan stack: %v", err)
	}
	for _, b := range kinds {
		m.Reset()
		got, err := m.gpuPipe.ProbeMixer(emb, 0, b)
		if err != nil {
			t.Fatalf("block %d mixer: %v", b, err)
		}
		ma, rel := diff(wantMix[b], got)
		kind := "delta net"
		if m.Cfg.Blocks[b].Type == BlockFullAttn {
			kind = "attention"
		}
		fmt.Printf("block %2d %-9s mixer: max=%.4g rel=%.4g\n", b, kind, ma, rel)
		if rel > 0.05 {
			t.Errorf("block %d's %s answers %.4g away from the processor's", b, kind, rel)
		}
	}
	for _, n := range []int{1, 2, 3, 4, 8, 16, m.trunk()} {
		m.Reset()
		got, err := m.gpuPipe.Probe(emb, 0, n)
		if err != nil {
			t.Fatalf("%d blocks: %v", n, err)
		}
		ma, rel := diff(want[n], got)
		fmt.Printf("after %2d blocks: max=%.4g rel=%.4g\n", n, ma, rel)
		if rel > 0.1 {
			t.Errorf("the stream after %d blocks is %.4g away from the processor's", n, rel)
		}
	}
}

// A pass carrying several columns has to answer what that many passes of one
// answer. The D4G products are dispatched in runs of the widest binary the
// kernels were built for, with the rest at narrower widths, and every one of
// them is told where its columns start — an offset that is easy to write and
// impossible to see in a pass of one, which is what the waypoints are.
func TestVulkanD4GWidePassMatchesTokenPath(t *testing.T) {
	path := golemPath(t)
	m, err := Open(path, 1024)
	if err != nil {
		t.Skipf("open: %v", err)
	}
	defer m.Close()
	if err := m.UseVulkanStack(); err != nil {
		t.Skipf("vulkan stack: %v", err)
	}

	for _, w := range []int{2, 3, 4, 8, 16, 32} {
		if w > m.gpuPipe.Columns() {
			continue
		}
		xs := make([][]float32, w)
		pos := make([]int, w)
		for i := range xs {
			xs[i] = make([]float32, m.Cfg.Dim)
			m.W.TokenEmbd.Row(1000+i*7, xs[i])
			pos[i] = i
		}

		m.gpuPipe.ResetState()
		hs, err := m.gpuPipe.ForwardColumns(xs, pos)
		if err != nil {
			t.Fatal(err)
		}
		wide := append([]float32(nil), hs[w-1]...)

		m.gpuPipe.ResetState()
		var narrow []float32
		for i := range xs {
			h, err := m.gpuPipe.ForwardColumns(xs[i:i+1], pos[i:i+1])
			if err != nil {
				t.Fatal(err)
			}
			narrow = append([]float32(nil), h[0]...)
		}
		maxAbs, rel := diff(wide, narrow)
		fmt.Printf("%3d columns: max|d|=%.4g rel=%.4g\n", w, maxAbs, rel)
		if rel > 1e-3 {
			t.Errorf("a pass of %d columns answers %.4g away from %d passes of one", w, rel, w)
		}
	}
}

// The prediction block, which is the one path a pass of one column never
// reaches: the command line drafts with it whenever the card holds it, and a
// draft that is nonsense is a generation that is nonsense.
//
// Two things are asked. How often the draft is the token the model itself goes
// on to choose — nonsense drafts agree with nothing — and whether the pass that
// verifies a draft answers, for the column that was already committed, what a
// plain pass answers for it. The second is the half that can corrupt the state
// rather than merely waste a draft.
func TestVulkanD4GPredictionBlock(t *testing.T) {
	path := golemPath(t)
	m, err := Open(path, 1024)
	if err != nil {
		t.Skipf("open: %v", err)
	}
	defer m.Close()
	if err := m.UseVulkan(); err != nil {
		t.Skipf("vulkan: %v", err)
	}
	if !m.HasMTP() {
		t.Skip("the checkpoint carries no prediction block")
	}
	if !m.Speculate() {
		t.Fatal("the prediction block is in the file and not on the card")
	}

	toks := []int32{9707, 11, 847, 829, 374, 264, 1273, 315, 279, 1614}
	hs := m.ForwardBatch(toks, 0)
	hidden := hs[len(hs)-1]
	logits := make([]float32, m.Cfg.Vocab)
	draft := make([]float32, m.Cfg.Vocab)

	pos := len(toks)
	m.Logits(hidden, logits)
	id := argmax(logits)

	accepted, tried := 0, 0
	for n := 0; n < 24; n++ {
		m.ForwardMTP(id, hidden, pos, draft)
		guess := argmax(draft)
		hidden = m.Forward(id, pos)
		pos++
		m.Logits(hidden, logits)
		truth := argmax(logits)
		tried++
		if guess == truth {
			accepted++
		}
		id = truth
	}
	rate := 100 * float64(accepted) / float64(tried)
	fmt.Printf("draft accepted %d/%d (%.1f%%)\n", accepted, tried, rate)
	// Not a quality bar — what the block is worth is its own measurement. This
	// is the difference between a block that predicts and one that answers
	// noise, and noise agrees with a 248320-token vocabulary approximately
	// never.
	if rate < 20 {
		t.Errorf("the prediction block agrees with the model %.1f%% of the time, which is what a broken one looks like", rate)
	}
}

// The goal this branch was opened for: a picture, through a .golem model, on
// the card.
//
// The tower is not the model. Its weights are fp16 in a projector file that no
// compression touches, and vk/vision.go reads them as it always did — so what
// is being asked here is not whether the tower works but whether the rows it
// makes reach a compressed trunk at the right positions and mean anything
// there. engine/media.go puts the tower on the card when the blocks are on it,
// which they now can be, so a .golem model sees on the card exactly as a Q4_0
// one does.
func TestVulkanD4GDescribesAnImage(t *testing.T) {
	path := golemPath(t)
	g, err := tensors.OpenGGUF(path)
	if err != nil {
		t.Skipf("open: %v", err)
	}
	vocab, err := bytebpe.Load(g)
	if err != nil {
		t.Fatal(err)
	}
	m, err := New(g, 1024)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	if start, ok := vocab.ID(VisionStart); ok {
		pad, _ := vocab.ID(ImagePad)
		end, _ := vocab.ID(VisionEnd)
		m.SetVisionMarkers(start, pad, end)
	}
	if err := m.OpenProjector(qwen38mmproj); err != nil {
		t.Skipf("projector: %v", err)
	}
	if err := m.UseVulkan(); err != nil {
		t.Fatalf("vulkan: %v", err)
	}
	on, resident := m.VisionVulkan()
	t.Logf("tower on the card: %v (resident %v)", on, resident)
	if !on {
		t.Error("the trunk is on the card and the tower is not, which is the one thing this was for")
	}

	raw, err := os.ReadFile(filepath.Join("..", "testdata", "gemma", "shapes.png"))
	if err != nil {
		t.Skipf("image: %v", err)
	}
	t0 := time.Now()
	rows, err := m.EncodeImage(raw)
	if err != nil {
		t.Fatal(err)
	}
	tower := time.Since(t0)
	grid, _ := m.GridOf(0)

	tpl := NewTemplate()
	rendered, err := tpl.Render([]chat.Message{{
		Role: "user", Content: "Describe this image in one sentence.", Images: [][]byte{raw},
	}}, chat.Options{AddGenerationPrompt: true})
	if err != nil {
		t.Fatal(err)
	}
	ids := vocab.Encode(rendered, false, true)

	p, err := m.BuildPrompt(ids, [][][]float32{rows}, [][2]int{grid})
	if err != nil {
		t.Fatal(err)
	}
	if len(p.Images) != 1 || p.Images[0].Count != len(rows) {
		t.Fatalf("%d pictures placed, %d rows of %d", len(p.Images), p.Images[0].Count, len(rows))
	}

	t1 := time.Now()
	states := m.ForwardPrompt(p, 0)
	hidden := states[len(states)-1]
	prefill := time.Since(t1)

	logits := make([]float32, m.Cfg.Vocab)
	var out strings.Builder
	pos := len(p.Tokens)
	t2 := time.Now()
	drawn := 0
	for i := 0; i < 32; i++ {
		m.Logits(hidden, logits)
		id := argmax(logits)
		if vocab.IsEOG(id) {
			break
		}
		out.WriteString(vocab.Piece(id, false))
		hidden = m.ForwardBatch([]int32{id}, pos)[0]
		pos++
		drawn++
	}
	draw := time.Since(t2)

	text := strings.TrimSpace(out.String())
	t.Logf("%d positions, %d of them the picture at %d", len(p.Tokens), p.Images[0].Count, p.Images[0].Start)
	t.Logf("tower %v, prefill %v, %d tokens in %v (%.1f/s)",
		tower.Round(time.Millisecond), prefill.Round(time.Millisecond),
		drawn, draw.Round(time.Millisecond), float64(drawn)/draw.Seconds())
	t.Logf("answer: %q", text)
	if text == "" {
		t.Fatal("the model said nothing about the picture")
	}
	// Not a check on what it said — that is the model's business — but on
	// whether it said anything shaped like prose. A broken splice gives
	// repeated punctuation or one token forever.
	if len([]rune(text)) < 8 {
		t.Fatalf("the answer is %q, which is not a description", text)
	}
}

// The pass that verifies a draft, which is the one the command line runs and
// no other test does.
//
// Drafting is harmless when it is wrong: the guess is thrown away. Verifying is
// not. It carries two columns where a plain pass carries one, it snapshots the
// delta nets before the drafted column so a refusal can put them back, and it
// commits the first column's token either way. A verification that answers
// differently from a plain pass is a generation that drifts from the model's
// own, silently, one token at a time.
//
// So: the same continuation drawn twice, once through the speculator and once
// a token at a time, and they have to be the same tokens.
func TestVulkanD4GSpeculationDrawsWhatTheModelDraws(t *testing.T) {
	path := golemPath(t)
	m, err := Open(path, 1024)
	if err != nil {
		t.Skipf("open: %v", err)
	}
	defer m.Close()
	if err := m.UseVulkan(); err != nil {
		t.Skipf("vulkan: %v", err)
	}
	if !m.Speculate() {
		t.Skip("this model cannot draft")
	}

	prompt := []int32{9707, 11, 847, 829, 374, 264, 1273, 315, 279, 1614}
	const want = 16
	logits := make([]float32, m.Cfg.Vocab)

	// A token at a time, which is what the model says.
	m.Reset()
	hs := m.ForwardBatch(prompt, 0)
	hidden := hs[len(hs)-1]
	m.Logits(hidden, logits)
	id := argmax(logits)
	plain := []int32{id}
	pos := len(prompt)
	for len(plain) < want {
		hidden = m.Forward(id, pos)
		pos++
		m.Logits(hidden, logits)
		id = argmax(logits)
		plain = append(plain, id)
	}

	// And through the prediction block, which draws two at a time when its
	// guess is right.
	m.Reset()
	if err := m.gpuPipe.ResetMTPCache(); err != nil {
		t.Fatal(err)
	}
	hs = m.ForwardBatch(prompt, 0)
	hidden = hs[len(hs)-1]
	m.Logits(hidden, logits)
	id = argmax(logits)
	sp, err := m.NewSpeculator()
	if err != nil {
		t.Fatal(err)
	}
	drafted := []int32{id}
	pos = len(prompt)
	for len(drafted) < want {
		got, h, err := sp.Step(id, hidden, pos, argmax)
		if err != nil {
			t.Fatal(err)
		}
		drafted = append(drafted, got...)
		hidden = h
		pos += len(got)
		id = drafted[len(drafted)-1]
	}
	drafted = drafted[:want]

	fmt.Printf("plain    %v\n", plain)
	fmt.Printf("drafted  %v\n", drafted)
	fmt.Printf("%d drafts, %d accepted\n", sp.Drafted, sp.Accepted)
	for i := range plain {
		if plain[i] != drafted[i] {
			t.Fatalf("token %d: a token at a time draws %d, the speculator draws %d", i, plain[i], drafted[i])
		}
	}
}

// Where the stream goes, block by block, in norm.
//
// Everything else here compares a .golem against itself — the card against the
// processor, a matrix against its own codes. None of that can see a model that
// is internally consistent and collectively wrong, which is what a stream that
// grows or collapses through the trunk looks like. This prints the size of it
// so that the block where it leaves the rails can be named.
func TestVulkanD4GStreamNorms(t *testing.T) {
	path := golemPath(t)
	m, err := Open(path, 512)
	if err != nil {
		t.Skipf("open: %v", err)
	}
	defer m.Close()

	const tok = 3000
	norm := func(v []float32) float64 {
		var s float64
		for _, x := range v {
			s += float64(x) * float64(x)
		}
		return math.Sqrt(s / float64(len(v)))
	}
	m.Reset()
	emb := make([]float32, m.Cfg.Dim)
	m.W.TokenEmbd.Row(tok, emb)
	fmt.Printf("embedding rms %.5g\n", norm(emb))
	for _, n := range []int{1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64} {
		m.Reset()
		x := m.cpuTrunk(tok, 0, n)
		kind := "delta net"
		if m.Cfg.Blocks[n-1].Type == BlockFullAttn {
			kind = "attention"
		}
		fmt.Printf("after %2d blocks (%s): rms %.5g  nan@%d\n", n, kind, norm(x), hasNaN(x))
	}
	logits := make([]float32, m.Cfg.Vocab)
	m.Reset()
	h := m.cpuTrunk(tok, 0, m.trunk())
	nn.RMSNormPlain(h, m.W.OutputNorm, m.Cfg.Eps)
	m.Logits(h, logits)
	type top struct {
		id int32
		v  float32
	}
	best := make([]top, 0, 5)
	for i, v := range logits {
		if len(best) < 5 || v > best[len(best)-1].v {
			best = append(best, top{int32(i), v})
			sort.Slice(best, func(a, b int) bool { return best[a].v > best[b].v })
			if len(best) > 5 {
				best = best[:5]
			}
		}
	}
	fmt.Printf("logits rms %.5g, top5 %v\n", norm(logits), best)
}

// The head on its own, which is the one part of the model a hidden state can
// be handed to twice.
//
// Everything above compares whole passes, and a whole pass has a trunk and a
// head in it. This gives the same hidden state to both heads and asks what
// each makes of it, which is the only way to say which of the two is wrong
// when a generation goes somewhere the processor does not.
func TestVulkanD4GHeadMatchesCPU(t *testing.T) {
	path := golemPath(t)
	m, err := Open(path, 512)
	if err != nil {
		t.Skipf("open: %v", err)
	}
	defer m.Close()

	prompt := []int32{9707, 11, 847, 829, 374, 264, 1273, 315, 279, 1614}
	hs := m.ForwardBatch(prompt, 0)
	hidden := append([]float32(nil), hs[len(hs)-1]...)

	cpu := make([]float32, m.Cfg.Vocab)
	m.Logits(hidden, cpu)
	cpuTop := argmax(cpu)

	// And the continuation the processor draws on its own, which is what the
	// card is being asked to reproduce.
	var plain []int32
	id, pos := cpuTop, len(prompt)
	for i := 0; i < 8; i++ {
		plain = append(plain, id)
		h := m.Forward(id, pos)
		pos++
		m.Logits(h, cpu)
		id = argmax(cpu)
	}
	m.Logits(hidden, cpu)
	fmt.Printf("processor draws %v\n", plain)

	if err := m.UseVulkanHead(); err != nil {
		t.Skipf("vulkan head: %v", err)
	}
	gpu := make([]float32, m.Cfg.Vocab)
	m.Logits(hidden, gpu)
	gpuTop := argmax(gpu)

	maxAbs, rel := diff(cpu, gpu)
	fmt.Printf("head on the same hidden state: max|d|=%.4g rel=%.4g  cpu=%d gpu=%d\n",
		maxAbs, rel, cpuTop, gpuTop)
	fmt.Printf("cpu[%d]=%.4f gpu[%d]=%.4f   cpu[%d]=%.4f gpu[%d]=%.4f\n",
		cpuTop, cpu[cpuTop], cpuTop, gpu[cpuTop], gpuTop, cpu[gpuTop], gpuTop, gpu[gpuTop])
	if rel > 0.01 {
		t.Errorf("the head on the card is %.4g away from the processor's on the same input", rel)
	}
	if cpuTop != gpuTop {
		t.Errorf("the same hidden state draws %d on the processor and %d on the card", cpuTop, gpuTop)
	}
}
