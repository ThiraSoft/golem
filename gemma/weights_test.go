package gemma

import (
	"testing"

	"github.com/ThiraSoft/golem/nn"
	"github.com/ThiraSoft/golem/tensors"
)

func TestLoadWeightsE2B(t *testing.T) {
	g := openModel(t)
	cfg, err := LoadConfig(g, 4096)
	if err != nil {
		t.Fatal(err)
	}
	w, err := LoadWeights(g, cfg)
	if err != nil {
		t.Fatal(err)
	}

	if w.TokenEmbd.Quant != nn.Q6_K || w.TokenEmbd.Rows != 262144 || w.TokenEmbd.Cols != 1536 {
		t.Fatalf("token embedding: %+v", w.TokenEmbd)
	}
	if w.PLEEmbd.Quant != nn.Q6_K || w.PLEEmbd.Rows != 262144 || w.PLEEmbd.Cols != 8960 {
		t.Fatalf("per-layer embedding: %+v", w.PLEEmbd)
	}
	if w.PLEProj.Quant != nn.BF16 || w.PLEProj.Rows != 8960 || w.PLEProj.Cols != 1536 {
		t.Fatalf("per-layer projection: %+v", w.PLEProj)
	}
	if len(w.PLEProjNorm) != 256 || len(w.OutputNorm) != 1536 {
		t.Fatalf("norms of %d and %d", len(w.PLEProjNorm), len(w.OutputNorm))
	}

	// The frequency factors of the global blocks: sixty-four rotated pairs,
	// then a hundred and ninety-two the conversion switched off with 1e30.
	if len(w.RoPEFreqs) != 256 {
		t.Fatalf("%d frequency factors", len(w.RoPEFreqs))
	}
	if w.RoPEFreqs[0] != 1 || w.RoPEFreqs[63] != 1 {
		t.Fatalf("the first pairs should rotate: %v %v", w.RoPEFreqs[0], w.RoPEFreqs[63])
	}
	if w.RoPEFreqs[64] < 1e29 || w.RoPEFreqs[255] < 1e29 {
		t.Fatalf("the last pairs should be switched off: %v %v", w.RoPEFreqs[64], w.RoPEFreqs[255])
	}

	// A window block that owns its cache.
	b := w.Blocks[0]
	if b.Q.Rows != 2048 || b.Q.Cols != 1536 || b.Q.Quant != nn.Q4_0 {
		t.Fatalf("block 0 query: %+v", b.Q)
	}
	if b.K.Rows != 256 || b.V.Rows != 256 || b.O.Rows != 1536 || b.O.Cols != 2048 {
		t.Fatalf("block 0 attention: k %+v v %+v o %+v", b.K, b.V, b.O)
	}
	if len(b.QNorm) != 256 || len(b.KNorm) != 256 {
		t.Fatalf("block 0 head norms: %d %d", len(b.QNorm), len(b.KNorm))
	}
	if b.Gate.Rows != 6144 || b.Down.Rows != 1536 || b.Down.Cols != 6144 {
		t.Fatalf("block 0 feed forward: gate %+v down %+v", b.Gate, b.Down)
	}
	if b.InpGate.Rows != 256 || b.InpGate.Cols != 1536 || b.Proj.Rows != 1536 || b.Proj.Cols != 256 {
		t.Fatalf("block 0 per-layer path: gate %+v proj %+v", b.InpGate, b.Proj)
	}
	if b.OutScale == 0 {
		t.Fatal("block 0 has no output scale")
	}

	// A global block: wider heads, so wider projections.
	if w.Blocks[4].Q.Rows != 4096 || w.Blocks[4].K.Rows != 512 || len(w.Blocks[4].QNorm) != 512 {
		t.Fatalf("block 4: %+v", w.Blocks[4])
	}

	// A block that shares: a query and an output, and nothing else.
	s := w.Blocks[15]
	if s.Q.Rows != 2048 {
		t.Fatalf("block 15 query: %+v", s.Q)
	}
	if s.K.Data != nil || s.V.Data != nil || s.KNorm != nil {
		t.Fatal("block 15 must carry no key or value of its own")
	}
	if s.Gate.Rows != 12288 {
		t.Fatalf("block 15 feed forward %d wide", s.Gate.Rows)
	}
}

func TestLoadWeightsRejectsAMissingTensor(t *testing.T) {
	g := openModel(t)
	cfg, err := LoadConfig(g, 4096)
	if err != nil {
		t.Fatal(err)
	}
	// A block that owns its cache but whose key projection is not in the file
	// is a file we do not understand, not a block to guess about.
	delete(g.Tensors, "blk.3.attn_k.weight")
	if _, err := LoadWeights(g, cfg); err == nil {
		t.Fatal("a missing key projection should be an error")
	}
}

// TestExpertStackIsAView pins that slicing one expert out of the
// three-dimensional tensor costs no copy and lands on the right bytes.
func TestExpertStackIsAView(t *testing.T) {
	e := ExpertStack{
		Data:  make([]byte, 3*2*18), // three experts, two rows, one Q4_0 block each
		Quant: nn.Q4_0,
		Rows:  2,
		Cols:  32,
		Count: 3,
	}
	for i := range e.Data {
		e.Data[i] = byte(i)
	}
	m := e.At(1)
	if m.Rows != 2 || m.Cols != 32 || m.Quant != nn.Q4_0 {
		t.Fatalf("the view is %dx%d %s", m.Rows, m.Cols, m.Quant)
	}
	if len(m.Data) != 36 {
		t.Fatalf("the view is %d bytes, expected 36", len(m.Data))
	}
	if &m.Data[0] != &e.Data[36] {
		t.Fatal("the view does not begin where the second expert does")
	}
	if &e.At(2).Data[35] != &e.Data[107] {
		t.Fatal("the last expert does not end at the last byte")
	}
}

// TestLoad26BExpertShapes checks the expert tensors against the geometry. A
// stride read from the wrong dimension is the failure this catches, and it is
// one that produces numbers rather than an error.
func TestLoad26BExpertShapes(t *testing.T) {
	g, err := tensors.OpenGGUF(model26BPath(t))
	if err != nil {
		t.Fatal(err)
	}
	defer g.Close()
	cfg, err := LoadConfig(g, 4096)
	if err != nil {
		t.Fatal(err)
	}
	w, err := LoadWeights(g, cfg)
	if err != nil {
		t.Fatal(err)
	}
	for i, bc := range cfg.Blocks {
		b := &w.Blocks[i]
		if !bc.MoE {
			if b.Router.Rows != 0 {
				t.Fatalf("block %d is dense and carries a router", i)
			}
			continue
		}
		if b.Router.Rows != cfg.Experts || b.Router.Cols != cfg.Dim {
			t.Fatalf("block %d: router is %dx%d, expected %dx%d",
				i, b.Router.Rows, b.Router.Cols, cfg.Experts, cfg.Dim)
		}
		if len(b.RouterScale) != cfg.Dim {
			t.Fatalf("block %d: the router scale has %d entries for a width of %d",
				i, len(b.RouterScale), cfg.Dim)
		}
		if b.GateUpExps.Count != cfg.Experts || b.GateUpExps.Rows != 2*cfg.ExpertFFN || b.GateUpExps.Cols != cfg.Dim {
			t.Fatalf("block %d: gate-and-up experts are %d of %dx%d, expected %d of %dx%d",
				i, b.GateUpExps.Count, b.GateUpExps.Rows, b.GateUpExps.Cols, cfg.Experts, 2*cfg.ExpertFFN, cfg.Dim)
		}
		if len(b.DownScale) != cfg.Experts {
			t.Fatalf("block %d: the down scale has %d entries for %d experts",
				i, len(b.DownScale), cfg.Experts)
		}
		if b.DownExps.Rows != cfg.Dim || b.DownExps.Cols != cfg.ExpertFFN {
			t.Fatalf("block %d: down experts are %dx%d, expected %dx%d",
				i, b.DownExps.Rows, b.DownExps.Cols, cfg.Dim, cfg.ExpertFFN)
		}
	}
}
