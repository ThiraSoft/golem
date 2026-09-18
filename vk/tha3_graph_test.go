package vk

import "testing"

// overlapFree checks the planner's one promise: two tensors alive at the same
// time never share a float of the arena.
func overlapFree(t *testing.T, g *THA3Graph) {
	t.Helper()
	first, last, _ := g.liveness()
	for a := range g.tensors {
		for b := a + 1; b < len(g.tensors); b++ {
			if first[a] == -2 || first[b] == -2 {
				continue // never used, never placed
			}
			if last[a] < first[b] || last[b] < first[a] {
				continue // never alive together
			}
			ra, rb := g.tensors[a], g.tensors[b]
			if ra.off < rb.off+alignFloats(rb.floats) && rb.off < ra.off+alignFloats(ra.floats) {
				t.Fatalf("tensors %d [%d,+%d) and %d [%d,+%d) overlap while both alive",
					a, ra.off, ra.floats, b, rb.off, rb.floats)
			}
		}
	}
}

// chain adds one op reading from and creating a tensor of from's size.
func chain(g *THA3Graph, from THA3Tensor) THA3Tensor {
	out := g.tensor(from.C, from.H, from.W)
	g.op([]THA3Tensor{from}, []THA3Tensor{out}, nil)
	return out
}

func TestTHA3PlanReuses(t *testing.T) {
	g := NewTHA3Graph()
	g.SetPhase(THA3PosePhase)
	x := g.Input(1, 8, 8)
	a := chain(g, x)
	b := chain(g, a)
	c := chain(g, b)
	g.op([]THA3Tensor{c}, nil, nil)
	total := g.plan()
	if g.tensors[c.id].off != g.tensors[a.id].off {
		t.Fatalf("c at %d did not take a's place at %d, freed one op earlier", g.tensors[c.id].off, g.tensors[a.id].off)
	}
	if total != 3*64 {
		t.Fatalf("arena %d floats, want %d: input, and two slots taking turns", total, 3*64)
	}
	overlapFree(t, g)
}

func TestTHA3PlanPinsAcrossPhases(t *testing.T) {
	g := NewTHA3Graph()
	x := g.Input(1, 8, 8)
	kept := chain(g, x)    // made by the image pass
	dropped := chain(g, x) // used by the image pass only
	g.op([]THA3Tensor{dropped}, nil, nil)
	g.SetPhase(THA3PosePhase)
	for i := 0; i < 4; i++ {
		chain(g, x)
	}
	g.op([]THA3Tensor{kept}, nil, nil)
	g.plan()
	_, _, pinned := g.liveness()
	if !pinned[kept.id] || pinned[dropped.id] {
		t.Fatalf("pinned kept=%v dropped=%v, want true and false", pinned[kept.id], pinned[dropped.id])
	}
	overlapFree(t, g)
}

func TestTHA3PlanNoOverlap(t *testing.T) {
	g := NewTHA3Graph()
	x := g.Input(3, 16, 16)
	g.SetPhase(THA3PosePhase)
	var live []THA3Tensor
	for i := 0; i < 40; i++ {
		src := x
		if len(live) > 0 {
			src = live[(i*7)%len(live)]
		}
		c := 1 + i%5
		out := g.tensor(c, 4+i%9, 4+i%11)
		g.op([]THA3Tensor{src, src.Channels(0, 1)}, []THA3Tensor{out}, nil)
		live = append(live, out)
		if len(live) > 6 {
			live = live[2:]
		}
	}
	g.plan()
	overlapFree(t, g)
}

func TestTHA3ChannelsOffset(t *testing.T) {
	g := NewTHA3Graph()
	x := g.Input(4, 2, 3)
	g.plan()
	v := x.Channels(1, 3)
	if got, want := g.offset(v), g.offset(x)+6; got != want {
		t.Fatalf("view at %d, want %d", got, want)
	}
	if v.C != 2 {
		t.Fatalf("view has %d channels, want 2", v.C)
	}
}

func TestTHA3PhasesGoForward(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Fatal("going back to the image phase did not panic")
		}
	}()
	g := NewTHA3Graph()
	g.SetPhase(THA3PosePhase)
	g.SetPhase(THA3ImagePhase)
}

func TestTHA3PlanPinsAcrossEntries(t *testing.T) {
	g := NewTHA3Graph()
	g.SetPhase(THA3PosePhase)
	x := g.Input(1, 8, 8)
	kept := chain(g, x)    // read on both sides of the entry
	dropped := chain(g, x) // read before it only
	g.op([]THA3Tensor{dropped}, nil, nil)
	if e := g.Entry(); e != 1 {
		t.Fatalf("first entry is %d, want 1", e)
	}
	after := chain(g, x) // made and read after it
	for i := 0; i < 4; i++ {
		chain(g, after)
	}
	g.op([]THA3Tensor{kept}, nil, nil)
	g.plan()
	_, _, pinned := g.liveness()
	if !pinned[kept.id] || pinned[dropped.id] || pinned[after.id] {
		t.Fatalf("pinned kept=%v dropped=%v after=%v, want true, false, false",
			pinned[kept.id], pinned[dropped.id], pinned[after.id])
	}
	overlapFree(t, g)
}
