package qwen35

import "testing"

// A picture cut across two passes.
//
// Boundary lets a pass end anywhere, so the second pass gets a span whose
// start is negative and whose count reaches past the slice. What the two
// passes place together has to be what one pass over the whole prompt places,
// because a row's axes are the picture's and not the pass's.
func TestPlacesSpanningACut(t *testing.T) {
	// Two text tokens, then a 3x4 picture of twelve rows, then one more.
	p := &Prompt{
		Tokens: make([]int32, 15),
		Embeds: make([][]float32, 15),
		Images: []ImageSpan{{Start: 2, Count: 12, Cols: 3, Rows: 4}},
	}

	whole := p.Places(0, 0)
	for cut := 1; cut < len(p.Tokens); cut++ {
		got := append([]Place(nil), p.Slice(0, cut).Places(0, 0)...)
		got = append(got, p.Slice(cut, len(p.Tokens)).Places(0, cut)...)
		if len(got) != len(whole) {
			t.Fatalf("cut %d: %d places against %d", cut, len(got), len(whole))
		}
		for i := range whole {
			if got[i] != whole[i] {
				t.Errorf("cut %d, position %d: %+v against %+v", cut, i, got[i], whole[i])
			}
		}
	}
}
