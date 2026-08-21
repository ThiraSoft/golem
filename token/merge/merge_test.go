package merge

import "testing"

// The loop's contract, without a vocabulary: what it merges, in what order,
// and what it gives back when it can merge nothing.
func TestApply(t *testing.T) {
	cases := []struct {
		name    string
		symbols []string
		ranks   map[Pair]int
		want    []string
	}{
		{
			name:    "empty",
			symbols: nil,
			ranks:   map[Pair]int{},
			want:    []string{},
		},
		{
			name:    "one symbol is returned whatever the table says",
			symbols: []string{"a"},
			ranks:   map[Pair]int{{"a", "a"}: 0},
			want:    []string{"a"},
		},
		{
			name:    "no rank, no merge",
			symbols: []string{"a", "b", "c"},
			ranks:   map[Pair]int{},
			want:    []string{"a", "b", "c"},
		},
		{
			name:    "one merge",
			symbols: []string{"a", "b", "c"},
			ranks:   map[Pair]int{{"a", "b"}: 0},
			want:    []string{"ab", "c"},
		},
		{
			// The lower rank wins even though the other pair is further left.
			name:    "rank beats position",
			symbols: []string{"a", "b", "c", "d"},
			ranks:   map[Pair]int{{"a", "b"}: 7, {"c", "d"}: 1},
			want:    []string{"ab", "cd"},
		},
		{
			// Equal ranks: the leftmost pair goes first. Here that decides the
			// outcome, because merging "ab" first destroys the "bc" pair.
			name:    "ties go to the left",
			symbols: []string{"a", "b", "c"},
			ranks:   map[Pair]int{{"a", "b"}: 3, {"b", "c"}: 3},
			want:    []string{"ab", "c"},
		},
		{
			// A pair that only exists after an earlier merge must be found.
			name:    "a merge creates the next one",
			symbols: []string{"a", "b", "c"},
			ranks:   map[Pair]int{{"a", "b"}: 0, {"ab", "c"}: 1},
			want:    []string{"abc"},
		},
		{
			// A queued pair whose halves have since merged elsewhere must be
			// discarded rather than applied to the wrong text.
			name:    "a stale pair is dropped",
			symbols: []string{"a", "b", "c", "d"},
			ranks: map[Pair]int{
				{"b", "c"}:  0,
				{"a", "b"}:  1,
				{"a", "bc"}: 2,
			},
			want: []string{"abc", "d"},
		},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			var m Merger
			got := m.Apply(c.symbols, c.ranks)
			if len(got) != len(c.want) {
				t.Fatalf("Apply = %q, want %q", got, c.want)
			}
			for i := range got {
				if got[i] != c.want[i] {
					t.Fatalf("Apply = %q, want %q", got, c.want)
				}
			}
		})
	}
}

// A Merger is reused across runs, and the buffers it keeps must not let one
// run's answer leak into the next.
func TestMergerIsReusable(t *testing.T) {
	ranks := map[Pair]int{{"a", "b"}: 0}
	var m Merger

	first := m.Apply([]string{"a", "b", "c", "d", "e"}, ranks)
	if len(first) != 4 {
		t.Fatalf("first = %q, want four symbols", first)
	}
	second := m.Apply([]string{"x", "y"}, ranks)
	if len(second) != 2 || second[0] != "x" || second[1] != "y" {
		t.Fatalf("second = %q, want [x y]", second)
	}
}
