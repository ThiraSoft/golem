package bpe

import (
	"slices"
	"strings"
	"testing"
)

func TestSplitMetaspace(t *testing.T) {
	s := Space
	cases := map[string][]string{
		"a":                {s + "a"},
		s + "a" + s + "b":  {s + "a", s + "b"},
		"a" + s + s + "b":  {s + "a", s, s + "b"},
		"a" + s:            {s + "a", s},
		s + s:              {s, s},
		"x\ny" + s + "z\t": {s + "x\ny", s + "z\t"},
	}
	for text, want := range cases {
		got := splitMetaspace(text)
		if !slices.Equal(got, want) {
			t.Errorf("%q: %q, want %q", text, got, want)
		}
		joined := strings.Join(got, "")
		if joined != text && joined != s+text {
			t.Errorf("%q: the words do not concatenate back", text)
		}
	}
}
