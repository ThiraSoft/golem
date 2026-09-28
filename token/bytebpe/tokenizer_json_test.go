package bytebpe

import (
	"slices"
	"strings"
	"testing"
)

func TestSplitGPT2(t *testing.T) {
	cases := map[string][]string{
		"Hello world": {"Hello", " world"},
		"a   b":       {"a", "  ", " b"},
		"a   ":        {"a", "   "},
		"it's we're":  {"it", "'s", " we", "'re"},
		"IT'S":        {"IT", "'", "S"},
		",b":          {",", "b"},
		"x 1234 y":    {"x", " 1234", " y"},
		"tab\tx":      {"tab", "\t", "x"},
		"a\n\nb":      {"a", "\n", "\n", "b"},
		" 'quoted'":   {" '", "quoted", "'"},
		"{\"k\": 42}": {"{\"", "k", "\":", " 42", "}"},
		"café 日本語 😀":  {"café", " 日本語", " 😀"},
	}
	for text, want := range cases {
		got := splitGPT2(text)
		if !slices.Equal(got, want) {
			t.Errorf("%q: %q, want %q", text, got, want)
		}
		if strings.Join(got, "") != text {
			t.Errorf("%q: the pieces do not concatenate back", text)
		}
	}
}
