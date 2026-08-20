package text

import "testing"

// TestPrepareRules covers the two per-language flags. Both change the text the
// model sees, and neither is visible in the audio until the prosody is wrong.
func TestPrepareRules(t *testing.T) {
	cases := []struct {
		in    string
		rules Rules
		want  string
	}{
		{"un texte ; avec un point-virgule", Rules{DropSemicolons: true}, "Un texte , avec un point-virgule."},
		{"un texte ; avec un point-virgule", Rules{}, "Un texte ; avec un point-virgule."},
		// Fewer than five words, padding on: eight spaces in front.
		{"Hello world.", Rules{PadShortInputs: true}, "        Hello world."},
		{"Hello world.", Rules{}, "Hello world."},
		// Five words or more: the padding does not apply.
		{"one two three four five", Rules{PadShortInputs: true}, "One two three four five."},
	}
	for _, c := range cases {
		if got := Prepare(c.in, c.rules); got != c.want {
			t.Errorf("Prepare(%q, %+v) = %q, want %q", c.in, c.rules, got, c.want)
		}
	}
}
