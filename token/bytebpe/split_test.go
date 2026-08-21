package bytebpe

import (
	"strings"
	"testing"
)

// The expectations here were traced by hand through llama.cpp's
// unicode_regex_split_custom_qwen2, rule by rule. The authority is
// reference_test.go, which compares whole segmentations against what
// llama.cpp actually produced; when the two disagree, this table is wrong.
func TestSplitQwen2(t *testing.T) {
	cases := []struct {
		name string
		in   string
		want []string
	}{
		{"empty", "", nil},
		{"word", "hello", []string{"hello"}},
		{"leading_space", " hello", []string{" hello"}},
		{"two_words", "hello world", []string{"hello", " world"}},

		// Rule 1, both cases, and the three-character forms.
		{"contraction", "don't", []string{"don", "'t"}},
		{"contraction_upper", "DON'T", []string{"DON", "'T"}},
		{"contraction_ll", "we'll", []string{"we", "'ll"}},
		{"contraction_ve", "they've", []string{"they", "'ve"}},
		{"apostrophe_alone", "rock 'n roll", []string{"rock", " '", "n", " roll"}},

		// Rule 2 takes its optional leading character whatever it is.
		{"punct_then_letter", "a,b", []string{"a", ",b"}},
		{"punct_space_letter", "a, b", []string{"a", ",", " b"}},

		// Rule 3 is one digit at a time.
		{"digits", "1234", []string{"1", "2", "3", "4"}},
		{"digits_spaced", " 42", []string{" ", "4", "2"}},

		// Rule 4, including the line endings it swallows.
		{"punct_run", "...", []string{"..."}},
		{"punct_newline", "!\n", []string{"!\n"}},
		{"space_punct", " .", []string{" ."}},

		// Rule 5: the run stops after the last line ending.
		{"newline", "a\nb", []string{"a", "\n", "b"}},
		{"newline_run", "a\n\n\nb", []string{"a", "\n\n\n", "b"}},
		{"spaces_then_newline", "a  \nb", []string{"a", "  \n", "b"}},

		// Rules 6 and 7: the last space goes to what follows, unless nothing does.
		{"interior_spaces", "a   b", []string{"a", "  ", " b"}},
		{"trailing_spaces", "a   ", []string{"a", "   "}},
		{"only_spaces", "   ", []string{"   "}},
		{"single_interior_space", "a b", []string{"a", " b"}},

		{"cjk", "日本語", []string{"日本語"}},
		{"mixed", "hello 世界", []string{"hello", " 世界"}},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got := splitQwen2(c.in)
			if strings.Join(got, "") != c.in {
				t.Fatalf("the pieces %q do not rebuild %q", got, c.in)
			}
			if len(got) != len(c.want) {
				t.Fatalf("splitQwen2(%q) = %q, want %q", c.in, got, c.want)
			}
			for i := range got {
				if got[i] != c.want[i] {
					t.Fatalf("splitQwen2(%q) = %q, want %q", c.in, got, c.want)
				}
			}
		})
	}
}

// Whatever the input, the pieces concatenate back to it. This is the property
// a hand-written scanner breaks first, and it holds even where the table above
// might be wrong about which pieces they are.
func TestSplitQwen2LosesNothing(t *testing.T) {
	inputs := []string{
		"", " ", "\n", "\r\n", "\t\t", "a", "  a  b  ",
		"L'élève a mangé une crêpe à Nîmes.",
		"Привет, мир — hello 世界",
		"bravo 🎉🇫🇷 et voilà",
		"func main() {\n\tfmt.Println(\"hi\")\n}",
		"1234567890 and 42 and 3.14159",
		"\xef\xbf\xbd unlikely \xe2\x80\x8b zero width",
		"Wait... really?! (yes) [no] {maybe} <a>",
		"line\r\nline",
		" non-breaking ",
	}
	for _, in := range inputs {
		if got := strings.Join(splitQwen2(in), ""); got != in {
			t.Errorf("splitQwen2(%q) rebuilt as %q", in, got)
		}
	}
}

// The scan must always move, whatever it is given: a rule that matches nothing
// and does not advance would hang on the first character it cannot classify.
func TestSplitQwen2Terminates(t *testing.T) {
	for r := rune(0); r < 0x300; r++ {
		s := string(r)
		if got := strings.Join(splitQwen2(s), ""); got != s {
			t.Fatalf("U+%04X alone rebuilt as %q", r, got)
		}
	}
}
