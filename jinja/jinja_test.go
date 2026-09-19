package jinja

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
)

// The fixture is what Jinja2 rendered, in transformers' environment, from the
// same templates and the same variables. ref/jinja/dump_cases.py writes it.
type fixture struct {
	Snippets  []fixtureCase `json:"snippets"`
	Templates []struct {
		File  string        `json:"file"`
		Cases []fixtureCase `json:"cases"`
	} `json:"templates"`
}

type fixtureCase struct {
	Name     string          `json:"name"`
	Template string          `json:"template"`
	Vars     json.RawMessage `json:"vars"`
	Rendered *string         `json:"rendered"`
	Error    string          `json:"error"`
}

func loadFixture(t *testing.T) fixture {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join("..", "testdata", "jinja", "cases.json"))
	if err != nil {
		t.Fatalf("the fixture is committed and should be readable: %v", err)
	}
	var f fixture
	if err := json.Unmarshal(raw, &f); err != nil {
		t.Fatal(err)
	}
	return f
}

// vars decodes a case's variables with their keys in the order Python had
// them, which tojson and iteration both show.
func vars(t *testing.T, raw json.RawMessage) map[string]any {
	t.Helper()
	v, err := ParseJSON(raw)
	if err != nil {
		t.Fatal(err)
	}
	d := v.(*Dict)
	out := map[string]any{}
	for _, k := range d.keys {
		out[k.(string)] = d.vals[k]
	}
	return out
}

func check(t *testing.T, tpl *Template, c fixtureCase) {
	t.Helper()
	got, err := tpl.Render(vars(t, c.Vars))
	if c.Rendered == nil {
		if err == nil {
			t.Fatalf("Jinja2 refused this (%s) and it rendered %q", c.Error, got)
		}
		return
	}
	if err != nil {
		t.Fatalf("Jinja2 rendered it and this failed: %v", err)
	}
	if got != *c.Rendered {
		t.Fatalf("rendered\n%q\nJinja2 rendered\n%q\n%s", got, *c.Rendered, firstDifference(got, *c.Rendered))
	}
}

func firstDifference(a, b string) string {
	i := 0
	for i < len(a) && i < len(b) && a[i] == b[i] {
		i++
	}
	lo := max(0, i-60)
	return "they part at byte " + strconv.Itoa(i) + ", after " + strings.ReplaceAll(a[lo:i], "\n", `\n`)
}

func TestSnippetsMatchJinja2(t *testing.T) {
	for _, c := range loadFixture(t).Snippets {
		t.Run(c.Template, func(t *testing.T) {
			tpl, err := Parse(c.Template)
			if err != nil {
				t.Fatal(err)
			}
			check(t, tpl, c)
		})
	}
}

// Every chat template golem has met in a model file, over every conversation
// the fixture holds, character for character.
func TestChatTemplatesMatchJinja2(t *testing.T) {
	for _, file := range loadFixture(t).Templates {
		src, err := os.ReadFile(filepath.Join("..", "testdata", "jinja", "templates", file.File))
		if err != nil {
			t.Fatal(err)
		}
		tpl, err := Parse(string(src))
		if err != nil {
			t.Fatalf("%s: %v", file.File, err)
		}
		for i, c := range file.Cases {
			t.Run(file.File+"/"+c.Name+"/"+strconv.Itoa(i%4), func(t *testing.T) {
				check(t, tpl, c)
			})
		}
	}
}

func TestRaiseExceptionIsAnError(t *testing.T) {
	tpl, err := Parse("{{ raise_exception('no ' ~ what) }}")
	if err != nil {
		t.Fatal(err)
	}
	_, err = tpl.Render(map[string]any{"what": "images"})
	e, ok := err.(*Error)
	if !ok || e.Message != "no images" {
		t.Fatalf("got %v, want the template's own message", err)
	}
}

func TestParseRefusesWhatItCannotRead(t *testing.T) {
	for _, src := range []string{
		"{% if x %}never closed",
		"{% endif %}",
		"{{ x ",
		"{% include 'other' %}",
		"{% for x in %}{% endfor %}",
		"{{ 'open }}",
	} {
		if _, err := Parse(src); err == nil {
			t.Errorf("%q parsed", src)
		}
	}
}

func TestFromGoSortsKeysAndReadsWholeNumbersAsIntegers(t *testing.T) {
	v := FromGo(map[string]any{"b": 3.0, "a": []any{1.5, "x"}})
	if got := repr(v); got != "{'a': [1.5, 'x'], 'b': 3}" {
		t.Fatalf("got %s", got)
	}
}
