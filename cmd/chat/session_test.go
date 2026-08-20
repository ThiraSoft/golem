package main

import (
	"strings"
	"testing"

	"github.com/ThiraSoft/golem/sample"
)

// A vocabulary of whole words: every space-separated piece of the fed text is
// one identifier. Enough to say what the session put in the context, which is
// what these tests are about.
type wordVocab struct {
	texts []string
	index map[string]int32
	fed   []string // what Encode was asked to turn into tokens, turn by turn
}

func newWordVocab() *wordVocab {
	return &wordVocab{index: map[string]int32{}}
}

func (v *wordVocab) id(text string) int32 {
	if id, ok := v.index[text]; ok {
		return id
	}
	v.texts = append(v.texts, text)
	id := int32(len(v.texts) - 1)
	v.index[text] = id
	return id
}

func (v *wordVocab) Encode(text string, addBOS, parseSpecial bool) []int32 {
	v.fed = append(v.fed, text)
	var out []int32
	for _, word := range strings.Fields(text) {
		out = append(out, v.id(word))
	}
	return out
}

func (v *wordVocab) Piece(id int32, special bool) string { return v.texts[id] }
func (v *wordVocab) IsEOG(id int32) bool                 { return v.texts[id] == "<turn|>" }

// An engine that answers from a script: each call to Logits names the next
// token of the script, and the hidden state carries nothing.
type scriptedEngine struct {
	vocab  *wordVocab
	script []string
	at     int
	fed    []int32
	posOf  []int
}

func (e *scriptedEngine) ForwardBatch(tokens []int32, startPos int) [][]float32 {
	hidden := make([][]float32, len(tokens))
	for i, token := range tokens {
		e.fed = append(e.fed, token)
		e.posOf = append(e.posOf, startPos+i)
		hidden[i] = []float32{0}
	}
	return hidden
}

func (e *scriptedEngine) Logits(hidden, out []float32) {
	for i := range out {
		out[i] = 0
	}
	word := "<turn|>"
	if e.at < len(e.script) {
		word = e.script[e.at]
	}
	e.at++
	out[e.vocab.id(word)] = 100
}

func newSession(t *testing.T, v *wordVocab, e *scriptedEngine, maxTokens int, system string) *Session {
	t.Helper()
	// Greedy, so the script is the answer.
	return NewSession(e, v, sample.Params{Temperature: 0}, 4096, 4096, maxTokens, system, false)
}

// The context a token at a time: what the model was fed, in order.
func (e *scriptedEngine) context(v *wordVocab) string {
	words := make([]string, len(e.fed))
	for i, id := range e.fed {
		words[i] = v.texts[id]
	}
	return strings.Join(words, " ")
}

func TestFirstTurnFeedsTheWholeTemplate(t *testing.T) {
	v := newWordVocab()
	e := &scriptedEngine{vocab: v, script: []string{"hello", "there", "<turn|>"}}
	s := newSession(t, v, e, 32, "Be terse.")

	turn, err := s.Ask("  hi  ", nil)
	if err != nil {
		t.Fatal(err)
	}
	if turn.Text != "hellothere" {
		t.Fatalf("answer %q", turn.Text)
	}
	if turn.Generated != 3 || turn.Truncated {
		t.Fatalf("%+v", turn)
	}
	want := "<bos><|turn>system Be terse.<turn|> <|turn>user hi<turn|> <|turn>model hello there"
	if got := e.context(v); got != want {
		t.Fatalf("context\n%q\nwant\n%q", got, want)
	}
	// The prompt was the render, the answer's own tokens are not prompt tokens.
	if turn.Prompt != 6 {
		t.Fatalf("%d prompt tokens", turn.Prompt)
	}
}

// The second turn encodes only what is new, and closes the turn the model
// stopped on — the end-of-turn token was drawn but never fed.
func TestSecondTurnOnlyEncodesTheContinuation(t *testing.T) {
	v := newWordVocab()
	e := &scriptedEngine{vocab: v, script: []string{"one", "<turn|>", "two", "<turn|>"}}
	s := newSession(t, v, e, 32, "")

	if _, err := s.Ask("first", nil); err != nil {
		t.Fatal(err)
	}
	if _, err := s.Ask("second", nil); err != nil {
		t.Fatal(err)
	}

	if len(v.fed) != 2 {
		t.Fatalf("%d encodings", len(v.fed))
	}
	want := "<turn|>\n<|turn>user\nsecond<turn|>\n<|turn>model\n"
	if v.fed[1] != want {
		t.Fatalf("the continuation was %q, want %q", v.fed[1], want)
	}
	if strings.Contains(v.fed[1], "<bos>") || strings.Contains(v.fed[1], "first") {
		t.Fatalf("the second turn re-sent the conversation: %q", v.fed[1])
	}

	full := "<bos><|turn>user first<turn|> <|turn>model one <turn|> <|turn>user second<turn|> <|turn>model two"
	if got := e.context(v); got != full {
		t.Fatalf("context\n%q\nwant\n%q", got, full)
	}
}

// Positions are consecutive across turns: the cache is never rebuilt.
func TestPositionsRunOnAcrossTurns(t *testing.T) {
	v := newWordVocab()
	e := &scriptedEngine{vocab: v, script: []string{"a", "<turn|>", "b", "<turn|>"}}
	s := newSession(t, v, e, 32, "")

	if _, err := s.Ask("one", nil); err != nil {
		t.Fatal(err)
	}
	if _, err := s.Ask("two", nil); err != nil {
		t.Fatal(err)
	}
	for i, pos := range e.posOf {
		if pos != i {
			t.Fatalf("token %d was fed at position %d", i, pos)
		}
	}
}

// A model that never stops is stopped, and the turn says so.
func TestTheTokenLimitCutsTheAnswerShort(t *testing.T) {
	v := newWordVocab()
	e := &scriptedEngine{vocab: v, script: []string{"and", "and", "and", "and", "and"}}
	s := newSession(t, v, e, 3, "")

	turn, err := s.Ask("go", nil)
	if err != nil {
		t.Fatal(err)
	}
	if turn.Generated != 3 || !turn.Truncated {
		t.Fatalf("%+v", turn)
	}
	if turn.Text != "andandand" {
		t.Fatalf("answer %q", turn.Text)
	}
}

// The answer reaches the writer as it is drawn, not once it is finished.
func TestTheAnswerIsStreamed(t *testing.T) {
	v := newWordVocab()
	e := &scriptedEngine{vocab: v, script: []string{"x", "y", "<turn|>"}}
	s := newSession(t, v, e, 32, "")

	var b strings.Builder
	turn, err := s.Ask("go", &b)
	if err != nil {
		t.Fatal(err)
	}
	if b.String() != turn.Text || b.String() != "xy" {
		t.Fatalf("written %q, returned %q", b.String(), turn.Text)
	}
}

func TestAConversationThatNoLongerFitsIsRefused(t *testing.T) {
	v := newWordVocab()
	e := &scriptedEngine{vocab: v, script: []string{"<turn|>"}}
	s := NewSession(e, v, sample.Params{Temperature: 0}, 4096, 4, 8, "", false)

	if _, err := s.Ask("this prompt is longer than four positions", nil); err == nil {
		t.Fatal("a prompt past the context should be refused rather than wrap the cache")
	}
}
