package main

import (
	"strings"
	"testing"
	"time"

	"github.com/ThiraSoft/golem/sample"
)

// A vocabulary of whole words: each space-separated piece is one token, which
// is enough to script an answer.
type wordVocab struct {
	texts []string
	index map[string]int32
}

func newWordVocab() *wordVocab { return &wordVocab{index: map[string]int32{}} }

func (v *wordVocab) id(text string) int32 {
	if id, ok := v.index[text]; ok {
		return id
	}
	v.texts = append(v.texts, text)
	v.index[text] = int32(len(v.texts) - 1)
	return int32(len(v.texts) - 1)
}

func (v *wordVocab) Encode(text string, addBOS, parseSpecial bool) []int32 {
	var out []int32
	for _, word := range strings.Fields(text) {
		out = append(out, v.id(word))
	}
	return out
}

func (v *wordVocab) Piece(id int32, special bool) string { return v.texts[id] }
func (v *wordVocab) IsEOG(id int32) bool                 { return v.texts[id] == "<turn|>" }

// An engine that speaks a script: each call to Logits names the next word.
type scriptedEngine struct {
	vocab  *wordVocab
	script []string
	at     int
}

func (e *scriptedEngine) ForwardBatch(tokens []int32, startPos int) [][]float32 {
	hidden := make([][]float32, len(tokens))
	for i := range tokens {
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

func (e *scriptedEngine) Reset() {}

func newGenerator(script []string, maxTokens int) (*Generator, *wordVocab) {
	v := newWordVocab()
	e := &scriptedEngine{vocab: v, script: script}
	for _, word := range script {
		v.id(word)
	}
	v.id("<turn|>")
	ctx := NewContext(e, 0, 4096, time.Now, 0)
	return NewGenerator(ctx, v, 4096, maxTokens), v
}

func greedy() sample.Params { return sample.Params{Temperature: 0} }

func TestGenerateStreamsProse(t *testing.T) {
	g, v := newGenerator([]string{"hello", "there", "<turn|>"}, 32)
	var seen []string
	answer, err := g.Generate(v.Encode("a b", false, true), greedy(), nil,
		func(text string) error { seen = append(seen, text); return nil })
	if err != nil {
		t.Fatal(err)
	}
	if answer.Text != "hellothere" || answer.Reason != "stop" {
		t.Fatalf("%+v", answer)
	}
	if strings.Join(seen, "") != "hellothere" || len(seen) != 2 {
		t.Fatalf("streamed %v", seen)
	}
}

func TestGenerateWithholdsAPartialCall(t *testing.T) {
	script := []string{"Looking.", `<|tool_call>call:weather{city:<|"|>Lyon<|"|>}`, "<tool_call|>", "<turn|>"}
	g, v := newGenerator(script, 32)
	var seen []string
	answer, err := g.Generate(v.Encode("a", false, true), greedy(), nil,
		func(text string) error { seen = append(seen, text); return nil })
	if err != nil {
		t.Fatal(err)
	}
	if len(seen) != 1 || seen[0] != "Looking." {
		t.Fatalf("streamed %v: a call reached the client as text", seen)
	}
	if len(answer.ToolCalls) != 1 || answer.ToolCalls[0].Name != "weather" {
		t.Fatalf("calls %#v", answer.ToolCalls)
	}
	if answer.ToolCalls[0].Arguments["city"] != "Lyon" {
		t.Fatalf("arguments %#v", answer.ToolCalls[0].Arguments)
	}
	if answer.Reason != "tool_calls" {
		t.Fatalf("reason %q", answer.Reason)
	}
	if answer.Text != "Looking." {
		t.Fatalf("text %q: the call belongs in ToolCalls, not in the content", answer.Text)
	}
	if answer.ToolCalls[0].ID == "" {
		t.Fatal("a call needs an identifier the client can echo back")
	}
}

// Identifiers are not reused between answers: a client holding two calls has
// two names for them.
func TestCallIdentifiersDoNotRepeat(t *testing.T) {
	call := `<|tool_call>call:now{}<tool_call|>`
	g, v := newGenerator([]string{call, "<turn|>", call, "<turn|>"}, 32)
	first, err := g.Generate(v.Encode("a", false, true), greedy(), nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	second, err := g.WithMaxTokens(8).Generate(v.Encode("b", false, true), greedy(), nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(second.ToolCalls) == 0 || first.ToolCalls[0].ID == second.ToolCalls[0].ID {
		t.Fatalf("%v then %v", first.ToolCalls, second.ToolCalls)
	}
}

func TestGenerateStopsOnTheTokenLimit(t *testing.T) {
	g, v := newGenerator([]string{"and", "and", "and", "and"}, 2)
	answer, err := g.Generate(v.Encode("a", false, true), greedy(), nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if answer.Generated != 2 || answer.Reason != "length" {
		t.Fatalf("%+v", answer)
	}
}

func TestGenerateStopsOnAStopString(t *testing.T) {
	g, v := newGenerator([]string{"one", "two", "STOP", "three"}, 32)
	answer, err := g.Generate(v.Encode("a", false, true), greedy(), []string{"STOP"}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if answer.Text != "onetwo" || answer.Reason != "stop" {
		t.Fatalf("%+v", answer)
	}
}
