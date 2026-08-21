package main

import (
	"context"
	"errors"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/ThiraSoft/golem/chat"
	"github.com/ThiraSoft/golem/sample"
)

// A vocabulary of whole words: each space-separated piece is one token, which
// is enough to script an answer.
// A real vocabulary is read only and shared freely; this one grows as words
// are met, so it holds a lock the real one has no use for.
type wordVocab struct {
	mu    sync.Mutex
	texts []string
	index map[string]int32
}

func newWordVocab() *wordVocab { return &wordVocab{index: map[string]int32{}} }

func (v *wordVocab) id(text string) int32 {
	v.mu.Lock()
	defer v.mu.Unlock()
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

func (v *wordVocab) Piece(id int32, special bool) string {
	v.mu.Lock()
	defer v.mu.Unlock()
	return v.texts[id]
}
func (v *wordVocab) IsEOG(id int32) bool { return v.texts[id] == "<turn|>" }

// An engine that speaks a script: each call to Logits names the next word.
type scriptedEngine struct {
	vocab  *wordVocab
	script []string
	at     int
}

func (e *scriptedEngine) ForwardSlots(tokens []int32, slots, positions []int) [][]float32 {
	return e.ForwardBatch(tokens, positions[0])
}

func (e *scriptedEngine) ForwardBatch(tokens []int32, startPos int) [][]float32 {
	hidden := make([][]float32, len(tokens))
	for i := range tokens {
		hidden[i] = []float32{0}
	}
	return hidden
}

// Each state scored takes the next word of the script, which is what makes a
// batch of several states legible: they take them in the order of the batch.
func (e *scriptedEngine) LogitsBatch(hidden [][]float32, out [][]float32) {
	for _, o := range out {
		clear(o)
		word := "<turn|>"
		if e.at < len(e.script) {
			word = e.script[e.at]
		}
		e.at++
		o[e.vocab.id(word)] = 100
	}
}

func (e *scriptedEngine) Reset()      {}
func (e *scriptedEngine) UseSlot(int) {}

func newGenerator(tb testing.TB, script []string, maxTokens int) (*Generator, *wordVocab) {
	v := newWordVocab()
	e := &scriptedEngine{vocab: v, script: script}
	for _, word := range script {
		v.id(word)
	}
	v.id("<turn|>")
	ctx := NewContext(running(tb, e), 0, 4096, time.Now, 0)
	return NewGenerator(ctx, v, wordTemplate{}, 4096, maxTokens), v
}

func greedy() sample.Params { return sample.Params{Temperature: 0} }

func TestGenerateStreamsProse(t *testing.T) {
	g, v := newGenerator(t, []string{"hello", "there", "<turn|>"}, 32)
	var seen []string
	answer, err := g.Generate(context.Background(), v.Encode("a b", false, true), greedy(), nil,
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
	script := []string{"Looking.", "CALL", "weather{city=Lyon}", "<turn|>"}
	g, v := newGenerator(t, script, 32)
	var seen []string
	answer, err := g.Generate(context.Background(), v.Encode("a", false, true), greedy(), nil,
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
	call := "CALLnow{}"
	g, v := newGenerator(t, []string{call, "<turn|>", call, "<turn|>"}, 32)
	first, err := g.Generate(context.Background(), v.Encode("a", false, true), greedy(), nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	second, err := g.WithMaxTokens(8).Generate(context.Background(), v.Encode("b", false, true), greedy(), nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(second.ToolCalls) == 0 || first.ToolCalls[0].ID == second.ToolCalls[0].ID {
		t.Fatalf("%v then %v", first.ToolCalls, second.ToolCalls)
	}
}

func TestGenerateStopsOnTheTokenLimit(t *testing.T) {
	g, v := newGenerator(t, []string{"and", "and", "and", "and"}, 2)
	answer, err := g.Generate(context.Background(), v.Encode("a", false, true), greedy(), nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if answer.Generated != 2 || answer.Reason != "length" {
		t.Fatalf("%+v", answer)
	}
}

func TestGenerateStopsOnAStopString(t *testing.T) {
	g, v := newGenerator(t, []string{"one", "two", "STOP", "three"}, 32)
	answer, err := g.Generate(context.Background(), v.Encode("a", false, true), greedy(), []string{"STOP"}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if answer.Text != "onetwo" || answer.Reason != "stop" {
		t.Fatalf("%+v", answer)
	}
}

// A client that hangs up stops the work. Without this the server would draw an
// answer nobody is waiting for, holding the one lock the next request needs.
func TestGenerateStopsWhenTheClientHangsUp(t *testing.T) {
	script := make([]string, 200)
	for i := range script {
		script[i] = "on"
	}
	g, v := newGenerator(t, script, 200)
	ctx, cancel := context.WithCancel(context.Background())
	drawn := 0
	_, err := g.Generate(ctx, v.Encode("a", false, true), greedy(), nil,
		func(string) error {
			drawn++
			if drawn == 3 {
				cancel()
			}
			return nil
		})
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("err %v, want the cancellation", err)
	}
	if drawn > 5 {
		t.Fatalf("%d pieces drawn after the client left", drawn)
	}
}

// A template of whole words, so that the server's tests say nothing about
// which engine answered. Prose stops at CALL; what follows is a function's
// name and, between braces, its arguments: CALLweather{city=Lyon}.
type wordTemplate struct{}

func (wordTemplate) Render(msgs []chat.Message, opt chat.Options) (string, error) {
	// Both real templates refuse these, so the fake has to as well: otherwise
	// the tests that check the server refuses them pass for the wrong reason.
	if len(msgs) == 0 {
		return "", errors.New("an empty conversation")
	}
	for i, m := range msgs {
		if m.Role != "tool" {
			continue
		}
		if i == 0 || (msgs[i-1].Role != "tool" && len(msgs[i-1].ToolCalls) == 0) {
			return "", errors.New("a tool result answering no call")
		}
	}
	var b strings.Builder
	for _, m := range msgs {
		b.WriteString(m.Role + " " + m.Content + " ")
	}
	for _, tool := range opt.Tools {
		b.WriteString("tool " + tool.Name + " ")
	}
	return b.String(), nil
}

func (wordTemplate) CallOpen() string { return "CALL" }

func (wordTemplate) ParseCalls(text string) (string, []chat.ToolCall, error) {
	at := strings.Index(text, "CALL")
	if at < 0 {
		return text, nil, nil
	}
	before, rest := text[:at], text[at:]
	var calls []chat.ToolCall
	for _, piece := range strings.Split(rest, "CALL")[1:] {
		name, args, found := strings.Cut(piece, "{")
		if name == "" {
			return before, nil, errors.New("a call with no function name")
		}
		call := chat.ToolCall{Name: name}
		if found {
			body, closed := strings.CutSuffix(args, "}")
			if !closed {
				return before, nil, errors.New("a call that never closes")
			}
			call.Arguments = map[string]any{}
			for _, pair := range strings.Split(body, ",") {
				if pair == "" {
					continue
				}
				key, value, ok := strings.Cut(pair, "=")
				if !ok {
					return before, nil, errors.New("an argument that is not a pair")
				}
				call.Arguments[key] = value
			}
		}
		calls = append(calls, call)
	}
	return before, calls, nil
}
