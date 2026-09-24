package qwen

import (
	"strings"
	"testing"
)

func TestParseToolCallsReadsWhatTheModelWrote(t *testing.T) {
	before, calls, err := ParseToolCalls(
		"Looking.\n<tool_call>\n{\"name\": \"weather\", \"arguments\": {\"city\": \"Lyon\"}}\n</tool_call>")
	if err != nil {
		t.Fatal(err)
	}
	if before != "Looking." {
		t.Fatalf("prose %q", before)
	}
	if len(calls) != 1 || calls[0].Name != "weather" || calls[0].Arguments["city"] != "Lyon" {
		t.Fatalf("calls %+v", calls)
	}
}

func TestParseToolCallsReadsTwo(t *testing.T) {
	_, calls, err := ParseToolCalls(
		"<tool_call>\n{\"name\": \"weather\", \"arguments\": {\"city\": \"Lyon\"}}\n</tool_call>\n" +
			"<tool_call>\n{\"name\": \"now\", \"arguments\": {}}\n</tool_call>")
	if err != nil {
		t.Fatal(err)
	}
	if len(calls) != 2 || calls[0].Name != "weather" || calls[1].Name != "now" {
		t.Fatalf("calls %+v", calls)
	}
	if len(calls[1].Arguments) != 0 {
		t.Fatalf("a call with no arguments got %+v", calls[1].Arguments)
	}
}

func TestParseToolCallsKeepsProseWithNoCall(t *testing.T) {
	before, calls, err := ParseToolCalls("just an answer")
	if err != nil || before != "just an answer" || calls != nil {
		t.Fatalf("%q %+v %v", before, calls, err)
	}
}

// What is parsed is fed straight back to the renderer when the conversation
// continues, so the two have to agree character for character.
func TestParseToolCallsRoundTripsThroughTheRenderer(t *testing.T) {
	written := "<tool_call>\n{\"name\": \"weather\", \"arguments\": {\"city\": \"Lyon\"}}\n</tool_call>"
	_, calls, err := ParseToolCalls(written)
	if err != nil {
		t.Fatal(err)
	}
	var b strings.Builder
	writeCall(&b, calls[0])
	if b.String() != written {
		t.Fatalf("rendered %q, want %q", b.String(), written)
	}
}

// A half-read call would be executed, so it is refused instead.
func TestParseToolCallsRefusesWhatItCannotRead(t *testing.T) {
	for _, text := range []string{
		"<tool_call>\n{\"name\": \"weather\", \"arguments\": {\"city\":",
		"<tool_call>\n{\"name\": \"weather\"}\n</tool_call>",
		"<tool_call>\n{\"arguments\": {}}\n</tool_call>",
		"<tool_call>\nnot json at all\n</tool_call>",
		"<tool_call>\n{\"name\": \"weather\", \"arguments\": \"Lyon\"}\n</tool_call>",
		"<tool_call>\n{\"name\": \"a\", \"arguments\": {}}\n</tool_call> and then prose",
	} {
		if _, _, err := ParseToolCalls(text); err == nil {
			t.Fatalf("%q was read as a call and should have been refused", text)
		}
	}
}

// Qwen3.5's template teaches the model another spelling, parameters as tags.
func TestParseToolCallsReadsTheTaggedSpelling(t *testing.T) {
	before, calls, err := ParseToolCalls("Je regarde.\n\n<tool_call>\n<function=weather>\n" +
		"<parameter=city>\nLyon\n</parameter>\n<parameter=days>\n3\n</parameter>\n" +
		"<parameter=note>\ntwo\nlines\n</parameter>\n<parameter=opts>\n{\"metric\": true}\n</parameter>\n" +
		"</function>\n</tool_call>\n<tool_call>\n<function=now>\n</function>\n</tool_call>")
	if err != nil {
		t.Fatal(err)
	}
	if before != "Je regarde." || len(calls) != 2 {
		t.Fatalf("%q %+v", before, calls)
	}
	a := calls[0].Arguments
	if calls[0].Name != "weather" || a["city"] != "Lyon" || a["days"] != 3.0 || a["note"] != "two\nlines" {
		t.Fatalf("call %+v", calls[0])
	}
	if opts, ok := a["opts"].(map[string]any); !ok || opts["metric"] != true {
		t.Fatalf("opts %+v", a["opts"])
	}
	if calls[1].Name != "now" || len(calls[1].Arguments) != 0 {
		t.Fatalf("call %+v", calls[1])
	}
}

func TestParseToolCallsRefusesAHalfTaggedCall(t *testing.T) {
	for _, text := range []string{
		"<tool_call>\n<function=weather>\n<parameter=city>\nLyon\n</function>\n</tool_call>",
		"<tool_call>\n<function=weather>\nLyon\n</function>\n</tool_call>",
		"<tool_call>\n<function=>\n</function>\n</tool_call>",
	} {
		if _, _, err := ParseToolCalls(text); err == nil {
			t.Fatalf("read %q", text)
		}
	}
}
