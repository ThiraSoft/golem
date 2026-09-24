package main

import (
	"encoding/json"
	"strings"
	"testing"
)

// frames splits an SSE body into the payloads it carried.
func frames(t *testing.T, body string) []string {
	t.Helper()
	var out []string
	for _, line := range strings.Split(body, "\n") {
		if strings.HasPrefix(line, "data: ") {
			out = append(out, strings.TrimPrefix(line, "data: "))
		}
	}
	if len(out) == 0 {
		t.Fatalf("no SSE frames in %q", body)
	}
	return out
}

func TestStreamSendsProseThenAFinishReasonThenDone(t *testing.T) {
	s := newTestServer(t, []string{"one", "two", "<turn|>"})
	w := post(t, s, `{"messages":[{"role":"user","content":"hi"}],"stream":true}`)
	if got := w.Header().Get("Content-Type"); got != "text/event-stream" {
		t.Fatalf("content type %q", got)
	}
	f := frames(t, w.Body.String())
	if f[len(f)-1] != "[DONE]" {
		t.Fatalf("the stream did not close with [DONE]: %v", f)
	}
	var text strings.Builder
	var reason string
	for _, frame := range f[:len(f)-1] {
		var chunk struct {
			Object  string `json:"object"`
			Choices []struct {
				Delta struct {
					Role    string `json:"role"`
					Content string `json:"content"`
				} `json:"delta"`
				FinishReason *string `json:"finish_reason"`
			} `json:"choices"`
		}
		if err := json.Unmarshal([]byte(frame), &chunk); err != nil {
			t.Fatalf("%q: %v", frame, err)
		}
		if chunk.Object != "chat.completion.chunk" {
			t.Fatalf("object %q", chunk.Object)
		}
		text.WriteString(chunk.Choices[0].Delta.Content)
		if chunk.Choices[0].FinishReason != nil {
			reason = *chunk.Choices[0].FinishReason
		}
	}
	if text.String() != "onetwo" {
		t.Fatalf("streamed %q", text.String())
	}
	if reason != "stop" {
		t.Fatalf("reason %q", reason)
	}
}

func TestStreamSendsACallInOnePiece(t *testing.T) {
	s := newTestServer(t, []string{"Looking.", "CALL", "weather{city=Lyon}", "<turn|>"})
	w := post(t, s, `{"messages":[{"role":"user","content":"hi"}],"stream":true}`)
	f := frames(t, w.Body.String())

	var prose strings.Builder
	callFrames := 0
	var reason string
	for _, frame := range f[:len(f)-1] {
		var chunk struct {
			Choices []struct {
				Delta struct {
					Content   string `json:"content"`
					ToolCalls []struct {
						Index    int    `json:"index"`
						ID       string `json:"id"`
						Function struct {
							Name      string `json:"name"`
							Arguments string `json:"arguments"`
						} `json:"function"`
					} `json:"tool_calls"`
				} `json:"delta"`
				FinishReason *string `json:"finish_reason"`
			} `json:"choices"`
		}
		if err := json.Unmarshal([]byte(frame), &chunk); err != nil {
			t.Fatal(err)
		}
		d := chunk.Choices[0].Delta
		prose.WriteString(d.Content)
		if len(d.ToolCalls) > 0 {
			callFrames++
			if d.ToolCalls[0].Function.Arguments != `{"city":"Lyon"}` {
				t.Fatalf("arguments %q left in pieces", d.ToolCalls[0].Function.Arguments)
			}
			if d.ToolCalls[0].ID == "" {
				t.Fatal("a streamed call needs its identifier")
			}
		}
		if chunk.Choices[0].FinishReason != nil {
			reason = *chunk.Choices[0].FinishReason
		}
	}
	if callFrames != 1 {
		t.Fatalf("%d frames carried the call, want exactly one", callFrames)
	}
	if strings.Contains(prose.String(), "CALL") {
		t.Fatalf("the call leaked into the prose: %q", prose.String())
	}
	if prose.String() != "Looking." {
		t.Fatalf("prose %q", prose.String())
	}
	if reason != "tool_calls" {
		t.Fatalf("reason %q", reason)
	}
}

// A streamed answer is constrained like any other: the grammar reaches the
// sampler through the same parameters, and nothing in the streaming path
// touches them.
func TestStreamStaysInsideTheGrammar(t *testing.T) {
	s := newTestServer(t, []string{"no", "no", "<turn|>"}, "yes")
	w := post(t, s, `{"messages":[{"role":"user","content":"hi"}],"stream":true,
		"grammar":"root ::= \"yes\""}`)

	var text strings.Builder
	for _, frame := range frames(t, w.Body.String()) {
		if frame == "[DONE]" {
			continue
		}
		var chunk struct {
			Choices []struct {
				Delta struct {
					Content string `json:"content"`
				} `json:"delta"`
			} `json:"choices"`
		}
		if err := json.Unmarshal([]byte(frame), &chunk); err != nil {
			t.Fatalf("%q: %v", frame, err)
		}
		text.WriteString(chunk.Choices[0].Delta.Content)
	}
	if text.String() != "yes" {
		t.Fatalf("the stream carried %q, and the grammar allows only yes", text.String())
	}
}

// The reasoning leaves in its own field, reasoning_content, as llama-server
// sends it: streamed as it is drawn, and whole in an answer that is not.
func TestReasoningLeavesApartFromTheContent(t *testing.T) {
	script := []string{"<think>", "pondering", "</think>", "Answer.", "<turn|>"}

	w := post(t, newTestServer(t, script), `{"messages":[{"role":"user","content":"hi"}],"stream":true}`)
	f := frames(t, w.Body.String())
	var content, reasoning strings.Builder
	for _, frame := range f[:len(f)-1] {
		var chunk struct {
			Choices []struct {
				Delta struct {
					Content   string `json:"content"`
					Reasoning string `json:"reasoning_content"`
				} `json:"delta"`
			} `json:"choices"`
		}
		if err := json.Unmarshal([]byte(frame), &chunk); err != nil {
			t.Fatalf("%q: %v", frame, err)
		}
		content.WriteString(chunk.Choices[0].Delta.Content)
		reasoning.WriteString(chunk.Choices[0].Delta.Reasoning)
	}
	if content.String() != "Answer." || reasoning.String() != "pondering" {
		t.Fatalf("streamed content %q, reasoning %q", content.String(), reasoning.String())
	}

	w = post(t, newTestServer(t, script), `{"messages":[{"role":"user","content":"hi"}]}`)
	var body struct {
		Choices []struct {
			Message struct {
				Content   string `json:"content"`
				Reasoning string `json:"reasoning_content"`
			} `json:"message"`
		} `json:"choices"`
	}
	if err := json.Unmarshal(w.Body.Bytes(), &body); err != nil {
		t.Fatal(err)
	}
	if m := body.Choices[0].Message; m.Content != "Answer." || m.Reasoning != "pondering" {
		t.Fatalf("content %q, reasoning %q", m.Content, m.Reasoning)
	}
}

// An error once the stream is open goes as its own event, never in the prose,
// where a client would show it as something the model said.
func TestStreamSendsAnErrorApartFromTheProse(t *testing.T) {
	s := newTestServer(t, []string{"Looking.", "CALL", "{city=Lyon}", "<turn|>"})
	w := post(t, s, `{"messages":[{"role":"user","content":"hi"}],"stream":true}`)
	f := frames(t, w.Body.String())
	last := f[len(f)-1]
	var event struct {
		Error struct {
			Message string `json:"message"`
			Type    string `json:"type"`
		} `json:"error"`
		Choices []json.RawMessage `json:"choices"`
	}
	if err := json.Unmarshal([]byte(last), &event); err != nil {
		t.Fatalf("%q: %v", last, err)
	}
	if event.Error.Type != "server_error" || !strings.Contains(event.Error.Message, "no function name") || event.Choices != nil {
		t.Fatalf("last frame %q", last)
	}
	if strings.Contains(w.Body.String(), "[golem") || strings.Contains(w.Body.String(), "[DONE]") {
		t.Fatalf("body %q", w.Body.String())
	}
}

// With stream_options.include_usage, a last chunk before [DONE] says what the
// answer cost, and has no choices.
func TestStreamSendsUsageWhenAsked(t *testing.T) {
	s := newTestServer(t, []string{"one", "two", "<turn|>"})
	w := post(t, s, `{"messages":[{"role":"user","content":"hi"}],"stream":true,"stream_options":{"include_usage":true}}`)
	f := frames(t, w.Body.String())
	var chunk struct {
		Choices []json.RawMessage `json:"choices"`
		Usage   *struct {
			Prompt     int `json:"prompt_tokens"`
			Completion int `json:"completion_tokens"`
		} `json:"usage"`
	}
	if err := json.Unmarshal([]byte(f[len(f)-2]), &chunk); err != nil {
		t.Fatal(err)
	}
	if chunk.Usage == nil || chunk.Usage.Completion == 0 || chunk.Usage.Prompt == 0 || len(chunk.Choices) != 0 {
		t.Fatalf("usage chunk %s", f[len(f)-2])
	}
	w = post(t, s, `{"messages":[{"role":"user","content":"hi"}],"stream":true}`)
	if strings.Contains(w.Body.String(), "prompt_tokens") {
		t.Fatal("usage sent unasked")
	}
}
