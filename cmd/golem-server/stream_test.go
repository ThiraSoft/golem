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
	s := newTestServer([]string{"one", "two", "<turn|>"})
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
	s := newTestServer([]string{"Looking.", "CALL", "weather{city=Lyon}", "<turn|>"})
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
