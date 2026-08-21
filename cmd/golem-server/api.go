package main

// The shapes OpenAI's API defines, as much of them as this server implements.

import (
	"encoding/json"

	"github.com/ThiraSoft/golem/gemma"
)

type completionRequest struct {
	Model       string          `json:"model"`
	Messages    []gemma.Message `json:"messages"`
	Tools       []gemma.Tool    `json:"tools"`
	ToolChoice  any             `json:"tool_choice"`
	Stream      bool            `json:"stream"`
	Temperature *float64        `json:"temperature"`
	TopP        *float64        `json:"top_p"`
	TopK        *int            `json:"top_k"`
	MaxTokens   *int            `json:"max_tokens"`
	Seed        *uint64         `json:"seed"`
	Stop        stopStrings     `json:"stop"`
	N           *int            `json:"n"`
	LogProbs    *bool           `json:"logprobs"`
}

// stopStrings reads the two spellings the API allows: one string, or a list.
type stopStrings []string

func (s *stopStrings) UnmarshalJSON(raw []byte) error {
	if len(raw) > 0 && raw[0] == '"' {
		var one string
		if err := json.Unmarshal(raw, &one); err != nil {
			return err
		}
		*s = []string{one}
		return nil
	}
	var many []string
	if err := json.Unmarshal(raw, &many); err != nil {
		return err
	}
	*s = many
	return nil
}

type responseMessage struct {
	Role      string           `json:"role,omitempty"`
	Content   string           `json:"content"`
	ToolCalls []gemma.ToolCall `json:"tool_calls,omitempty"`
}

type choice struct {
	Index        int              `json:"index"`
	Message      *responseMessage `json:"message,omitempty"`
	Delta        *responseMessage `json:"delta,omitempty"`
	FinishReason *string          `json:"finish_reason"`
}

type usage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

type completionResponse struct {
	ID      string   `json:"id"`
	Object  string   `json:"object"`
	Created int64    `json:"created"`
	Model   string   `json:"model"`
	Choices []choice `json:"choices"`
	Usage   *usage   `json:"usage,omitempty"`
}

type apiError struct {
	Error struct {
		Message string `json:"message"`
		Type    string `json:"type"`
		Code    string `json:"code,omitempty"`
	} `json:"error"`
}
