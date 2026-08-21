package main

// The shapes OpenAI's API defines, as much of them as this server implements.

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"os"
	"strings"

	"github.com/ThiraSoft/golem/chat"
)

type completionRequest struct {
	Model       string          `json:"model"`
	Messages    requestMessages `json:"messages"`
	Tools       []chat.Tool     `json:"tools"`
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

// requestMessages reads the two shapes the API gives a turn's content: a
// string, or an array of parts. A part is text or a picture, and the pictures
// of a turn are collected in the order they were sent, before its text.
//
// This lives here rather than in chat, which is neutral about wire formats:
// what a JSON body looks like is this server's business.
type requestMessages []chat.Message

type contentPart struct {
	Type     string `json:"type"`
	Text     string `json:"text"`
	ImageURL struct {
		URL string `json:"url"`
	} `json:"image_url"`
}

func (m *requestMessages) UnmarshalJSON(raw []byte) error {
	// Everything but the content decodes as it always did.
	type plain struct {
		chat.Message
		Content json.RawMessage `json:"content"`
	}
	var wire []plain
	if err := json.Unmarshal(raw, &wire); err != nil {
		return err
	}
	out := make([]chat.Message, 0, len(wire))
	for i, w := range wire {
		msg := w.Message
		switch {
		case len(w.Content) == 0 || string(w.Content) == "null":
		case w.Content[0] == '"':
			if err := json.Unmarshal(w.Content, &msg.Content); err != nil {
				return err
			}
		default:
			var parts []contentPart
			if err := json.Unmarshal(w.Content, &parts); err != nil {
				return fmt.Errorf("message %d: content is neither a string nor an array of parts: %w", i, err)
			}
			var text []string
			for _, part := range parts {
				switch part.Type {
				case "text":
					text = append(text, part.Text)
				case "image_url":
					data, err := readImageURL(part.ImageURL.URL)
					if err != nil {
						return fmt.Errorf("message %d: %w", i, err)
					}
					msg.Images = append(msg.Images, data)
				default:
					return fmt.Errorf("message %d holds a content part of type %q, which this server does not read", i, part.Type)
				}
			}
			msg.Content = strings.Join(text, "\n")
		}
		out = append(out, msg)
	}
	*m = requestMessages(out)
	return nil
}

// readImageURL takes the two forms this server accepts, and refuses the one it
// will not: an http URL would make a prompt able to send the server somewhere,
// and a server that fetches what a prompt names is a server that can be aimed.
func readImageURL(url string) ([]byte, error) {
	if strings.HasPrefix(url, "http://") || strings.HasPrefix(url, "https://") {
		return nil, fmt.Errorf("this server does not fetch pictures over the network; send the bytes as a data: URI")
	}
	if strings.HasPrefix(url, "data:") {
		comma := strings.IndexByte(url, ',')
		if comma < 0 {
			return nil, fmt.Errorf("a data: URI with no comma in it")
		}
		if !strings.Contains(url[:comma], ";base64") {
			return nil, fmt.Errorf("a data: URI that is not base64")
		}
		return base64.StdEncoding.DecodeString(url[comma+1:])
	}
	return os.ReadFile(url)
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
	Role      string          `json:"role,omitempty"`
	Content   string          `json:"content"`
	ToolCalls []chat.ToolCall `json:"tool_calls,omitempty"`
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
