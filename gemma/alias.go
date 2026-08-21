package gemma

import "github.com/ThiraSoft/golem/chat"

// The conversation types moved to chat/, where neither engine owns them.
// These aliases are what keeps the rest of the package, and its tests, from
// having to move in the same change.
type (
	Message  = chat.Message
	Tool     = chat.Tool
	Schema   = chat.Schema
	ToolCall = chat.ToolCall
)
