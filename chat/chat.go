package chat

// Message is one turn of the conversation. Role is "system", "developer",
// "user", "assistant", "model" or "tool". What an engine makes of those names
// is the engine's business: Gemma's template renames "assistant" to "model",
// Qwen's does not.
type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
	// Name is the function a tool result came from.
	Name string `json:"name,omitempty"`
	// ToolCalls are the calls a model turn made.
	ToolCalls []ToolCall `json:"tool_calls,omitempty"`
	// ToolCallID ties a tool result to the call it answers.
	ToolCallID string `json:"tool_call_id,omitempty"`
}

// Options are what every template reads. What only one checkpoint spells —
// Gemma's empty thought channel, for instance — belongs to that engine's
// template value and not here.
type Options struct {
	// EnableThinking asks for the reasoning channel to be opened.
	EnableThinking bool
	// Tools are the functions declared to the model.
	Tools []Tool
	// AddGenerationPrompt appends the empty assistant turn the model is meant
	// to continue. A caller rendering a conversation for training rather than
	// for generation leaves it false.
	AddGenerationPrompt bool
}

// Template writes a conversation the way one checkpoint was trained to read
// it, and reads back the calls the model wrote.
//
// Three methods, because three are what the commands consume. A fourth would
// mean a command has learned something about an engine.
type Template interface {
	// Render writes the whole conversation, leading marker included. The
	// caller encodes the result with addBOS false and parseSpecial true.
	Render(msgs []Message, opt Options) (string, error)
	// ParseCalls splits an answer into the prose before the first call and the
	// calls that followed. It refuses what it cannot read rather than
	// returning half a call, because a half-read call would be executed.
	ParseCalls(text string) (before string, calls []ToolCall, err error)
	// CallOpen is where prose stops and a call begins. A streaming server
	// watches for it to stop sending text and start holding a call back.
	CallOpen() string
}
