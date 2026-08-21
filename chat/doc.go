// Package chat holds the shape of a conversation, and the interface an engine
// implements to write one out.
//
// The types here are OpenAI's wire format, not any model's architecture: they
// lived in gemma/ only because that engine needed them first. An engine turns
// them into the text its own checkpoint was trained on, and reads back what it
// wrote.
package chat
