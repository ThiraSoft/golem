package engine

// Which template writes the conversation.
//
// The one the file carries, when it has one golem/jinja reads: it is the
// checkpoint's own. The engine's, written out in Go, when the file carries
// none or one that does not parse, and whenever the caller asks for it.

import (
	"strings"

	"github.com/ThiraSoft/golem/chat"
	"github.com/ThiraSoft/golem/tensors"
)

// fileTemplate puts the file's template in front of the engine's own, and
// says which of the two renders. prepare may be nil.
func fileTemplate(g *tensors.GGUF, own chat.Template, media *strings.Replacer, prepare func([]chat.Message) []chat.Message) (chat.Template, string) {
	src, err := g.String("tokenizer.chat_template")
	if err != nil || src == "" {
		return own, "built in, the file carries none"
	}
	t, err := chat.NewFileTemplate(src, own, specialTokens(g), media)
	if err != nil {
		return own, "built in, the file's does not parse: " + err.Error()
	}
	if prepare != nil {
		t.Prepare(prepare)
	}
	return t, "the file's"
}

// specialTokens are the pieces a template prints by name.
func specialTokens(g *tensors.GGUF) map[string]string {
	out := map[string]string{}
	pieces, err := g.Strings("tokenizer.ggml.tokens")
	if err != nil {
		return out
	}
	for _, name := range []string{"bos", "eos"} {
		if id, err := g.Uint32("tokenizer.ggml." + name + "_token_id"); err == nil && int(id) < len(pieces) {
			out[name+"_token"] = pieces[id]
		}
	}
	return out
}

// UseBuiltinTemplate sets the file's template aside for the engine's own.
func (m *Model) UseBuiltinTemplate() {
	if t, ok := m.Template.(*chat.FileTemplate); ok {
		m.Template = t.Own()
		m.TemplateFrom = "built in, as asked"
	}
}
