package engine

// A prompt with media spliced into it, as a command handles one.
//
// The type is an interface because the two engines that have media do not
// build the same thing: Gemma carries per-layer inputs and spans it attends to
// in both directions, and Qwen3.8 carries a position with three axes. A
// command needs neither — it needs to know how long a prompt is, where it may
// be cut, and how to cut it — so that is what this says, and the rest stays
// each engine's own.

// Prompt is one engine's tokenized conversation with its pictures already in
// it. It is made by Media.Prompt and handed back to Media.ForwardPrompt.
type Prompt interface {
	// Len is how many positions it occupies, which is what a caller advances
	// its cache by.
	Len() int
	// Boundary is the largest cut no further than want that a pass may end
	// on. An engine that must not cut a picture in half says so here; one
	// that may cut anywhere returns want.
	Boundary(from, want int) int
	// Slice is the run [from, to) as a prompt of its own.
	Slice(from, to int) Prompt
	// Tokens is what the cache holds, media placeholders included. A caller
	// compares it against what a slot already holds to find the divergence.
	Tokens() []int32
	// Embeds is the row given at each position, and nil where the token table
	// is read instead. A prompt with no picture returns nil.
	Embeds() [][]float32
	// Extras are what one engine's pass needs and the other's does not, for a
	// slice fed at startPos. Gemma fills the per-layer lookup and how far
	// forward each position may look; Qwen3.8 fills the three axes its
	// rotation turns by. Each leaves the other's nil, and a batch carries both
	// because a batch is one pass whatever its conversations hold.
	Extras(startPos int) (ple []int32, until []int, axes [][3]int)
}

// TextPrompt is a prompt with nothing spliced into it, which every engine
// reads the same way. It is what a conversation carrying no picture becomes,
// and it belongs here rather than in an engine because none of them owns it.
func TextPrompt(ids []int32) Prompt { return textPrompt(ids) }

type textPrompt []int32

func (t textPrompt) Len() int              { return len(t) }
func (t textPrompt) Tokens() []int32       { return t }
func (t textPrompt) Embeds() [][]float32   { return nil }
func (t textPrompt) Slice(a, b int) Prompt { return textPrompt(t[a:b]) }

// Extras are what text needs, which is nothing: every engine reads a position
// with no picture in it the same way.
func (t textPrompt) Extras(startPos int) ([]int32, []int, [][3]int) { return nil, nil, nil }

// Boundary is want, clamped: text may be cut anywhere.
func (t textPrompt) Boundary(from, want int) int {
	if want > len(t) {
		return len(t)
	}
	return want
}
