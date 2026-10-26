package gemma

// Where one token of a batch goes.
//
// A batch used to be a run of consecutive positions of one conversation, and
// said so with a single starting position. It is now a list of places: each
// token names the cache it writes to and the position it writes at, which is
// what lets one pass over the weights carry tokens belonging to different
// conversations.
//
// Nothing else changes. The arithmetic a token sees depends on its own place
// and on nothing in the rest of the batch, so a mixed batch gives every token
// what it would have been given alone — TestMixedBatchAgreesWithOneAtATime
// holds it to that, bit for bit.

// Place is one token's cache, its position in it, and the furthest position it
// is allowed to see.
//
// Until is that last one. Text is causal and a text token's Until is its own
// position, which is what Run writes. An image is not: Gemma attends to the
// tokens of one picture in both directions, so every token of a span carries
// the span's last position and the widening stops there. It is a wider window,
// not a new mask — a position still reads a contiguous run of the cache, and
// every key of the batch is stored before any score is computed.
type Place struct {
	Cache *Cache
	Pos   int
	Until int
	// Shift and Holes put the token on a branch: one of several continuations
	// of the same prefix, each written after the one before it in the cache.
	// The token is turned, and its window measured, as if it stood Shift
	// positions before Pos, which is where its branch would stand alone, and
	// it does not see the cache over either hole, each from its first
	// position up to its second: the branches it does not continue. An option
	// of a question sees the state, its question's words and itself, which
	// takes two. All are zero on the main line.
	//
	// It is what lets the questions about a state and all their options be
	// weighed in as few passes as their positions fill
	// (cmd/golem-server/weigh.go).
	Shift int
	Holes [2][2]int
}

// visible is the inclusive range of the cache this token's block may read,
// hole and all. The window is the turned position's; the last position is the
// written one.
func (p Place) visible(b BlockConfig) (first, last int) {
	first, last = p.Cache.Visible(b, p.Pos-p.Shift, p.Until-p.Shift)
	return first, last + p.Shift
}

// blind says whether the token does not see position at of the cache.
func (p Place) blind(at int) bool {
	return (at >= p.Holes[0][0] && at < p.Holes[0][1]) || (at >= p.Holes[1][0] && at < p.Holes[1][1])
}

// Run is the places of a run of consecutive positions in one cache, which is
// what reading a prompt and drawing an answer both offer.
func Run(cache *Cache, startPos, n int) []Place {
	at := make([]Place, n)
	for i := range at {
		at[i] = Place{Cache: cache, Pos: startPos + i, Until: startPos + i}
	}
	return at
}

// ForwardSlots is ForwardMixed for a caller that names the caches by index
// rather than by pointer, which is what a package above the engines can do
// without knowing what a cache of this architecture is.
func (m *Model) ForwardSlots(tokens []int32, slots, positions []int) [][]float32 {
	at := make([]Place, len(tokens))
	for i := range tokens {
		at[i] = Place{Cache: m.Slot(slots[i]), Pos: positions[i], Until: positions[i]}
	}
	return m.ForwardMixed(tokens, at)
}
