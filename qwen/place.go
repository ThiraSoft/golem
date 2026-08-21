package qwen

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

// Place is one token's cache and its position in it.
type Place struct {
	Cache *Cache
	Pos   int
}

// Run is the places of a run of consecutive positions in one cache, which is
// what reading a prompt and drawing an answer both offer.
func Run(cache *Cache, startPos, n int) []Place {
	at := make([]Place, n)
	for i := range at {
		at[i] = Place{Cache: cache, Pos: startPos + i}
	}
	return at
}
