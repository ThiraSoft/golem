//go:build amd64

package nn

// spinPause is one turn of a waiting loop: eight PAUSE instructions, short
// enough not to hold the memory system and long enough not to saturate the
// counter it watches.
//
// PAUSE is the instruction the processor offers for exactly this: it tells the
// core that the loop it is running is a wait, so that it neither fills the
// pipeline with speculated iterations nor punishes the core that finally writes
// the value being waited on. An arithmetic loop would do neither, and one that
// counted on a package-level variable would have every waiting core writing the
// same cache line — the one thing a waiting loop must not do.
//
//go:noescape
func spinPause()
