package gemma

// Watching the routing, without changing it.

// RouteWatch, when it is not nil, is handed every position's chosen experts as
// each mixture block routes, in the order the blocks run.
//
// It exists for one question: how much of a token's expert reads a cache of a
// given size would already hold. Answering that needs the routing and nothing
// else — no card, no cache, and no change to what the model computes — so this
// is a function pointer read once a block rather than a recording built into
// the engine.
//
// **The card does not route here.** shaders/router_logits.comp and
// shaders/router_pick.comp pick the experts on the device and leave them in
// device memory, one block's worth at a time, which no one on this side reads.
// What this hook watches is the processor's router over the same weights and
// the same arithmetic; it is the routing a streamed path would have to predict,
// measured on the path that can be measured.
//
// The slice it is handed is the engine's own and is overwritten by the next
// block. A watcher that keeps it keeps a copy.
var RouteWatch func(ids [][]int32)
