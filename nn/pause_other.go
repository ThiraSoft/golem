//go:build !amd64

package nn

import "runtime"

// spinPause is one turn of a waiting loop. Without a pause instruction to lean
// on, handing the processor back to the scheduler is the honest thing to do:
// an arithmetic loop would burn the core, and doing it on a shared variable
// would make every waiting core fight over one cache line.
func spinPause() { runtime.Gosched() }
