// Package heavy says whether the tests that load this machine may run.
//
// Most of golem's tests are seconds of arithmetic and answer a question about
// correctness. A few are not: they open a checkpoint of tens of gigabytes,
// push it across the bus, or time a kernel over thousands of passes. Those are
// as much a load test as a test — the card and the processor both run flat out,
// and both get hot — and running them all in one go has taken this machine
// down.
//
// So they are off unless asked for. `go test ./...` is the correctness suite
// and stays safe to run at any time; GOLEM_FULL_TEST=1 adds the rest.
//
//	GOLEM_FULL_TEST=1 go test ./vk/          # one package at a time is the way
//
// A test guarded here says why it is heavy, and that reason reaches the skip
// line: a suite that skips half of itself has to say what it skipped.
package heavy

import (
	"os"
	"testing"
)

// On reports whether the heavy tests were asked for.
func On() bool { return os.Getenv("GOLEM_FULL_TEST") != "" }

// Skip leaves the test unless GOLEM_FULL_TEST is set. why is what makes it heavy —
// what it opens, what it moves, or how long it runs.
func Skip(tb testing.TB, why string) {
	tb.Helper()
	if !On() {
		tb.Skipf("%s; GOLEM_FULL_TEST=1 runs it", why)
	}
	if testing.Short() {
		tb.Skipf("%s; -short leaves it out", why)
	}
}
