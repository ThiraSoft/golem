// Package version says which build of golem this is.
//
// The value is set at link time by the release workflow:
//
//	go build -ldflags "-X github.com/ThiraSoft/golem/internal/version.value=v1.2.3"
//
// A build that was not stamped says so rather than claiming a number, because
// a version nobody set is worse than no version at all when a bug report
// arrives.
package version

import "runtime/debug"

// value is what the linker writes. Empty in a build that was not stamped.
var value string

// String is the version to print. It falls back to the module version the Go
// toolchain records for a `go install` build, and to "devel" for a build from
// a working tree.
func String() string {
	if value != "" {
		return value
	}
	if info, ok := debug.ReadBuildInfo(); ok && info.Main.Version != "" && info.Main.Version != "(devel)" {
		return info.Main.Version
	}
	return "devel"
}
