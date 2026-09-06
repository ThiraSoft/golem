module github.com/ThiraSoft/golem

// 1.25 is not a language requirement, it is a linker one. Go 1.23.12 and
// 1.24.9 link this module into a binary that dies before main when cgo is off:
// purego imports dlopen through //go:cgo_import_dynamic, and those two drop the
// libc and libpthread NEEDED entries, so the first call lands on address zero.
//
// It is the go line rather than a toolchain line because of where the rule
// bites. `go install pkg@version` has no main module, so it reads this line and
// nothing else to decide whether to step the toolchain up; a toolchain line is
// read only when the module is the one being built from a checkout.
go 1.25.0

require (
	github.com/ebitengine/purego v0.8.0
	github.com/hajimehoshi/go-mp3 v0.3.4
	github.com/mewkiz/flac v1.0.14
	golang.org/x/image v0.28.0
)

require (
	github.com/icza/bitio v1.1.0 // indirect
	github.com/mewkiz/pkg v0.0.0-20250417130911-3f050ff8c56d // indirect
	github.com/mewpkg/term v0.0.0-20241026122259-37a80af23985 // indirect
)
