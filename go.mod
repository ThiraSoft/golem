module github.com/ThiraSoft/golem

// 1.23.2 is what the dependencies ask for, and nothing here asks for more.
//
// It used to say 1.25, for a linker reason rather than a language one: purego
// v0.8.0 imported dlopen through //go:cgo_import_dynamic in a shape that Go
// 1.23.12 and 1.24.9 linked into a binary keeping only libdl.so.2 as NEEDED.
// On glibc 2.34 and later dlopen is not in libdl any more, so the first call
// landed on address zero and the binary died before main. purego v0.10.0
// emits the libc and libpthread entries too, on every toolchain from 1.23.12
// up, which is why the floor could come back down.
//
// Do not step purego to v0.11.0 to be current: its own go.mod says 1.25.0,
// which puts this module's floor straight back where it was.
go 1.23.2

require (
	github.com/ebitengine/purego v0.10.0
	github.com/hajimehoshi/go-mp3 v0.3.4
	github.com/mewkiz/flac v1.0.14
	golang.org/x/image v0.28.0
)

require (
	github.com/icza/bitio v1.1.0 // indirect
	github.com/mewkiz/pkg v0.0.0-20250417130911-3f050ff8c56d // indirect
	github.com/mewpkg/term v0.0.0-20241026122259-37a80af23985 // indirect
)
