//go:build !linux && !darwin && arm64

package nn

// Windows, the BSDs, and anything else on arm64. There is a way to ask on each
// of them and no way to test any of them here, so the answer is the one that is
// never wrong about correctness.
const dotprod = false
