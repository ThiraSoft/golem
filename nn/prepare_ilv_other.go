//go:build !amd64

package nn

// PrepareIlv reports that it did nothing. The interleaved layout exists to
// suit one kernel, and without that kernel the caller writes the plain order
// and reads it with the plain product.
func PrepareIlv(dst, src []float32, lo, hi float32) bool {
	return false
}
