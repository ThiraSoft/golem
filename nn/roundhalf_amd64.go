//go:build amd64

package nn

//go:noescape
func roundHalfAVX2(v *float32, n int)
