//go:build !amd64

package nn

func dotPQ2_0(w []byte, a *ternaryColumn, n int) float32  { return dotPQ2_0Go(w, a, n) }
func dotPTQ1_0(w []byte, a *ternaryColumn, n int) float32 { return dotPTQ1_0Go(w, a, n) }
