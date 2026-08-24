//go:build darwin && arm64

package nn

// Every Apple Silicon part has FEAT_DotProd, from the M1 onward, so there is
// no Mac for this to be wrong about and nothing to ask the kernel.
//
// Asking anyway would mean golang.org/x/sys/cpu, whose darwin support arrived
// only in v0.42.0 — earlier versions report false here, silently, which would
// leave every Mac on the portable path with nothing to show why. And v0.42.0
// wants Go 1.25, against a README that promises 1.23 and nothing else.
const dotprod = true
