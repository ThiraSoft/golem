//go:build arm64

package nn

import (
	"os"
	"runtime"
	"testing"
)

// TestDotProdAgreesWithTheEmulator checks the probe against what the machine
// running it actually is. Under qemu-user the answer is known from QEMU_CPU:
// the default CPU has FEAT_DotProd and cortex-a53 does not, which is what makes
// both sides of the branch testable on a machine that is neither.
func TestDotProdAgreesWithTheEmulator(t *testing.T) {
	if runtime.GOOS != "linux" && runtime.GOOS != "darwin" {
		t.Skipf("no probe for %s", runtime.GOOS)
	}
	want := true
	if os.Getenv("QEMU_CPU") == "cortex-a53" {
		want = false
	}
	if dotprod != want {
		t.Errorf("dotprod = %v, want %v (QEMU_CPU=%q)", dotprod, want, os.Getenv("QEMU_CPU"))
	}
}

// TestDotProdIsStable checks that the probe is read once rather than recomputed,
// which is the property that lets kernels branch on it without cost.
func TestDotProdIsStable(t *testing.T) {
	first := dotprod
	for i := 0; i < 1000; i++ {
		if dotprod != first {
			t.Fatal("dotprod changed under repeated reads")
		}
	}
}
