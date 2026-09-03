package vk

// The fused gate-and-up binaries, held apart from each other.
//
// Every width of every format here is a separate SPIR-V file compiled with a
// different -DCOLUMNS, embedded by name, and handed to Pipeline.Wide. Nothing
// in that chain checks that the file is what its name says: a wide entry built
// at one column has the right descriptor set, the right push constants and the
// right entry point, so it binds, dispatches and answers — filling the first of
// its columns and leaving the rest as the buffer found them.
//
// That is not hypothetical. Three of the four Q4_K binaries in this repository
// were byte-identical to the one-column build, which made every prompt pass
// wider than four columns on a Q4_K_M read one column and invent the others.
// A test that compares the files is the only thing that sees it, because the
// engine cannot: the shapes agree.

import (
	"crypto/sha256"
	"testing"
)

func TestFusedWidthsAreTheirOwnBinaries(t *testing.T) {
	for _, family := range []struct {
		name  string
		spirv map[string][]byte
	}{
		{"Q4_0", map[string][]byte{
			"1": moeGateUpSPIRV, "8": moeGateUpWideSPIRV, "16": moeGateUpMidSPIRV,
		}},
		{"Q4_K", map[string][]byte{
			"1": moeGateUpQ4KSPIRV, "8": moeGateUpQ4KWideSPIRV,
			"16": moeGateUpQ4KMidSPIRV, "32": moeGateUpQ4KWidestSPIRV,
		}},
		{"Q3_K", map[string][]byte{
			"1": moeGateUpQ3KSPIRV, "8": moeGateUpQ3KWideSPIRV,
			"16": moeGateUpQ3KMidSPIRV, "32": moeGateUpQ3KWidestSPIRV,
		}},
		{"Q8_0", map[string][]byte{
			"1": moeGateUpQ80SPIRV, "8": moeGateUpQ80WideSPIRV,
			"16": moeGateUpQ80MidSPIRV, "32": moeGateUpQ80WidestSPIRV,
		}},
	} {
		t.Run(family.name, func(t *testing.T) {
			seen := map[[32]byte]string{}
			for width, spirv := range family.spirv {
				if len(spirv) == 0 {
					t.Fatalf("%s at %s columns is empty", family.name, width)
				}
				sum := sha256.Sum256(spirv)
				if other, ok := seen[sum]; ok {
					t.Fatalf("%s at %s columns is byte for byte the binary at %s columns; "+
						"one of them was built with the wrong -DCOLUMNS and answers only its first column",
						family.name, width, other)
				}
				seen[sum] = width
			}
		})
	}
}
