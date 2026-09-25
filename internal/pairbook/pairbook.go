// Package pairbook holds the codebooks of golem's pair-trellis tiers, as the
// half-precision bits a kernel holds and a file carries. It is its own package
// because two others need the same numbers and neither may import the other:
// nn decodes with them, and tensors refuses a file written with another table.
// cmd/paircodebook writes the tables.
package pairbook

// Of is a tier's codebook by type name, or nil.
func Of(dtype string) []uint16 {
	switch dtype {
	case "H3G":
		return H3G[:]
	case "H4G":
		return H4G[:]
	}
	return nil
}
