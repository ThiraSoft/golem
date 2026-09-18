package tha3

import (
	"os"
	"path/filepath"
)

// The five networks, by the name of their file and of their waypoints.
const (
	netEyebrowDecomposer = "eyebrow_decomposer"
	netEyebrowCombiner   = "eyebrow_morphing_combiner"
	netFaceMorpher       = "face_morpher"
	netRotator           = "two_algo_face_body_rotator"
	netEditor            = "editor"
)

// Dir is where the converted weights are: $GOLEM_THA3 when set, else
// ~/.cache/golem/tha3/separable_float, where ref/tha3/README.md puts them.
func Dir() string {
	if d := os.Getenv("GOLEM_THA3"); d != "" {
		return d
	}
	home, _ := os.UserHomeDir()
	return filepath.Join(home, ".cache", "golem", "tha3", "separable_float")
}
