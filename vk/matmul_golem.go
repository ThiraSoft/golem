package vk

// The tiled golem product: the same weights as vk/golem.go's mat-vec, read once
// for a whole tile of the answer instead of once per column of the pass.
//
// shaders/matmul_golem.comp says what the shape is and why a prompt wants it.
// What belongs here is the arithmetic the host has to agree with the shader
// about — the tile, and the workgroup count that covers the answer with it.
// Both are said twice, and left disagreeing the dispatch covers a fraction of
// the rows and leaves the rest zero, which reads as a speed-up. That trap has
// been sprung twice in this directory already; vk/matmul.go's header names both.

import (
	_ "embed"
	"os"
	"unsafe"

	"github.com/ThiraSoft/golem/nn"
)

//go:generate glslc -O -DKBITS=3 -DCOLUMNS=32 --target-env=vulkan1.1 -fshader-stage=compute shaders/matmul_golem.comp -o shaders/matmul_t3g_32.spv
//go:generate glslc -O -DKBITS=3 -DCOLUMNS=64 --target-env=vulkan1.1 -fshader-stage=compute shaders/matmul_golem.comp -o shaders/matmul_t3g_64.spv
//go:generate glslc -O -DKBITS=4 -DCOLUMNS=32 --target-env=vulkan1.1 -fshader-stage=compute shaders/matmul_golem.comp -o shaders/matmul_t4g_32.spv
//go:generate glslc -O -DKBITS=4 -DCOLUMNS=64 --target-env=vulkan1.1 -fshader-stage=compute shaders/matmul_golem.comp -o shaders/matmul_t4g_64.spv
//go:generate glslc -O -DKBITS=5 -DCOLUMNS=32 --target-env=vulkan1.1 -fshader-stage=compute shaders/matmul_golem.comp -o shaders/matmul_t5g_32.spv
//go:generate glslc -O -DKBITS=5 -DCOLUMNS=64 --target-env=vulkan1.1 -fshader-stage=compute shaders/matmul_golem.comp -o shaders/matmul_t5g_64.spv

//go:embed shaders/matmul_t3g_32.spv
var matmulT3G32SPIRV []byte

//go:embed shaders/matmul_t3g_64.spv
var matmulT3G64SPIRV []byte

//go:embed shaders/matmul_t4g_32.spv
var matmulT4G32SPIRV []byte

//go:embed shaders/matmul_t4g_64.spv
var matmulT4G64SPIRV []byte

//go:embed shaders/matmul_t5g_32.spv
var matmulT5G32SPIRV []byte

//go:embed shaders/matmul_t5g_64.spv
var matmulT5G64SPIRV []byte

// GolemTiledWidths are the pass widths the tiled product is built for, and they
// begin where GolemWidths stops. A pass narrower than the tile has nothing to
// tile: the mat-vec is the right shape for it, and vk/golem.go says why eight
// is where that shape ends.
var GolemTiledWidths = []int{32, 64}

// golemAllWidths is every pass width the golem kernels answer, narrowest
// first: the mat-vec's, then the tiled product's.
//
// GOLEM_NO_TILE takes the tiled widths back out, which leaves the mat-vec
// answering a prompt eight columns at a time as it did before this file
// existed. It is there so that the two shapes can be read off one binary in
// one process: a prefill measured in two builds is a measurement of the card's
// clock, which vk/golemtune.go's header says at length.
func golemAllWidths() []int {
	out := make([]int, 0, len(GolemWidths)+len(GolemTiledWidths))
	out = append(out, GolemWidths...)
	if !golemTiled() {
		return out
	}
	return append(out, GolemTiledWidths...)
}

// golemTiled is whether a prompt goes through the tiled product. It is the one
// place GOLEM_NO_TILE is read, because the width of a pass has to follow the
// shape that answers it: vk/qwen_pipeline.go's golemWidestPass is two hundred
// and fifty-six only for the tile.
func golemTiled() bool { return os.Getenv("GOLEM_NO_TILE") == "" }

// golemTileRows is shaders/matmul_golem.comp's BM and golemTileCols its BN,
// which is clamped to the width of the pass. A workgroup answers one of those
// tiles.
const (
	golemTileRows = 64
	golemTileCols = 32
)

// golemTiledGroups is how many workgroups a tiled pass of that width needs to
// cover that many rows: a grid of rows over BM by columns over BN.
func golemTiledGroups(rows, columns int) uint32 {
	block := min(golemTileCols, columns)
	cols := columns / block
	return uint32((rows+golemTileRows-1)/golemTileRows) * uint32(cols)
}

// golemTiledSPIRV is one binary a tier and a width, or nothing where the tier
// has none.
func golemTiledSPIRV(q nn.Quant) (map[int][]byte, bool) {
	switch q {
	case nn.T3G:
		return map[int][]byte{32: matmulT3G32SPIRV, 64: matmulT3G64SPIRV}, true
	case nn.T4G:
		return map[int][]byte{32: matmulT4G32SPIRV, 64: matmulT4G64SPIRV}, true
	case nn.T5G:
		return map[int][]byte{32: matmulT5G32SPIRV, 64: matmulT5G64SPIRV}, true
	}
	return nil, false
}

// golemPushSize is the block both kernels take, and they take the same one so
// that a matrix binds one descriptor set per width whichever shape reads it.
var golemPushSize = uint32(unsafe.Sizeof(golemPush{}))
