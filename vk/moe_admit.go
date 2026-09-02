package vk

// Where a copy of an expert lives.

import _ "embed"

//go:generate glslc -O --target-env=vulkan1.1 -fshader-stage=compute shaders/moe_admit.comp -o shaders/moe_admit.spv

//go:embed shaders/moe_admit.spv
var moeAdmitSPIRV []byte

// admitPush is shaders/moe_admit.comp's.
type admitPush struct {
	Experts uint32
	Slots   uint32
	Used    uint32
	Columns uint32
}

//go:generate glslc -O --target-env=vulkan1.1 -fshader-stage=compute shaders/moe_fill.comp -o shaders/moe_fill.spv

//go:embed shaders/moe_fill.spv
var moeFillSPIRV []byte

// fillPush is shaders/moe_fill.comp's. groups is how many workgroups share one
// expert, which the dispatch is sized by and the kernel divides by.
type fillPush struct {
	Words  uint32
	Groups uint32
}
