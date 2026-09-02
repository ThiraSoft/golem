package vk

import _ "embed"

//go:embed shaders/ssm_conv1d.spv
var ssmConv1dSPIRV []byte

// The two kernels a gated delta net is: the causal convolution over its q, k
// and v channels, and the recurrence itself. Both walk the columns of a pass
// inside one workgroup, because both carry state from one position to the next.
//
//go:generate glslc -O --target-env=vulkan1.1 -fshader-stage=compute shaders/ssm_conv1d.comp -o shaders/ssm_conv1d.spv
//go:generate glslc -O --target-env=vulkan1.1 -fshader-stage=compute shaders/ssm_scan.comp -o shaders/ssm_scan.spv
//go:generate glslc -O --target-env=vulkan1.1 -fshader-stage=compute shaders/ssm_qknorm.comp -o shaders/ssm_qknorm.spv
//go:generate glslc -O --target-env=vulkan1.1 -fshader-stage=compute shaders/ssm_gate.comp -o shaders/ssm_gate.spv

//go:embed shaders/ssm_scan.spv
var ssmScanSPIRV []byte

// The two normalisations the recurrence used to carry, and the reason it could
// not be split. See the header of shaders/ssm_scan.comp.
//
//go:embed shaders/ssm_qknorm.spv
var ssmQKNormSPIRV []byte

//go:embed shaders/ssm_gate.spv
var ssmGateSPIRV []byte

// ssmQKNormPush and ssmGatePush are those kernels' blocks.
type ssmQKNormPush struct {
	ConvDim uint32
	Eps     float32
}

type ssmGatePush struct {
	Inner uint32
	Eps   float32
}

type ssmConvPush struct {
	Channels uint32
	Kernel   uint32
	Columns  uint32
	SnapAt   uint32
}

// ssmScanPush is shaders/ssm_scan.comp's block, and the order of these fields
// is the order of that block's. A field out of place here is not a compile
// error on either side: the kernel reads whatever integer lands at the offset
// it expects, and answers with the wrong strides.
type ssmScanPush struct {
	NumHeads  uint32
	StateSize uint32
	Eps       float32
	Scale     float32
	Columns   uint32
	SnapAt    uint32
	ConvDim   uint32
	Inner     uint32
	Rank      uint32
}

type matvecKPush struct {
	Dim uint32
	FFN uint32
	// Col is the first column of the pass a dispatch answers. These kernels
	// are built for a fixed number of columns, so a pass wider than the widest
	// binary is run as several dispatches at an offset. See QwenPipeline's
	// product.
	Col uint32
}
