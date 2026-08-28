package vk

import (
	_ "embed"
	"unsafe"
)

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

//go:embed shaders/matvec_q4k.spv
var matvecQ4KSPIRV []byte

//go:embed shaders/matvec_q5k.spv
var matvecQ5KSPIRV []byte

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

type SSMBlock struct {
	d *Device

	convWeight *Buffer // [4, 10240]
	convState  *Buffer // [3, 10240]
	convOut    *Buffer // [10240]

	alphaBuf *Buffer // [48]
	betaBuf  *Buffer // [48]
	dtBias   *Buffer // [48]
	ssmA     *Buffer // [48]
	ssmNorm  *Buffer // [128]
	ssmState *Buffer // [48 * 128 * 128]
	ySSM     *Buffer // [6144]

	// Projections
	wQKV   *Buffer // [5120, 10240] Q4_0
	wGate  *Buffer // [5120, 6144] Q4_0
	wAlpha *Buffer // [5120, 48] F32
	wBeta  *Buffer // [5120, 48] F32
	wOut   *Buffer // [6144, 5120] Q5_K or Q4_K

	qkvBuf  *Buffer // [10240]
	gateBuf *Buffer // [6144]
	outBuf  *Buffer // [5120]

	pipeConv   *Pipeline
	setConv    *Set
	pipeScan   *Pipeline
	setScan    *Set
	pipeOutMat *Pipeline
	setOutMat  *Set
}

func NewSSMBlock(
	d *Device,
	convWeight []float32,
	dtBias []float32,
	ssmA []float32,
	ssmNorm []float32,
	wQKVData []byte,
	wGateData []byte,
	wOutData []byte,
	outIsQ5K bool,
	inX *Buffer,
) (*SSMBlock, error) {
	b := &SSMBlock{d: d}
	var err error

	if b.convWeight, err = d.Upload(unsafe.Slice((*byte)(unsafe.Pointer(&convWeight[0])), len(convWeight)*4)); err != nil {
		return nil, err
	}
	if b.convState, err = d.Local(3*10240*4, bufferUsageStorage); err != nil {
		return nil, err
	}
	if b.convOut, err = d.Local(10240*4, bufferUsageStorage); err != nil {
		return nil, err
	}
	if b.alphaBuf, err = d.Local(48*4, bufferUsageStorage); err != nil {
		return nil, err
	}
	if b.betaBuf, err = d.Local(48*4, bufferUsageStorage); err != nil {
		return nil, err
	}
	if b.dtBias, err = d.Upload(unsafe.Slice((*byte)(unsafe.Pointer(&dtBias[0])), len(dtBias)*4)); err != nil {
		return nil, err
	}
	if b.ssmA, err = d.Upload(unsafe.Slice((*byte)(unsafe.Pointer(&ssmA[0])), len(ssmA)*4)); err != nil {
		return nil, err
	}
	if b.ssmNorm, err = d.Upload(unsafe.Slice((*byte)(unsafe.Pointer(&ssmNorm[0])), len(ssmNorm)*4)); err != nil {
		return nil, err
	}
	// 48 heads * 128 * 128 * 4 bytes = 3,145,728 bytes
	if b.ssmState, err = d.Local(48*128*128*4, bufferUsageStorage); err != nil {
		return nil, err
	}
	if b.ySSM, err = d.Local(6144*4, bufferUsageStorage); err != nil {
		return nil, err
	}

	if b.qkvBuf, err = d.Local(10240*4, bufferUsageStorage); err != nil {
		return nil, err
	}
	if b.gateBuf, err = d.Local(6144*4, bufferUsageStorage); err != nil {
		return nil, err
	}
	if b.outBuf, err = d.Local(5120*4, bufferUsageStorage); err != nil {
		return nil, err
	}

	if b.wOut, err = d.Upload(wOutData); err != nil {
		return nil, err
	}

	// 1. Pipeline Conv1D: (weight, qkv, convState, convOut)
	convBuffers := []*Buffer{b.convWeight, b.qkvBuf, b.convState, b.convOut}
	if b.pipeConv, err = d.NewPipeline(ssmConv1dSPIRV, len(convBuffers), uint32(unsafe.Sizeof(ssmConvPush{}))); err != nil {
		return nil, err
	}
	if b.setConv, err = b.pipeConv.NewSet(convBuffers); err != nil {
		return nil, err
	}

	// 2. Pipeline Scan: (convOut, gateZ, alpha, dtBias, ssmA, beta, ssmNorm, ssmState, ySSM)
	scanBuffers := []*Buffer{b.convOut, b.gateBuf, b.alphaBuf, b.dtBias, b.ssmA, b.betaBuf, b.ssmNorm, b.ssmState, b.ySSM}
	if b.pipeScan, err = d.NewPipeline(ssmScanSPIRV, len(scanBuffers), uint32(unsafe.Sizeof(ssmScanPush{}))); err != nil {
		return nil, err
	}
	if b.setScan, err = b.pipeScan.NewSet(scanBuffers); err != nil {
		return nil, err
	}

	// 3. Pipeline Out MatVec (ySSM [6144] -> outBuf [5120])
	outSPIRV := matvecQ4KSPIRV
	if outIsQ5K {
		outSPIRV = matvecQ5KSPIRV
	}
	outBuffers := []*Buffer{b.wOut, b.ySSM, b.outBuf}
	if b.pipeOutMat, err = d.NewPipeline(outSPIRV, len(outBuffers), uint32(unsafe.Sizeof(matvecKPush{}))); err != nil {
		return nil, err
	}
	if b.setOutMat, err = b.pipeOutMat.NewSet(outBuffers); err != nil {
		return nil, err
	}

	return b, nil
}

func (b *SSMBlock) Output() *Buffer {
	return b.outBuf
}
