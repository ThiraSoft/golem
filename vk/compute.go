package vk

// A compute pipeline over a fixed set of storage buffers.
//
// Everything a dispatch needs is built once — the module, the layout, the
// descriptor set, the pipeline — and only the push constants and the contents
// of the host-visible buffers change from one call to the next. A kernel that
// runs once per token cannot afford to build any of that per token.

import (
	"unsafe"
)

type Pipeline struct {
	d *Device

	module     uint64
	setLayout  uint64
	layout     uint64
	pipeline   uint64
	pool       uint64
	set        uint64
	pushBytes  uint32
	bufferInfo []descriptorBufferInfo // kept alive for the driver's write
}

// NewPipeline compiles one SPIR-V compute shader and binds the given buffers
// to bindings 0..n-1, in order.
func (d *Device) NewPipeline(spirv []byte, buffers []*Buffer, pushBytes uint32) (*Pipeline, error) {
	p := &Pipeline{d: d, pushBytes: pushBytes}

	smci := shaderModuleCreateInfo{
		sType:    structShaderModuleCreateInfo,
		codeSize: uint64(len(spirv)),
		pCode:    uintptr(unsafe.Pointer(&spirv[0])),
	}
	if err := check("vkCreateShaderModule", vkCreateShaderModule(d.dev, &smci, 0, &p.module)); err != nil {
		return nil, err
	}

	bindings := make([]descriptorSetLayoutBinding, len(buffers))
	for i := range bindings {
		bindings[i] = descriptorSetLayoutBinding{
			binding:         uint32(i),
			descriptorType:  descriptorStorageBuffer,
			descriptorCount: 1,
			stageFlags:      shaderStageCompute,
		}
	}
	dsl := descriptorSetLayoutCreateInfo{
		sType:        structDescriptorSetLayoutInfo,
		bindingCount: uint32(len(bindings)),
		pBindings:    uintptr(unsafe.Pointer(&bindings[0])),
	}
	if err := check("vkCreateDescriptorSetLayout", vkCreateDescriptorSetLayout(d.dev, &dsl, 0, &p.setLayout)); err != nil {
		p.Close()
		return nil, err
	}

	rng := pushConstantRange{stageFlags: shaderStageCompute, offset: 0, size: pushBytes}
	plci := pipelineLayoutCreateInfo{
		sType:          structPipelineLayoutCreateInfo,
		setLayoutCount: 1,
		pSetLayouts:    uintptr(unsafe.Pointer(&p.setLayout)),
	}
	if pushBytes > 0 {
		plci.pushConstantRangeCount = 1
		plci.pPushConstantRanges = uintptr(unsafe.Pointer(&rng))
	}
	if err := check("vkCreatePipelineLayout", vkCreatePipelineLayout(d.dev, &plci, 0, &p.layout)); err != nil {
		p.Close()
		return nil, err
	}

	name := append([]byte("main"), 0)
	cpci := computePipelineCreateInfo{
		sType: structComputePipelineCreateInfo,
		stage: pipelineShaderStageCreateInfo{
			sType:  structPipelineShaderStageInfo,
			stage:  shaderStageCompute,
			module: p.module,
			pName:  uintptr(unsafe.Pointer(&name[0])),
		},
		layout: p.layout,
	}
	if err := check("vkCreateComputePipelines", vkCreateComputePipelines(d.dev, 0, 1, &cpci, 0, &p.pipeline)); err != nil {
		p.Close()
		return nil, err
	}

	size := descriptorPoolSize{kind: descriptorStorageBuffer, count: uint32(len(buffers))}
	dpci := descriptorPoolCreateInfo{
		sType:         structDescriptorPoolCreateInfo,
		maxSets:       1,
		poolSizeCount: 1,
		pPoolSizes:    uintptr(unsafe.Pointer(&size)),
	}
	if err := check("vkCreateDescriptorPool", vkCreateDescriptorPool(d.dev, &dpci, 0, &p.pool)); err != nil {
		p.Close()
		return nil, err
	}
	dsai := descriptorSetAllocateInfo{
		sType:              structDescriptorSetAllocateInfo,
		descriptorPool:     p.pool,
		descriptorSetCount: 1,
		pSetLayouts:        uintptr(unsafe.Pointer(&p.setLayout)),
	}
	if err := check("vkAllocateDescriptorSets", vkAllocateDescriptorSets(d.dev, &dsai, &p.set)); err != nil {
		p.Close()
		return nil, err
	}

	p.bufferInfo = make([]descriptorBufferInfo, len(buffers))
	writes := make([]writeDescriptorSet, len(buffers))
	for i, b := range buffers {
		p.bufferInfo[i] = descriptorBufferInfo{buffer: b.handle, offset: 0, rng: b.size}
		writes[i] = writeDescriptorSet{
			sType:           structWriteDescriptorSet,
			dstSet:          p.set,
			dstBinding:      uint32(i),
			descriptorCount: 1,
			descriptorType:  descriptorStorageBuffer,
			pBufferInfo:     uintptr(unsafe.Pointer(&p.bufferInfo[i])),
		}
	}
	vkUpdateDescriptorSets(d.dev, uint32(len(writes)), &writes[0], 0, 0)
	return p, nil
}

// Dispatch runs the shader over groups workgroups and waits for it.
func (p *Pipeline) Dispatch(groups uint32, push unsafe.Pointer) error {
	return p.DispatchTimes(groups, push, 1)
}

// DispatchTimes runs the same dispatch n times inside one submission, each
// waiting on the last through a memory barrier.
//
// One dispatch tells you what a token costs, round trip and all. It does not
// tell you what the kernel costs, because a card handed five milliseconds of
// work and then left alone does not raise its clocks: measured one at a time
// this head runs at half the card's frequency and a fifth of its power. Both
// numbers are worth having and they are not the same number.
func (p *Pipeline) DispatchTimes(groups uint32, push unsafe.Pointer, n int) error {
	return p.d.run(func(cb commandBuffer) {
		vkCmdBindPipeline(cb, pipelineBindCompute, p.pipeline)
		vkCmdBindDescriptorSets(cb, pipelineBindCompute, p.layout, 0, 1, &p.set, 0, 0)
		if p.pushBytes > 0 {
			vkCmdPushConstants(cb, p.layout, shaderStageCompute, 0, p.pushBytes, push)
		}
		barrier := memoryBarrier{
			sType:         structMemoryBarrier,
			srcAccessMask: accessShaderWrite,
			dstAccessMask: accessShaderRead | accessShaderWrite,
		}
		for i := 0; i < n; i++ {
			if i > 0 {
				vkCmdPipelineBarrier(cb, stageComputeShader, stageComputeShader, 0, 1, &barrier, 0, 0, 0, 0)
			}
			vkCmdDispatch(cb, groups, 1, 1)
		}
	})
}

func (p *Pipeline) Close() {
	if p.pool != 0 {
		vkDestroyDescriptorPool(p.d.dev, p.pool, 0)
		p.pool = 0
	}
	if p.pipeline != 0 {
		vkDestroyPipeline(p.d.dev, p.pipeline, 0)
		p.pipeline = 0
	}
	if p.layout != 0 {
		vkDestroyPipelineLayout(p.d.dev, p.layout, 0)
		p.layout = 0
	}
	if p.setLayout != 0 {
		vkDestroyDescriptorSetLayout(p.d.dev, p.setLayout, 0)
		p.setLayout = 0
	}
	if p.module != 0 {
		vkDestroyShaderModule(p.d.dev, p.module, 0)
		p.module = 0
	}
}
