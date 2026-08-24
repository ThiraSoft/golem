package vk

// Compute pipelines, the descriptor sets that point them at buffers, and the
// recorder that puts several of them in one submission.
//
// The three are separate because they have three lifetimes. A pipeline is the
// compiled shader and is built once for the whole model. A set is one binding
// of that shader to a particular group of buffers, and there is one per block,
// because thirty blocks hold thirty different sets of expert weights and share
// the kernel that reads them. A recording lasts one token.
//
// The separation is what keeps the round trips down. One submission costs
// sixty-three microseconds whatever is in it, which is nothing beside a token
// and a great deal beside a dispatch that takes a hundred: a design that
// crosses to the card twice per block would spend four milliseconds a token
// asking rather than computing. So a block's two dispatches go in together,
// with a barrier between them.

import (
	"unsafe"
)

// A Pipeline is one compiled compute shader and the layout of the buffers it
// reads. It holds no buffers itself; a Set does that.
type Pipeline struct {
	d *Device

	module    uint64
	setLayout uint64
	layout    uint64
	pipeline  uint64
	pushBytes uint32
	bindings  int
}

// NewPipeline compiles one SPIR-V compute shader that reads the given number
// of storage buffers, at bindings 0..bindings-1.
func (d *Device) NewPipeline(spirv []byte, bindings int, pushBytes uint32) (*Pipeline, error) {
	p := &Pipeline{d: d, pushBytes: pushBytes, bindings: bindings}

	smci := shaderModuleCreateInfo{
		sType:    structShaderModuleCreateInfo,
		codeSize: uint64(len(spirv)),
		pCode:    uintptr(unsafe.Pointer(&spirv[0])),
	}
	if err := check("vkCreateShaderModule", vkCreateShaderModule(d.dev, &smci, 0, &p.module)); err != nil {
		return nil, err
	}

	layout := make([]descriptorSetLayoutBinding, bindings)
	for i := range layout {
		layout[i] = descriptorSetLayoutBinding{
			binding:         uint32(i),
			descriptorType:  descriptorStorageBuffer,
			descriptorCount: 1,
			stageFlags:      shaderStageCompute,
		}
	}
	dsl := descriptorSetLayoutCreateInfo{
		sType:        structDescriptorSetLayoutInfo,
		bindingCount: uint32(len(layout)),
		pBindings:    uintptr(unsafe.Pointer(&layout[0])),
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
	return p, nil
}

func (p *Pipeline) Close() {
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

// A Set points one pipeline at one group of buffers.
type Set struct {
	p    *Pipeline
	pool uint64
	set  uint64
	info []descriptorBufferInfo // kept alive for the driver's write
}

// NewSet binds buffers to bindings 0..n-1, in order.
func (p *Pipeline) NewSet(buffers []*Buffer) (*Set, error) {
	if len(buffers) != p.bindings {
		return nil, errBindings(len(buffers), p.bindings)
	}
	s := &Set{p: p}

	size := descriptorPoolSize{kind: descriptorStorageBuffer, count: uint32(len(buffers))}
	dpci := descriptorPoolCreateInfo{
		sType:         structDescriptorPoolCreateInfo,
		maxSets:       1,
		poolSizeCount: 1,
		pPoolSizes:    uintptr(unsafe.Pointer(&size)),
	}
	if err := check("vkCreateDescriptorPool", vkCreateDescriptorPool(p.d.dev, &dpci, 0, &s.pool)); err != nil {
		return nil, err
	}
	dsai := descriptorSetAllocateInfo{
		sType:              structDescriptorSetAllocateInfo,
		descriptorPool:     s.pool,
		descriptorSetCount: 1,
		pSetLayouts:        uintptr(unsafe.Pointer(&p.setLayout)),
	}
	if err := check("vkAllocateDescriptorSets", vkAllocateDescriptorSets(p.d.dev, &dsai, &s.set)); err != nil {
		s.Close()
		return nil, err
	}

	s.info = make([]descriptorBufferInfo, len(buffers))
	writes := make([]writeDescriptorSet, len(buffers))
	for i, b := range buffers {
		s.info[i] = descriptorBufferInfo{buffer: b.handle, offset: 0, rng: b.size}
		writes[i] = writeDescriptorSet{
			sType:           structWriteDescriptorSet,
			dstSet:          s.set,
			dstBinding:      uint32(i),
			descriptorCount: 1,
			descriptorType:  descriptorStorageBuffer,
			pBufferInfo:     uintptr(unsafe.Pointer(&s.info[i])),
		}
	}
	vkUpdateDescriptorSets(p.d.dev, uint32(len(writes)), &writes[0], 0, 0)
	return s, nil
}

func (s *Set) Close() {
	if s.pool != 0 {
		vkDestroyDescriptorPool(s.p.d.dev, s.pool, 0)
		s.pool = 0
	}
}

// A Recorder is one command buffer being written. Dispatch adds a shader run;
// Barrier makes everything after it wait for everything before.
type Recorder struct{ cb commandBuffer }

// Dispatch runs a set's pipeline over groups workgroups.
func (r *Recorder) Dispatch(s *Set, groups uint32, push unsafe.Pointer) {
	p := s.p
	vkCmdBindPipeline(r.cb, pipelineBindCompute, p.pipeline)
	vkCmdBindDescriptorSets(r.cb, pipelineBindCompute, p.layout, 0, 1, &s.set, 0, 0)
	if p.pushBytes > 0 {
		vkCmdPushConstants(r.cb, p.layout, shaderStageCompute, 0, p.pushBytes, push)
	}
	vkCmdDispatch(r.cb, groups, 1, 1)
}

// Barrier separates a dispatch that writes from one that reads what it wrote.
func (r *Recorder) Barrier() {
	b := memoryBarrier{
		sType:         structMemoryBarrier,
		srcAccessMask: accessShaderWrite,
		dstAccessMask: accessShaderRead | accessShaderWrite,
	}
	vkCmdPipelineBarrier(r.cb, stageComputeShader, stageComputeShader, 0, 1, &b, 0, 0, 0, 0)
}

// Submit records one command buffer, runs it, and waits. Everything here is
// synchronous: the caller wants the answer, not a pipeline.
func (d *Device) Submit(record func(*Recorder)) error {
	return d.run(func(cb commandBuffer) { record(&Recorder{cb: cb}) })
}

// Dispatch is one set, once, in a submission of its own.
func (s *Set) Dispatch(groups uint32, push unsafe.Pointer) error {
	return s.p.d.Submit(func(r *Recorder) { r.Dispatch(s, groups, push) })
}

// DispatchTimes runs the same dispatch n times inside one submission, each
// waiting on the last.
//
// One dispatch tells you what a token costs, round trip and all. It does not
// tell you what the kernel costs, because a card handed a few milliseconds of
// work and then left alone does not raise its clocks: measured one at a time
// the logit head runs the card at half its frequency and a fifth of its power.
// Both numbers are worth having and they are not the same number.
func (s *Set) DispatchTimes(groups uint32, push unsafe.Pointer, n int) error {
	return s.p.d.Submit(func(r *Recorder) {
		for i := 0; i < n; i++ {
			if i > 0 {
				r.Barrier()
			}
			r.Dispatch(s, groups, push)
		}
	})
}
