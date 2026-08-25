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

	// The wide form of the same kernel, built from a second SPIR-V over the
	// same layout, so that one Set serves both. Wide says what that means:
	// a kernel that answers several columns in one pass rather than one.
	wideModule uint64
	wide       uint64
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

// Wide compiles a second binary of the same kernel over this pipeline's own
// layout — same bindings, same push block, different code. The sets already
// made for the pipeline reach it without being made again, which is the point:
// there are sixty-five of them on a deep model and they name the same buffers.
//
// What differs in the binary is the number of columns it answers, which is a
// compile-time constant there because the accumulators have to stay in
// registers. shaders/matvec.comp says why.
func (p *Pipeline) Wide(spirv []byte) error {
	smci := shaderModuleCreateInfo{
		sType:    structShaderModuleCreateInfo,
		codeSize: uint64(len(spirv)),
		pCode:    uintptr(unsafe.Pointer(&spirv[0])),
	}
	if err := check("vkCreateShaderModule", vkCreateShaderModule(p.d.dev, &smci, 0, &p.wideModule)); err != nil {
		return err
	}
	name := append([]byte("main"), 0)
	cpci := computePipelineCreateInfo{
		sType: structComputePipelineCreateInfo,
		stage: pipelineShaderStageCreateInfo{
			sType:  structPipelineShaderStageInfo,
			stage:  shaderStageCompute,
			module: p.wideModule,
			pName:  uintptr(unsafe.Pointer(&name[0])),
		},
		layout: p.layout,
	}
	return check("vkCreateComputePipelines", vkCreateComputePipelines(p.d.dev, 0, 1, &cpci, 0, &p.wide))
}

func (p *Pipeline) Close() {
	if p.wide != 0 {
		vkDestroyPipeline(p.d.dev, p.wide, 0)
		p.wide = 0
	}
	if p.wideModule != 0 {
		vkDestroyShaderModule(p.d.dev, p.wideModule, 0)
		p.wideModule = 0
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
	r.dispatch(s, s.p.pipeline, groups, 1, push)
}

// DispatchColumns is Dispatch with a second axis, which the kernels that carry
// one column per workgroup use for exactly that: the norms, the combine, and
// both halves of the attention. A prompt passes one column per position and a
// token passes one, and nothing else about the two differs.
func (r *Recorder) DispatchColumns(s *Set, groups, columns uint32, push unsafe.Pointer) {
	r.dispatch(s, s.p.pipeline, groups, columns, push)
}

// DispatchWide runs the binary Pipeline.Wide compiled. Its columns are inside
// the kernel rather than on an axis — that is the whole difference between a
// prompt that reads the weights once for eight positions and one that reads
// them eight times — so the grid is the same as the narrow form's.
func (r *Recorder) DispatchWide(s *Set, groups uint32, push unsafe.Pointer) {
	r.dispatch(s, s.p.wide, groups, 1, push)
}

func (r *Recorder) dispatch(s *Set, pipeline uint64, groups, columns uint32, push unsafe.Pointer) {
	p := s.p
	vkCmdBindPipeline(r.cb, pipelineBindCompute, pipeline)
	vkCmdBindDescriptorSets(r.cb, pipelineBindCompute, p.layout, 0, 1, &s.set, 0, 0)
	if p.pushBytes > 0 {
		vkCmdPushConstants(r.cb, p.layout, shaderStageCompute, 0, p.pushBytes, push)
	}
	vkCmdDispatch(r.cb, groups, columns, 1)
}

// Barrier separates a dispatch that writes from one that reads what it wrote,
// and covers the copies too: a recording that traces its own intermediates has
// a transfer between two dispatches.
func (r *Recorder) Barrier() {
	b := memoryBarrier{
		sType:         structMemoryBarrier,
		srcAccessMask: accessShaderWrite | accessTransferWrite,
		dstAccessMask: accessShaderRead | accessShaderWrite | accessTransferRead,
	}
	const stages = stageComputeShader | stageTransfer
	vkCmdPipelineBarrier(r.cb, stages, stages, 0, 1, &b, 0, 0, 0, 0)
}

// Copy takes size bytes from the start of src into dst at an offset. It is how
// a recording keeps a waypoint it would otherwise overwrite.
func (r *Recorder) Copy(dst *Buffer, offset int, src *Buffer, size int) {
	r.CopyFrom(dst, offset, src, 0, size)
}

// CopyFrom is Copy from somewhere other than the start of the source, which is
// what a pass carrying several columns needs: the tracer keeps one of them.
func (r *Recorder) CopyFrom(dst *Buffer, offset int, src *Buffer, from, size int) {
	region := bufferCopy{srcOffset: uint64(from), dstOffset: uint64(offset), size: uint64(size)}
	vkCmdCopyBuffer(r.cb, src.handle, dst.handle, 1, &region)
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

// A Program is a command buffer recorded once and submitted many times.
//
// It exists because recording is not free. A token's stack is four hundred
// dispatches and as many barriers, and every one of them is four calls across
// purego into the loader — measured, one and a quarter milliseconds a token of
// CPU beside the card's ten. Nothing about the recording changes between two
// tokens of the same model: the same kernels over the same buffers in the same
// order. What changes is what those buffers hold, and the one thing that used
// to be recorded rather than held — the position, and the range of the cache a
// block may read — moved into a buffer of its own so that this could be true.
// vk/attention.go's SetWhere is that buffer.
type Program struct {
	d  *Device
	cb commandBuffer
}

// Compile records once. The recording is kept until Close.
func (d *Device) Compile(record func(*Recorder)) (*Program, error) {
	p := &Program{d: d}
	cbai := commandBufferAllocateInfo{
		sType:              structCommandBufferAllocateInfo,
		commandPool:        d.cmdPool,
		level:              0,
		commandBufferCount: 1,
	}
	if err := check("vkAllocateCommandBuffers", vkAllocateCommandBuffers(d.dev, &cbai, &p.cb)); err != nil {
		return nil, err
	}
	bi := commandBufferBeginInfo{sType: structCommandBufferBeginInfo}
	if err := check("vkBeginCommandBuffer", vkBeginCommandBuffer(p.cb, &bi)); err != nil {
		return nil, err
	}
	record(&Recorder{cb: p.cb})
	if err := check("vkEndCommandBuffer", vkEndCommandBuffer(p.cb)); err != nil {
		return nil, err
	}
	return p, nil
}

// Run submits the recording and waits, as everything else here does.
func (p *Program) Run() error {
	si := submitInfo{
		sType:              structSubmitInfo,
		commandBufferCount: 1,
		pCommandBuffers:    uintptr(unsafe.Pointer(&p.cb)),
	}
	if err := check("vkQueueSubmit", vkQueueSubmit(p.d.queue, 1, &si, 0)); err != nil {
		return err
	}
	return check("vkQueueWaitIdle", vkQueueWaitIdle(p.d.queue))
}

func (p *Program) Close() {
	if p.cb != 0 {
		vkFreeCommandBuffers(p.d.dev, p.d.cmdPool, 1, &p.cb)
		p.cb = 0
	}
}
