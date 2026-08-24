package vk

// Opening a compute device, and the memory that lives on it.

import (
	"fmt"
	"unsafe"
)

// A Device is one Vulkan compute queue and the pools that feed it. It is not
// safe for concurrent use: one dispatch is in flight at a time, which is what
// a logit head submitted once per token needs.
type Device struct {
	inst  instance
	phys  physicalDevice
	dev   device
	queue queue

	family  uint32
	memory  physicalDeviceMemoryProperties
	cmdPool uint64
	cmd     commandBuffer
}

// Open finds the first device with a compute queue and takes it.
func Open() (*Device, error) {
	if err := load(); err != nil {
		return nil, err
	}
	d := &Device{}

	ici := instanceCreateInfo{sType: structInstanceCreateInfo}
	if err := check("vkCreateInstance", vkCreateInstance(&ici, 0, &d.inst)); err != nil {
		return nil, err
	}

	var count uint32
	if err := check("vkEnumeratePhysicalDevices", vkEnumeratePhysicalDevices(d.inst, &count, nil)); err != nil {
		d.Close()
		return nil, err
	}
	if count == 0 {
		d.Close()
		return nil, fmt.Errorf("vk: no Vulkan device")
	}
	devices := make([]physicalDevice, count)
	if err := check("vkEnumeratePhysicalDevices", vkEnumeratePhysicalDevices(d.inst, &count, &devices[0])); err != nil {
		d.Close()
		return nil, err
	}

	found := false
	for _, p := range devices {
		var n uint32
		vkGetPhysicalDeviceQueueFamilyProps(p, &n, nil)
		if n == 0 {
			continue
		}
		families := make([]queueFamilyProperties, n)
		vkGetPhysicalDeviceQueueFamilyProps(p, &n, &families[0])
		for i, f := range families {
			if f.queueFlags&queueCompute != 0 {
				d.phys, d.family, found = p, uint32(i), true
				break
			}
		}
		if found {
			break
		}
	}
	if !found {
		d.Close()
		return nil, fmt.Errorf("vk: no compute queue on any device")
	}
	vkGetPhysicalDeviceMemoryProperties(d.phys, &d.memory)

	priority := float32(1)
	qci := deviceQueueCreateInfo{
		sType:            structDeviceQueueCreateInfo,
		queueFamilyIndex: d.family,
		queueCount:       1,
		pQueuePriorities: uintptr(unsafe.Pointer(&priority)),
	}
	dci := deviceCreateInfo{
		sType:                structDeviceCreateInfo,
		queueCreateInfoCount: 1,
		pQueueCreateInfos:    uintptr(unsafe.Pointer(&qci)),
	}
	if err := check("vkCreateDevice", vkCreateDevice(d.phys, &dci, 0, &d.dev)); err != nil {
		d.Close()
		return nil, err
	}
	vkGetDeviceQueue(d.dev, d.family, 0, &d.queue)

	cpci := commandPoolCreateInfo{
		sType:            structCommandPoolCreateInfo,
		flags:            0x2, // reset individual command buffers
		queueFamilyIndex: d.family,
	}
	if err := check("vkCreateCommandPool", vkCreateCommandPool(d.dev, &cpci, 0, &d.cmdPool)); err != nil {
		d.Close()
		return nil, err
	}
	cbai := commandBufferAllocateInfo{
		sType:              structCommandBufferAllocateInfo,
		commandPool:        d.cmdPool,
		level:              0,
		commandBufferCount: 1,
	}
	if err := check("vkAllocateCommandBuffers", vkAllocateCommandBuffers(d.dev, &cbai, &d.cmd)); err != nil {
		d.Close()
		return nil, err
	}
	return d, nil
}

// Close releases the device. Buffers and pipelines built on it must be closed
// first.
func (d *Device) Close() {
	if d.cmdPool != 0 {
		vkDestroyCommandPool(d.dev, d.cmdPool, 0)
		d.cmdPool = 0
	}
	if d.dev != 0 {
		vkDestroyDevice(d.dev, 0)
		d.dev = 0
	}
	if d.inst != 0 {
		vkDestroyInstance(d.inst, 0)
		d.inst = 0
	}
}

// memoryTypeFor picks a memory type the requirements allow and that carries
// every wanted property.
func (d *Device) memoryTypeFor(bits uint32, want uint32) (uint32, error) {
	for i := uint32(0); i < d.memory.memoryTypeCount; i++ {
		if bits&(1<<i) == 0 {
			continue
		}
		if d.memory.memoryTypes[i].propertyFlags&want == want {
			return i, nil
		}
	}
	return 0, fmt.Errorf("vk: no memory type with properties %#x", want)
}

// A Buffer is one allocation and the VkBuffer bound to it.
type Buffer struct {
	d      *Device
	handle uint64
	mem    uint64
	size   uint64

	// mapped is the driver's address for a host-visible allocation. It is kept
	// as a pointer rather than an integer so that nothing here ever converts an
	// integer back into a pointer: vkMapMemory is handed the address of this
	// field, and the slices below are taken from it directly.
	mapped unsafe.Pointer
}

// newBuffer allocates size bytes with the given usage and memory properties.
func (d *Device) newBuffer(size uint64, usage uint32, props uint32) (*Buffer, error) {
	b := &Buffer{d: d, size: size}
	bci := bufferCreateInfo{sType: structBufferCreateInfo, size: size, usage: usage}
	if err := check("vkCreateBuffer", vkCreateBuffer(d.dev, &bci, 0, &b.handle)); err != nil {
		return nil, err
	}
	var req memoryRequirements
	vkGetBufferMemoryRequirements(d.dev, b.handle, &req)
	kind, err := d.memoryTypeFor(req.memoryTypeBits, props)
	if err != nil {
		b.Close()
		return nil, err
	}
	mai := memoryAllocateInfo{sType: structMemoryAllocateInfo, allocationSize: req.size, memoryTypeIndex: kind}
	if err := check("vkAllocateMemory", vkAllocateMemory(d.dev, &mai, 0, &b.mem)); err != nil {
		b.Close()
		return nil, err
	}
	if err := check("vkBindBufferMemory", vkBindBufferMemory(d.dev, b.handle, b.mem, 0)); err != nil {
		b.Close()
		return nil, err
	}
	if props&memoryHostVisible != 0 {
		if err := check("vkMapMemory", vkMapMemory(d.dev, b.mem, 0, ^uint64(0), 0, (*uintptr)(unsafe.Pointer(&b.mapped)))); err != nil {
			b.Close()
			return nil, err
		}
	}
	return b, nil
}

// Host allocates a buffer the CPU writes and the shader reads directly. It is
// what the small per-token inputs use: an activation is a few kilobytes and
// staging it through device memory would cost more than reading it over the
// bus once.
func (d *Device) Host(size int, usage uint32) (*Buffer, error) {
	return d.newBuffer(uint64(size), usage, memoryHostVisible|memoryHostCoherent)
}

// Bytes is the mapped buffer as a slice. It panics on a buffer that lives in
// device memory, which has no address on this side.
func (b *Buffer) Bytes() []byte {
	if b.mapped == nil {
		panic("vk: buffer is not host visible")
	}
	return unsafe.Slice((*byte)(b.mapped), b.size)
}

// Floats is the mapped buffer read as float32.
func (b *Buffer) Floats() []float32 {
	if b.mapped == nil {
		panic("vk: buffer is not host visible")
	}
	return unsafe.Slice((*float32)(b.mapped), b.size/4)
}

// Close releases the buffer and its memory.
func (b *Buffer) Close() {
	if b.mapped != nil {
		vkUnmapMemory(b.d.dev, b.mem)
		b.mapped = nil
	}
	if b.mem != 0 {
		vkFreeMemory(b.d.dev, b.mem, 0)
		b.mem = 0
	}
	if b.handle != 0 {
		vkDestroyBuffer(b.d.dev, b.handle, 0)
		b.handle = 0
	}
}

// Upload copies data into device-local memory through a staging buffer. This
// is the load-time path for a weight tensor: it runs once, and afterwards the
// weights are read at the speed of the card's own memory rather than the bus.
func (d *Device) Upload(data []byte) (*Buffer, error) {
	dst, err := d.newBuffer(uint64(len(data)), bufferUsageStorage|bufferUsageTransferDst, memoryDeviceLocal)
	if err != nil {
		return nil, err
	}
	// A staging buffer of the whole tensor would need a second copy of it in
	// pinned memory. Sixty-four mebibytes at a time keeps that bounded.
	const chunk = 64 << 20
	stage, err := d.Host(min(chunk, len(data)), bufferUsageTransferSrc)
	if err != nil {
		dst.Close()
		return nil, err
	}
	defer stage.Close()

	for off := 0; off < len(data); off += chunk {
		n := min(chunk, len(data)-off)
		copy(stage.Bytes(), data[off:off+n])
		if err := d.run(func(cb commandBuffer) {
			region := bufferCopy{srcOffset: 0, dstOffset: uint64(off), size: uint64(n)}
			vkCmdCopyBuffer(cb, stage.handle, dst.handle, 1, &region)
		}); err != nil {
			dst.Close()
			return nil, err
		}
	}
	return dst, nil
}

// run records one command buffer, submits it, and waits. Everything here is
// synchronous: the caller wants the answer, not a pipeline.
func (d *Device) run(record func(commandBuffer)) error {
	if err := check("vkResetCommandBuffer", vkResetCommandBuffer(d.cmd, 0)); err != nil {
		return err
	}
	bi := commandBufferBeginInfo{sType: structCommandBufferBeginInfo, flags: commandBufferOneTime}
	if err := check("vkBeginCommandBuffer", vkBeginCommandBuffer(d.cmd, &bi)); err != nil {
		return err
	}
	record(d.cmd)
	if err := check("vkEndCommandBuffer", vkEndCommandBuffer(d.cmd)); err != nil {
		return err
	}
	si := submitInfo{
		sType:              structSubmitInfo,
		commandBufferCount: 1,
		pCommandBuffers:    uintptr(unsafe.Pointer(&d.cmd)),
	}
	if err := check("vkQueueSubmit", vkQueueSubmit(d.queue, 1, &si, 0)); err != nil {
		return err
	}
	return check("vkQueueWaitIdle", vkQueueWaitIdle(d.queue))
}
