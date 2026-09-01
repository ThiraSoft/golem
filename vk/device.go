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

	// coopmat says the device took the matrix-core extensions at creation, so
	// a kernel that declares CooperativeMatrixKHR may be compiled on it.
	coopmat bool
}

// Open finds the first device with a compute queue and takes it.
func Open() (*Device, error) {
	if err := load(); err != nil {
		return nil, err
	}
	d := &Device{}

	// The kernels declare the integer dot product, which a 1.0 instance may
	// not be given. dotProductExtension below is the other half of the ask.
	name := append([]byte("golem"), 0)
	ai := applicationInfo{
		sType:            structApplicationInfo,
		pApplicationName: uintptr(unsafe.Pointer(&name[0])),
		pEngineName:      uintptr(unsafe.Pointer(&name[0])),
		apiVersion:       apiVersion11,
	}
	ici := instanceCreateInfo{sType: structInstanceCreateInfo, pApplicationInfo: uintptr(unsafe.Pointer(&ai))}
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
	have, err := d.extensions()
	if err != nil {
		d.Close()
		return nil, err
	}
	if !have[dotProductExtension] {
		d.Close()
		return nil, fmt.Errorf("vk: the device does not offer %s", dotProductExtension)
	}
	names := [][]byte{append([]byte(dotProductExtension), 0)}

	// The matrix cores, if this device has them. They are asked for as a group
	// or not at all — the product needs all four capabilities — and a device
	// without them keeps the kernel that reaches the same answer with a
	// four-byte dot product. shaders/matmul_coop.comp says what the difference
	// is worth.
	dot := shaderIntegerDotProductFeatures{sType: structDotProductFeatures, shaderIntegerDotProduct: 1}
	coop := cooperativeMatrixFeatures{sType: structCooperativeMatrixFeatures, cooperativeMatrix: 1}
	model := memoryModelFeatures{sType: structMemoryModelFeatures, vulkanMemoryModel: 1}
	f16 := shaderFloat16Int8Features{sType: structFloat16Int8Features, shaderFloat16: 1, shaderInt8: 1}
	st16 := storage16BitFeatures{sType: struct16BitStorageFeatures, storageBuffer16BitAccess: 1, uniformAndStorageBuffer16BitAccess: 1}
	st8 := storage8BitFeatures{sType: struct8BitStorageFeatures, storageBuffer8BitAccess: 1, uniformAndStorageBuffer8BitAccess: 1}
	waves := subgroupSizeFeatures{sType: structSubgroupSizeFeatures, subgroupSizeControl: 1, computeFullSubgroups: 1}
	chain := uintptr(unsafe.Pointer(&dot))
	d.coopmat = true
	for _, name := range coopmatExtensions {
		if !have[name] {
			d.coopmat = false
		}
	}
	if d.coopmat {
		for _, name := range coopmatExtensions {
			names = append(names, append([]byte(name), 0))
		}
		dot.pNext = uintptr(unsafe.Pointer(&coop))
		coop.pNext = uintptr(unsafe.Pointer(&model))
		model.pNext = uintptr(unsafe.Pointer(&f16))
		f16.pNext = uintptr(unsafe.Pointer(&st16))
		st16.pNext = uintptr(unsafe.Pointer(&st8))
		st8.pNext = uintptr(unsafe.Pointer(&waves))
	}
	pointers := make([]uintptr, len(names))
	for i := range names {
		pointers[i] = uintptr(unsafe.Pointer(&names[i][0]))
	}
	dci := deviceCreateInfo{
		sType:                   structDeviceCreateInfo,
		pNext:                   chain,
		queueCreateInfoCount:    1,
		pQueueCreateInfos:       uintptr(unsafe.Pointer(&qci)),
		enabledExtensionCount:   uint32(len(pointers)),
		ppEnabledExtensionNames: uintptr(unsafe.Pointer(&pointers[0])),
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

// dotProductExtension is what the Q4_0 and Q6_K kernels are written against.
// A four-byte-at-a-time signed dot product with a 32-bit accumulator is one
// instruction on this hardware where the unpacked form is eight, and those
// kernels are a quarter arithmetic: measured with the multiplies taken out
// altogether, the attention's projections ran twenty-seven percent faster.
//
// It has been core since Vulkan 1.3 and an extension since 1.1. A device
// without it gets an error here rather than a slower path, and the engine
// falls back to the CPU as it does when there is no Vulkan at all.
const dotProductExtension = "VK_KHR_shader_integer_dot_product"

// coopmatExtensions is what the cooperative-matrix product is written
// against: the matrix cores, the memory model its loads are defined in, and
// the sixteen-bit types the dequantised tiles are staged as.
var coopmatExtensions = []string{
	"VK_KHR_cooperative_matrix",
	"VK_KHR_vulkan_memory_model",
	"VK_KHR_shader_float16_int8",
	"VK_KHR_16bit_storage",
	"VK_KHR_8bit_storage",
	"VK_EXT_subgroup_size_control",
}

// coopmatWave is the width the cooperative product is built for. The shader
// divides its rows between a fixed number of waves, so the number has to be
// fixed: shaders/matmul_coop.comp's THREADS divided by this. RADV runs compute
// at sixty-four by default and offers thirty-two, and this asks for what the
// shader was written against rather than taking what it is given.
const coopmatWave = 64

// Coopmat says whether this device took the matrix cores, which decides which
// product a wide pass runs.
func (d *Device) Coopmat() bool { return d.coopmat }

// extensions is the set the chosen device offers.
func (d *Device) extensions() (map[string]bool, error) {
	var n uint32
	if err := check("vkEnumerateDeviceExtensionProperties",
		vkEnumerateDeviceExtensionProperties(d.phys, 0, &n, nil)); err != nil {
		return nil, err
	}
	have := map[string]bool{}
	if n == 0 {
		return have, nil
	}
	props := make([]extensionProperties, n)
	if err := check("vkEnumerateDeviceExtensionProperties",
		vkEnumerateDeviceExtensionProperties(d.phys, 0, &n, &props[0])); err != nil {
		return nil, err
	}
	for _, p := range props[:n] {
		end := 0
		for end < len(p.name) && p.name[end] != 0 {
			end++
		}
		have[string(p.name[:end])] = true
	}
	return have, nil
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

// Host allocates a buffer the CPU writes and the shader reads directly.
//
// It is for what the CPU actually writes each pass — the positions, the
// rotation angles — and for nothing else. A buffer allocated here lives in
// system memory, and a shader reading it reaches across the bus for every line
// of it that is not already in a cache.
//
// **That is not a small thing, and it was the largest single cost in the
// prompt.** The stream between two kernels — the normed activation, what the
// output projection makes, the shared branch's input — is written by the card
// and read by the card, and it was allocated here because the per-block API
// that has since been replaced wanted to write it from this side. Moving those
// six buffers to Local took a prefill of sixty-four positions on the Qwen3 4B
// from 71.9 milliseconds to 46.9, and a token of the 26B from 96.8 to 102.8 a
// second. Anything a kernel writes and a kernel reads belongs in Local.
func (d *Device) Host(size int, usage uint32) (*Buffer, error) {
	return d.newBuffer(uint64(size), usage, memoryHostVisible|memoryHostCoherent)
}

// Readback allocates a buffer the shader writes and the CPU reads back whole.
//
// It asks for cached memory as well as visible, and the difference is not
// small. An uncached host-visible allocation on a discrete card is
// write-combined: writing it is fast and reading it comes back over the bus a
// word at a time with nothing held on to. The logit head writes a megabyte of
// logits a token, and copying that megabyte out of a write-combined buffer
// cost three milliseconds against the kernel's one and a half — most of what
// the head appeared to cost was the read, not the product.
//
// A card with no cached host-visible type gives the uncached one back, which
// is what the buffer would have been anyway.
func (d *Device) Readback(size int, usage uint32) (*Buffer, error) {
	b, err := d.newBuffer(uint64(size), usage, memoryHostVisible|memoryHostCoherent|memoryHostCached)
	if err == nil {
		return b, nil
	}
	return d.Host(size, usage)
}

// Local allocates a buffer in device memory that no one on this side reads or
// writes. It is what an intermediate between two dispatches wants: the card
// produces it and the card consumes it, and a host-visible allocation would
// put both across the bus for nothing.
func (d *Device) Local(size int, usage uint32) (*Buffer, error) {
	return d.newBuffer(uint64(size), usage, memoryDeviceLocal)
}

// Size is how many bytes the buffer holds, whether or not it has an address on
// this side.
func (b *Buffer) Size() int { return int(b.size) }

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

// Uints is the mapped buffer read as uint32.
func (b *Buffer) Uints() []uint32 {
	if b.mapped == nil {
		panic("vk: buffer is not host visible")
	}
	return unsafe.Slice((*uint32)(b.mapped), b.size/4)
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
func (d *Device) Upload(data []byte) (*Buffer, error) { return d.UploadTail(data, 0) }

// UploadTail is Upload with room left after the data.
//
// A kernel that reads its weights a word at a time reads a whole word even for
// the last byte it wants, and one that slides a window across a misaligned
// stream reads the word after that as well. matvec_t4g.comp does both, so the
// last block of the last row reaches a few bytes past the tensor. Those bytes
// are multiplied by nothing — the loop that wants them has already stopped —
// but they have to be inside the allocation, so the caller says how many.
func (d *Device) UploadTail(data []byte, tail int) (*Buffer, error) {
	// Rounded up to a word. Every kernel here reads a storage buffer as uint[],
	// so a tensor whose byte count is not a multiple of four would put its last
	// bytes in a word past the end of the buffer — which most drivers answer
	// with zeros and one answers with a fault. A T4G row is 67·n/128 bytes and
	// is odd whenever the row is not a multiple of 512 wide, which a vision
	// tower's 1152 is not.
	size := ((len(data) + tail) + 3) &^ 3
	dst, err := d.newBuffer(uint64(size), bufferUsageStorage|bufferUsageTransferDst, memoryDeviceLocal)
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

// DeviceLocalBytes is the largest device-local heap the card reports. It is
// what a caller sizing a working set against the card has to divide, and it is
// asked rather than assumed: this repository's own card holds sixteen
// gigabytes, and a constant tuned against that is wrong on a twelve-gigabyte
// card in one direction and on a twenty-four in the other.
//
// It is the heap's size and not what is free in it. Vulkan reports the free
// figure only through VK_EXT_memory_budget, which is not required and is not
// asked for here; a caller keeps a margin instead.
func (d *Device) DeviceLocalBytes() uint64 {
	var most uint64
	for i := uint32(0); i < d.memory.memoryHeapCount; i++ {
		h := d.memory.memoryHeaps[i]
		if h.flags&memoryDeviceLocal != 0 && h.size > most {
			most = h.size
		}
	}
	return most
}
