package vk

// Opening a compute device, and the memory that lives on it.

import (
	"fmt"
	"os"
	"unsafe"

	"github.com/ebitengine/purego"
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

	// importAlign is what a host pointer has to be aligned to before the card
	// can be given it directly, or zero on a device that cannot be. See Import.
	importAlign uint64

	// stage is the staging pair every upload alternates between. See staging.
	stage [2]*Buffer

	// budget says the driver will answer DeviceLocalFree with something real.
	// See that function for what it is for and why it is asked rather than
	// computed.
	budget bool
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

	// Importing host memory, if this device offers it. It is what makes a
	// streamed weight cost one crossing instead of a crossing and a copy, and a
	// device without it falls back to the staging path rather than failing.
	if have[hostImportExtension] {
		names = append(names, append([]byte(hostImportExtension), 0))
	}
	// What is free on the card, rather than how large it is. A caller sizing a
	// working set has been dividing the heap by a constant found by trial; this
	// is the driver's own answer, and it moves with what else is resident.
	if have[budgetExtension] {
		names = append(names, append([]byte(budgetExtension), 0))
		d.budget = true
	}

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

	if have[hostImportExtension] {
		d.findHostImport()
	}

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

// hostImportExtension lets a pointer this process already owns be handed to the
// card as device memory.
//
// It is what the streamed paths want. Staging a weight costs a write into
// host-visible memory and then a read of it by the copy engine, and the two do
// not add up to a copy and a crossing: they contend for the same host memory
// controller. Measured here on a gigabyte, the staged path reaches 2.6 GB/s and
// double-buffering it 4.8, against 6.9 for the crossing alone. Importing the
// pointer removes the copy rather than hiding it.
const hostImportExtension = "VK_EXT_external_memory_host"

// budgetExtension reports what each heap has left.
//
// It matters where a working set is sized against the card rather than chosen:
// qwen35's streamed window takes two thirds of the heap's *size*, a fraction
// arrived at by losing the device at three quarters. Two thirds of the size is
// seventy-one per cent of what was actually free the moment this was written,
// and that figure moves with the desktop, with the driver's own allocations,
// and with whatever the previous window has not finished releasing.
const budgetExtension = "VK_EXT_memory_budget"

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
	for i, b := range d.stage {
		if b != nil {
			b.Close()
			d.stage[i] = nil
		}
	}
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

// findHostImport resolves the import entry point and the alignment a pointer
// must satisfy. Both are asked of the driver: the entry point because an
// extension function need not be an exported symbol of the loader, and the
// alignment because the specification allows anything up to sixty-four
// kibibytes and a constant here would be this driver's number on every other.
func (d *Device) findHostImport() {
	name := append([]byte("vkGetMemoryHostPointerPropertiesEXT"), 0)
	fn := vkGetDeviceProcAddr(d.dev, uintptr(unsafe.Pointer(&name[0])))
	if fn == 0 {
		return
	}
	purego.RegisterFunc(&vkGetMemoryHostPointerProperties, fn)

	// VkPhysicalDeviceProperties2 carries the whole of VkPhysicalDeviceProperties
	// inline, which is eight hundred-odd bytes this file has no reason to
	// describe. It is given room the driver cannot overrun and read through the
	// pNext chain, which is the only part wanted.
	ext := externalMemoryHostProperties{sType: structExternalMemoryHostProps}
	room := make([]byte, 4096)
	*(*uint32)(unsafe.Pointer(&room[0])) = structProperties2
	*(*uintptr)(unsafe.Pointer(&room[8])) = uintptr(unsafe.Pointer(&ext))
	vkGetPhysicalDeviceProperties2(d.phys, unsafe.Pointer(&room[0]))
	d.importAlign = ext.minImportedHostPointerAlignment

	// A driver that answers zero is not saying the alignment is zero, it is
	// saying nothing — which is what this one did until the instance stopped
	// asking for Vulkan 1.0 under the name of 1.1 (see apiVersion11). The
	// fallback is kept because it costs one page and it is the difference
	// between a wrong answer and no import: a page is what every driver that
	// offers this asks for, and a device that refuses the trial keeps the
	// staging path.
	if d.importAlign == 0 {
		d.importAlign = uint64(os.Getpagesize())
		page := make([]byte, 2*d.importAlign)
		at := (d.importAlign - uint64(uintptr(unsafe.Pointer(&page[0])))%d.importAlign) % d.importAlign
		b, err := d.Import(unsafe.Pointer(&page[at]), int(d.importAlign), bufferUsageTransferSrc)
		if err != nil {
			d.importAlign = 0
			return
		}
		b.Close()
	}
}

// Import wraps memory this process owns as a buffer the card reads directly,
// with no staging buffer and no copy on this side.
//
// The pointer and the length must both be multiples of ImportAlignment, which
// is a page here. That is not a burden for what this is for: a weight streamed
// from a mapped checkpoint is page-aligned because mmap returns pages, and a
// pool this side allocates can be asked for the same.
//
// The card reads the memory over the bus every time a kernel touches it, so an
// imported buffer is for a tensor that is read once — a weight on its way to
// device memory, or an expert used for one token. A tensor read every token
// belongs in Local.
//
// The caller keeps the memory alive: the returned buffer does not own it, and
// Close releases the card's handle on it and nothing else.
func (d *Device) Import(p unsafe.Pointer, size int, usage uint32) (*Buffer, error) {
	if d.importAlign == 0 {
		return nil, fmt.Errorf("vk: the device cannot import host memory")
	}
	if uintptr(p)%uintptr(d.importAlign) != 0 || uint64(size)%d.importAlign != 0 {
		return nil, fmt.Errorf("vk: an imported pointer and length must be multiples of %d", d.importAlign)
	}
	props := memoryHostPointerProperties{sType: structHostPointerProperties}
	if err := check("vkGetMemoryHostPointerPropertiesEXT",
		vkGetMemoryHostPointerProperties(d.dev, handleTypeHostAllocation, p, &props)); err != nil {
		return nil, err
	}

	b := &Buffer{d: d, size: uint64(size)}
	ext := externalMemoryBufferCreateInfo{sType: structExternalMemoryBuffer, handleTypes: handleTypeHostAllocation}
	bci := bufferCreateInfo{
		sType: structBufferCreateInfo,
		pNext: uintptr(unsafe.Pointer(&ext)),
		size:  uint64(size),
		usage: usage,
	}
	if err := check("vkCreateBuffer", vkCreateBuffer(d.dev, &bci, 0, &b.handle)); err != nil {
		return nil, err
	}
	var req memoryRequirements
	vkGetBufferMemoryRequirements(d.dev, b.handle, &req)

	// The type has to suit both the buffer and the pointer: the driver decides
	// which types an imported allocation may be, and it is not every type the
	// buffer would otherwise accept.
	kind, err := d.memoryTypeFor(req.memoryTypeBits&props.memoryTypeBits, memoryHostVisible)
	if err != nil {
		b.Close()
		return nil, err
	}
	imp := importMemoryHostPointerInfo{
		sType:        structImportMemoryHostPointer,
		handleType:   handleTypeHostAllocation,
		pHostPointer: p,
	}
	mai := memoryAllocateInfo{
		sType:           structMemoryAllocateInfo,
		pNext:           uintptr(unsafe.Pointer(&imp)),
		allocationSize:  uint64(size),
		memoryTypeIndex: kind,
	}
	if err := check("vkAllocateMemory", vkAllocateMemory(d.dev, &mai, 0, &b.mem)); err != nil {
		b.Close()
		return nil, err
	}
	if err := check("vkBindBufferMemory", vkBindBufferMemory(d.dev, b.handle, b.mem, 0)); err != nil {
		b.Close()
		return nil, err
	}
	return b, nil
}

// ImportAlignment is what Import demands of a pointer and a length, or zero on
// a device that cannot import at all.
func (d *Device) ImportAlignment() uint64 { return d.importAlign }

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
	if err := d.CopyInto(dst, 0, data); err != nil {
		dst.Close()
		return nil, err
	}
	return dst, nil
}

// stageChunk is how much of a tensor crosses in one submission.
//
// It is the width the two stages balance at, measured on a gigabyte: at four
// mebibytes the submissions cost more than they hide (3.8 GB/s), at sixteen the
// copy is fully behind the transfer (5.9), and at sixty-four and above the tail
// of the last chunk is long enough to show in the total and the answer starts
// moving between runs. The ceiling is what the copy engine alone reaches, 6.7.
const stageChunk = 16 << 20

// staging is the pair of host buffers every upload passes through, allocated on
// first use and kept for the life of the device.
//
// A pair, and not one: the copy into a staging buffer and the card's read out
// of it are on opposite sides of the machine and used to run one after the
// other, which cost more than either. Alternating between two buffers puts the
// copy of the next chunk beside the transfer of this one. Measured on a
// gigabyte, 2.6 GB/s becomes 5.9 against a bus that carries 6.7 — this card
// reaches the processor over eight lanes at 8 GT/s, whatever the card's own
// port reports, so 6.7 is nearly the whole link and there is nothing else here
// to win.
//
// They are kept rather than made per call because a checkpoint is hundreds of
// tensors and thirty-two mebibytes of host memory is faulted in once.
func (d *Device) staging() ([2]*Buffer, error) {
	// Both or neither. A first buffer kept after the second failed would leave
	// the next call finding d.stage[0] set, returning a pair whose second half
	// is nil, and saying nothing was wrong — which CopyInto dereferences.
	if d.stage[0] == nil {
		var made [2]*Buffer
		for i := range made {
			b, err := d.Host(stageChunk, bufferUsageTransferSrc)
			if err != nil {
				for _, m := range made {
					if m != nil {
						m.Close()
					}
				}
				return [2]*Buffer{}, err
			}
			made[i] = b
		}
		d.stage = made
	}
	return d.stage, nil
}

// CopyInto writes data into an existing device-local buffer at an offset,
// through the shared staging pair. It is what UploadTail is built on, and what
// a streamed weight wants directly: the destination outlives the transfer.
func (d *Device) CopyInto(dst *Buffer, at int, data []byte) error {
	stage, err := d.staging()
	if err != nil {
		return err
	}
	// One chunk is one copy and one submission, with nothing to overlap and no
	// goroutine to pay for. Most tensors are this.
	if len(data) <= stageChunk {
		copy(stage[0].Bytes()[:len(data)], data)
		return d.copyBuffer(stage[0], dst, uint64(at), uint64(len(data)))
	}

	// free carries the slots this side may write. A slot returns to it only
	// once the card has finished reading it: without that handshake the copy
	// runs a buffer ahead into one still in flight and overwrites the bytes
	// being sent, which is a wrong answer and not a slow one.
	free := make(chan int, len(stage))
	for i := range stage {
		free <- i
	}
	type ready struct {
		slot int
		off  int
	}
	filled := make(chan ready, 1)
	go func() {
		for off := 0; off < len(data); off += stageChunk {
			slot := <-free
			n := min(stageChunk, len(data)-off)
			copy(stage[slot].Bytes()[:n], data[off:off+n])
			filled <- ready{slot, off}
		}
		close(filled)
	}()
	var failed error
	for r := range filled {
		n := min(stageChunk, len(data)-r.off)
		if failed == nil {
			failed = d.copyBuffer(stage[r.slot], dst, uint64(at+r.off), uint64(n))
		}
		free <- r.slot
	}
	return failed
}

// copyBuffer submits one region copy and waits for it.
func (d *Device) copyBuffer(src, dst *Buffer, at, n uint64) error {
	return d.run(func(cb commandBuffer) {
		region := bufferCopy{srcOffset: 0, dstOffset: at, size: n}
		vkCmdCopyBuffer(cb, src.handle, dst.handle, 1, &region)
	})
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

// DeviceLocalFree is what the largest device-local heap has left, or zero when
// the driver will not say.
//
// It is the answer DeviceLocalBytes cannot give. A caller that has just
// released a window of ten gigabytes and is about to ask for another has no way
// to know from the heap's size whether the first one is actually gone: the
// driver frees asynchronously, and an allocation that succeeds against a heap
// the kernel has not finished reclaiming is where a hard recovery comes from.
// This is asked again each time rather than cached for exactly that reason.
func (d *Device) DeviceLocalFree() uint64 {
	if !d.budget {
		return 0
	}
	// VkPhysicalDeviceMemoryProperties2 carries the whole of
	// VkPhysicalDeviceMemoryProperties inline. It is given room the driver
	// cannot overrun and read through the pNext chain, which is the only part
	// wanted — the same arrangement as findHostImport, and for the same reason.
	ext := memoryBudgetProperties{sType: structMemoryBudget}
	room := make([]byte, 4096)
	*(*uint32)(unsafe.Pointer(&room[0])) = structMemoryProperties2
	*(*uintptr)(unsafe.Pointer(&room[8])) = uintptr(unsafe.Pointer(&ext))
	vkGetPhysicalDeviceMemoryProps2(d.phys, unsafe.Pointer(&room[0]))

	// The same heap DeviceLocalBytes names, so that the free figure and the
	// size are about the same memory.
	var at uint32
	var most uint64
	for i := uint32(0); i < d.memory.memoryHeapCount && i < 16; i++ {
		h := d.memory.memoryHeaps[i]
		if h.flags&memoryDeviceLocal != 0 && h.size > most {
			most, at = h.size, i
		}
	}
	if most == 0 || ext.budget[at] <= ext.usage[at] {
		return 0
	}
	return ext.budget[at] - ext.usage[at]
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
