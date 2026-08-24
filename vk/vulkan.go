package vk

// The part of Vulkan a compute dispatch needs, and nothing else.
//
// There is no cgo here. purego opens libvulkan.so.1 and binds the core entry
// points, which the loader exports by name, so golem keeps building with
// CGO_ENABLED=0 and keeps cross-compiling. What that costs is this file: every
// structure Vulkan reads has to be laid out by hand with the padding a C
// compiler would have inserted, because Go's rules are not C's. Each one below
// is written with its C offsets in the comment, and a wrong offset is not a
// compile error — it is a driver reading a pointer out of the middle of two
// integers. That is the price of the boundary, paid once.

import (
	"fmt"
	"unsafe"

	"github.com/ebitengine/purego"
)

// Handles. The dispatchable ones are pointers; the rest are 64-bit values even
// on a 32-bit host, which is why they are uint64 and not uintptr.
type (
	instance       uintptr
	physicalDevice uintptr
	device         uintptr
	queue          uintptr
	commandBuffer  uintptr
)

const (
	success = 0

	structInstanceCreateInfo        = 1
	structDeviceQueueCreateInfo     = 2
	structDeviceCreateInfo          = 3
	structSubmitInfo                = 4
	structMemoryAllocateInfo        = 5
	structFenceCreateInfo           = 8
	structBufferCreateInfo          = 12
	structShaderModuleCreateInfo    = 16
	structPipelineShaderStageInfo   = 18
	structComputePipelineCreateInfo = 29
	structPipelineLayoutCreateInfo  = 30
	structDescriptorSetLayoutInfo   = 32
	structDescriptorPoolCreateInfo  = 33
	structDescriptorSetAllocateInfo = 34
	structWriteDescriptorSet        = 35
	structCommandPoolCreateInfo     = 39
	structCommandBufferAllocateInfo = 40
	structCommandBufferBeginInfo    = 42
	structMemoryBarrier             = 46
	structQueryPoolCreateInfo       = 11

	queryTypeTimestamp = 2

	queueCompute = 0x2

	bufferUsageTransferSrc = 0x1
	bufferUsageTransferDst = 0x2
	bufferUsageStorage     = 0x20

	memoryDeviceLocal  = 0x1
	memoryHostVisible  = 0x2
	memoryHostCoherent = 0x4

	descriptorStorageBuffer = 7
	shaderStageCompute      = 0x20

	commandBufferOneTime = 0x1
	pipelineBindCompute  = 1

	stageBottomOfPipe   = 0x2000
	stageComputeShader  = 0x800
	stageTransfer       = 0x1000
	accessShaderRead    = 0x20
	accessShaderWrite   = 0x40
	accessTransferRead  = 0x800
	accessTransferWrite = 0x1000
)

// vkApplicationInfo is skipped everywhere: it is optional and an instance
// without it is a 1.0 instance, which is all a compute dispatch asks for.

// instanceCreateInfo: sType 0, pNext 8, flags 16, pApplicationInfo 24,
// enabledLayerCount 32, ppEnabledLayerNames 40, enabledExtensionCount 48,
// ppEnabledExtensionNames 56.
type instanceCreateInfo struct {
	sType                   uint32
	_                       uint32
	pNext                   uintptr
	flags                   uint32
	_                       uint32
	pApplicationInfo        uintptr
	enabledLayerCount       uint32
	_                       uint32
	ppEnabledLayerNames     uintptr
	enabledExtensionCount   uint32
	_                       uint32
	ppEnabledExtensionNames uintptr
}

// deviceQueueCreateInfo: sType 0, pNext 8, flags 16, queueFamilyIndex 20,
// queueCount 24, pQueuePriorities 32.
type deviceQueueCreateInfo struct {
	sType            uint32
	_                uint32
	pNext            uintptr
	flags            uint32
	queueFamilyIndex uint32
	queueCount       uint32
	_                uint32
	pQueuePriorities uintptr
}

// deviceCreateInfo: sType 0, pNext 8, flags 16, queueCreateInfoCount 20,
// pQueueCreateInfos 24, enabledLayerCount 32, ppEnabledLayerNames 40,
// enabledExtensionCount 48, ppEnabledExtensionNames 56, pEnabledFeatures 64.
type deviceCreateInfo struct {
	sType                   uint32
	_                       uint32
	pNext                   uintptr
	flags                   uint32
	queueCreateInfoCount    uint32
	pQueueCreateInfos       uintptr
	enabledLayerCount       uint32
	_                       uint32
	ppEnabledLayerNames     uintptr
	enabledExtensionCount   uint32
	_                       uint32
	ppEnabledExtensionNames uintptr
	pEnabledFeatures        uintptr
}

// queueFamilyProperties: queueFlags 0, queueCount 4, timestampValidBits 8,
// minImageTransferGranularity 12 (three uint32).
type queueFamilyProperties struct {
	queueFlags         uint32
	queueCount         uint32
	timestampValidBits uint32
	granularity        [3]uint32
}

// memoryType is 8 bytes, memoryHeap is 16 (a uint64 first).
type memoryType struct {
	propertyFlags uint32
	heapIndex     uint32
}

type memoryHeap struct {
	size  uint64
	flags uint32
	_     uint32
}

// physicalDeviceMemoryProperties: typeCount 0, types 4..260, heapCount 260,
// heaps 264 (the array of uint64-aligned heaps forces four bytes of padding).
type physicalDeviceMemoryProperties struct {
	memoryTypeCount uint32
	memoryTypes     [32]memoryType
	memoryHeapCount uint32
	_               uint32
	memoryHeaps     [16]memoryHeap
}

// bufferCreateInfo: sType 0, pNext 8, flags 16, size 24, usage 32,
// sharingMode 36, queueFamilyIndexCount 40, pQueueFamilyIndices 48.
type bufferCreateInfo struct {
	sType                 uint32
	_                     uint32
	pNext                 uintptr
	flags                 uint32
	_                     uint32
	size                  uint64
	usage                 uint32
	sharingMode           uint32
	queueFamilyIndexCount uint32
	_                     uint32
	pQueueFamilyIndices   uintptr
}

// memoryRequirements: size 0, alignment 8, memoryTypeBits 16.
type memoryRequirements struct {
	size           uint64
	alignment      uint64
	memoryTypeBits uint32
	_              uint32
}

// memoryAllocateInfo: sType 0, pNext 8, allocationSize 16, memoryTypeIndex 24.
type memoryAllocateInfo struct {
	sType           uint32
	_               uint32
	pNext           uintptr
	allocationSize  uint64
	memoryTypeIndex uint32
	_               uint32
}

// descriptorSetLayoutBinding: binding 0, descriptorType 4, descriptorCount 8,
// stageFlags 12, pImmutableSamplers 16.
type descriptorSetLayoutBinding struct {
	binding            uint32
	descriptorType     uint32
	descriptorCount    uint32
	stageFlags         uint32
	pImmutableSamplers uintptr
}

// descriptorSetLayoutCreateInfo: sType 0, pNext 8, flags 16, bindingCount 20,
// pBindings 24.
type descriptorSetLayoutCreateInfo struct {
	sType        uint32
	_            uint32
	pNext        uintptr
	flags        uint32
	bindingCount uint32
	pBindings    uintptr
}

type pushConstantRange struct {
	stageFlags uint32
	offset     uint32
	size       uint32
}

// pipelineLayoutCreateInfo: sType 0, pNext 8, flags 16, setLayoutCount 20,
// pSetLayouts 24, pushConstantRangeCount 32, pPushConstantRanges 40.
type pipelineLayoutCreateInfo struct {
	sType                  uint32
	_                      uint32
	pNext                  uintptr
	flags                  uint32
	setLayoutCount         uint32
	pSetLayouts            uintptr
	pushConstantRangeCount uint32
	_                      uint32
	pPushConstantRanges    uintptr
}

// shaderModuleCreateInfo: sType 0, pNext 8, flags 16, codeSize 24, pCode 32.
type shaderModuleCreateInfo struct {
	sType    uint32
	_        uint32
	pNext    uintptr
	flags    uint32
	_        uint32
	codeSize uint64
	pCode    uintptr
}

// pipelineShaderStageCreateInfo: sType 0, pNext 8, flags 16, stage 20,
// module 24, pName 32, pSpecializationInfo 40. Forty-eight bytes.
type pipelineShaderStageCreateInfo struct {
	sType               uint32
	_                   uint32
	pNext               uintptr
	flags               uint32
	stage               uint32
	module              uint64
	pName               uintptr
	pSpecializationInfo uintptr
}

// computePipelineCreateInfo: sType 0, pNext 8, flags 16, stage 24 (48 bytes),
// layout 72, basePipelineHandle 80, basePipelineIndex 88.
type computePipelineCreateInfo struct {
	sType              uint32
	_                  uint32
	pNext              uintptr
	flags              uint32
	_                  uint32
	stage              pipelineShaderStageCreateInfo
	layout             uint64
	basePipelineHandle uint64
	basePipelineIndex  int32
	_                  int32
}

type descriptorPoolSize struct {
	kind  uint32
	count uint32
}

// descriptorPoolCreateInfo: sType 0, pNext 8, flags 16, maxSets 20,
// poolSizeCount 24, pPoolSizes 32.
type descriptorPoolCreateInfo struct {
	sType         uint32
	_             uint32
	pNext         uintptr
	flags         uint32
	maxSets       uint32
	poolSizeCount uint32
	_             uint32
	pPoolSizes    uintptr
}

// descriptorSetAllocateInfo: sType 0, pNext 8, descriptorPool 16,
// descriptorSetCount 24, pSetLayouts 32.
type descriptorSetAllocateInfo struct {
	sType              uint32
	_                  uint32
	pNext              uintptr
	descriptorPool     uint64
	descriptorSetCount uint32
	_                  uint32
	pSetLayouts        uintptr
}

type descriptorBufferInfo struct {
	buffer uint64
	offset uint64
	rng    uint64
}

// writeDescriptorSet: sType 0, pNext 8, dstSet 16, dstBinding 24,
// dstArrayElement 28, descriptorCount 32, descriptorType 36, pImageInfo 40,
// pBufferInfo 48, pTexelBufferView 56.
type writeDescriptorSet struct {
	sType            uint32
	_                uint32
	pNext            uintptr
	dstSet           uint64
	dstBinding       uint32
	dstArrayElement  uint32
	descriptorCount  uint32
	descriptorType   uint32
	pImageInfo       uintptr
	pBufferInfo      uintptr
	pTexelBufferView uintptr
}

// commandPoolCreateInfo: sType 0, pNext 8, flags 16, queueFamilyIndex 20.
type commandPoolCreateInfo struct {
	sType            uint32
	_                uint32
	pNext            uintptr
	flags            uint32
	queueFamilyIndex uint32
}

// commandBufferAllocateInfo: sType 0, pNext 8, commandPool 16, level 24,
// commandBufferCount 28.
type commandBufferAllocateInfo struct {
	sType              uint32
	_                  uint32
	pNext              uintptr
	commandPool        uint64
	level              uint32
	commandBufferCount uint32
}

// commandBufferBeginInfo: sType 0, pNext 8, flags 16, pInheritanceInfo 24.
type commandBufferBeginInfo struct {
	sType            uint32
	_                uint32
	pNext            uintptr
	flags            uint32
	_                uint32
	pInheritanceInfo uintptr
}

// submitInfo: sType 0, pNext 8, waitSemaphoreCount 16, pWaitSemaphores 24,
// pWaitDstStageMask 32, commandBufferCount 40, pCommandBuffers 48,
// signalSemaphoreCount 56, pSignalSemaphores 64.
type submitInfo struct {
	sType                uint32
	_                    uint32
	pNext                uintptr
	waitSemaphoreCount   uint32
	_                    uint32
	pWaitSemaphores      uintptr
	pWaitDstStageMask    uintptr
	commandBufferCount   uint32
	_                    uint32
	pCommandBuffers      uintptr
	signalSemaphoreCount uint32
	_                    uint32
	pSignalSemaphores    uintptr
}

// memoryBarrier: sType 0, pNext 8, srcAccessMask 16, dstAccessMask 20.
type memoryBarrier struct {
	sType         uint32
	_             uint32
	pNext         uintptr
	srcAccessMask uint32
	dstAccessMask uint32
}

type queryPoolCreateInfo struct {
	sType              uint32
	_                  uint32
	pNext              uintptr
	flags              uint32
	queryType          uint32
	queryCount         uint32
	pipelineStatistics uint32
}

type bufferCopy struct {
	srcOffset uint64
	dstOffset uint64
	size      uint64
}

// The entry points, bound once by load().
var (
	vkCreateInstance                    func(*instanceCreateInfo, uintptr, *instance) int32
	vkDestroyInstance                   func(instance, uintptr)
	vkEnumeratePhysicalDevices          func(instance, *uint32, *physicalDevice) int32
	vkGetPhysicalDeviceQueueFamilyProps func(physicalDevice, *uint32, *queueFamilyProperties)
	vkGetPhysicalDeviceMemoryProperties func(physicalDevice, *physicalDeviceMemoryProperties)
	vkCreateDevice                      func(physicalDevice, *deviceCreateInfo, uintptr, *device) int32
	vkDestroyDevice                     func(device, uintptr)
	vkGetDeviceQueue                    func(device, uint32, uint32, *queue)
	vkCreateBuffer                      func(device, *bufferCreateInfo, uintptr, *uint64) int32
	vkDestroyBuffer                     func(device, uint64, uintptr)
	vkGetBufferMemoryRequirements       func(device, uint64, *memoryRequirements)
	vkAllocateMemory                    func(device, *memoryAllocateInfo, uintptr, *uint64) int32
	vkFreeMemory                        func(device, uint64, uintptr)
	vkBindBufferMemory                  func(device, uint64, uint64, uint64) int32
	vkMapMemory                         func(device, uint64, uint64, uint64, uint32, *uintptr) int32
	vkUnmapMemory                       func(device, uint64)
	vkCreateShaderModule                func(device, *shaderModuleCreateInfo, uintptr, *uint64) int32
	vkDestroyShaderModule               func(device, uint64, uintptr)
	vkCreateDescriptorSetLayout         func(device, *descriptorSetLayoutCreateInfo, uintptr, *uint64) int32
	vkDestroyDescriptorSetLayout        func(device, uint64, uintptr)
	vkCreatePipelineLayout              func(device, *pipelineLayoutCreateInfo, uintptr, *uint64) int32
	vkDestroyPipelineLayout             func(device, uint64, uintptr)
	vkCreateComputePipelines            func(device, uint64, uint32, *computePipelineCreateInfo, uintptr, *uint64) int32
	vkDestroyPipeline                   func(device, uint64, uintptr)
	vkCreateDescriptorPool              func(device, *descriptorPoolCreateInfo, uintptr, *uint64) int32
	vkDestroyDescriptorPool             func(device, uint64, uintptr)
	vkAllocateDescriptorSets            func(device, *descriptorSetAllocateInfo, *uint64) int32
	vkUpdateDescriptorSets              func(device, uint32, *writeDescriptorSet, uint32, uintptr)
	vkCreateCommandPool                 func(device, *commandPoolCreateInfo, uintptr, *uint64) int32
	vkDestroyCommandPool                func(device, uint64, uintptr)
	vkAllocateCommandBuffers            func(device, *commandBufferAllocateInfo, *commandBuffer) int32
	vkBeginCommandBuffer                func(commandBuffer, *commandBufferBeginInfo) int32
	vkEndCommandBuffer                  func(commandBuffer) int32
	vkResetCommandBuffer                func(commandBuffer, uint32) int32
	vkCmdBindPipeline                   func(commandBuffer, uint32, uint64)
	vkCmdBindDescriptorSets             func(commandBuffer, uint32, uint64, uint32, uint32, *uint64, uint32, uintptr)
	vkCmdPushConstants                  func(commandBuffer, uint64, uint32, uint32, uint32, unsafe.Pointer)
	vkCmdDispatch                       func(commandBuffer, uint32, uint32, uint32)
	vkCmdCopyBuffer                     func(commandBuffer, uint64, uint64, uint32, *bufferCopy)
	vkCmdPipelineBarrier                func(commandBuffer, uint32, uint32, uint32, uint32, *memoryBarrier, uint32, uintptr, uint32, uintptr)
	vkCreateQueryPool                   func(device, *queryPoolCreateInfo, uintptr, *uint64) int32
	vkDestroyQueryPool                  func(device, uint64, uintptr)
	vkCmdResetQueryPool                 func(commandBuffer, uint64, uint32, uint32)
	vkCmdWriteTimestamp                 func(commandBuffer, uint32, uint64, uint32)
	vkGetQueryPoolResults               func(device, uint64, uint32, uint32, uint64, unsafe.Pointer, uint64, uint32) int32
	vkQueueSubmit                       func(queue, uint32, *submitInfo, uint64) int32
	vkQueueWaitIdle                     func(queue) int32
)

var loaded bool

// load binds libvulkan.so.1. It is idempotent and safe to call from anywhere;
// the caller that opens a device does it first.
func load() error {
	if loaded {
		return nil
	}
	lib, err := purego.Dlopen("libvulkan.so.1", purego.RTLD_NOW|purego.RTLD_GLOBAL)
	if err != nil {
		return fmt.Errorf("vk: no Vulkan loader: %w", err)
	}
	bind := func(p any, name string) {
		purego.RegisterLibFunc(p, lib, name)
	}
	bind(&vkCreateInstance, "vkCreateInstance")
	bind(&vkDestroyInstance, "vkDestroyInstance")
	bind(&vkEnumeratePhysicalDevices, "vkEnumeratePhysicalDevices")
	bind(&vkGetPhysicalDeviceQueueFamilyProps, "vkGetPhysicalDeviceQueueFamilyProperties")
	bind(&vkGetPhysicalDeviceMemoryProperties, "vkGetPhysicalDeviceMemoryProperties")
	bind(&vkCreateDevice, "vkCreateDevice")
	bind(&vkDestroyDevice, "vkDestroyDevice")
	bind(&vkGetDeviceQueue, "vkGetDeviceQueue")
	bind(&vkCreateBuffer, "vkCreateBuffer")
	bind(&vkDestroyBuffer, "vkDestroyBuffer")
	bind(&vkGetBufferMemoryRequirements, "vkGetBufferMemoryRequirements")
	bind(&vkAllocateMemory, "vkAllocateMemory")
	bind(&vkFreeMemory, "vkFreeMemory")
	bind(&vkBindBufferMemory, "vkBindBufferMemory")
	bind(&vkMapMemory, "vkMapMemory")
	bind(&vkUnmapMemory, "vkUnmapMemory")
	bind(&vkCreateShaderModule, "vkCreateShaderModule")
	bind(&vkDestroyShaderModule, "vkDestroyShaderModule")
	bind(&vkCreateDescriptorSetLayout, "vkCreateDescriptorSetLayout")
	bind(&vkDestroyDescriptorSetLayout, "vkDestroyDescriptorSetLayout")
	bind(&vkCreatePipelineLayout, "vkCreatePipelineLayout")
	bind(&vkDestroyPipelineLayout, "vkDestroyPipelineLayout")
	bind(&vkCreateComputePipelines, "vkCreateComputePipelines")
	bind(&vkDestroyPipeline, "vkDestroyPipeline")
	bind(&vkCreateDescriptorPool, "vkCreateDescriptorPool")
	bind(&vkDestroyDescriptorPool, "vkDestroyDescriptorPool")
	bind(&vkAllocateDescriptorSets, "vkAllocateDescriptorSets")
	bind(&vkUpdateDescriptorSets, "vkUpdateDescriptorSets")
	bind(&vkCreateCommandPool, "vkCreateCommandPool")
	bind(&vkDestroyCommandPool, "vkDestroyCommandPool")
	bind(&vkAllocateCommandBuffers, "vkAllocateCommandBuffers")
	bind(&vkBeginCommandBuffer, "vkBeginCommandBuffer")
	bind(&vkEndCommandBuffer, "vkEndCommandBuffer")
	bind(&vkResetCommandBuffer, "vkResetCommandBuffer")
	bind(&vkCmdBindPipeline, "vkCmdBindPipeline")
	bind(&vkCmdBindDescriptorSets, "vkCmdBindDescriptorSets")
	bind(&vkCmdPushConstants, "vkCmdPushConstants")
	bind(&vkCmdDispatch, "vkCmdDispatch")
	bind(&vkCmdCopyBuffer, "vkCmdCopyBuffer")
	bind(&vkCmdPipelineBarrier, "vkCmdPipelineBarrier")
	bind(&vkCreateQueryPool, "vkCreateQueryPool")
	bind(&vkDestroyQueryPool, "vkDestroyQueryPool")
	bind(&vkCmdResetQueryPool, "vkCmdResetQueryPool")
	bind(&vkCmdWriteTimestamp, "vkCmdWriteTimestamp")
	bind(&vkGetQueryPoolResults, "vkGetQueryPoolResults")
	bind(&vkQueueSubmit, "vkQueueSubmit")
	bind(&vkQueueWaitIdle, "vkQueueWaitIdle")
	loaded = true
	return nil
}

// check turns a VkResult into an error naming the call that produced it.
func check(what string, r int32) error {
	if r != success {
		return fmt.Errorf("vk: %s failed (VkResult %d)", what, r)
	}
	return nil
}

// errBindings is the one shape mismatch a caller can make between a pipeline
// and the buffers handed to it.
func errBindings(given, want int) error {
	return fmt.Errorf("vk: the pipeline reads %d buffers, given %d", want, given)
}
