package vk

import "unsafe"

// A Stream is a command buffer and a fence of its own, so that several
// submissions can be on the card at once and each be waited for alone.
//
// Start and Wait on the Device share one command buffer and wait with
// vkQueueWaitIdle, which is one submission in flight and a wait for all of it.
// That is right for a caller with one thing to do. It is wrong for one with
// several independent batches whose kernels are each a long chain on few
// lanes: the card could run them side by side, and the device's single buffer
// makes them take turns. A Stream is what lets them overlap.
//
// On a device from Open, streams share the main queue: they save the waits,
// but the card still runs their work in turn. On a device from OpenWith with
// ComputeQueues, each new stream takes the next compute queue, and streams on
// different queues run side by side. Device.Wait waits for the main queue
// only, so for streams on it and not for the others.
//
// Queues and command pools are not things Vulkan lets two threads touch at
// once: Start and Close on every stream of a device, and everything else that
// submits on it, come from one goroutine. Wait only waits on the stream's
// fence and may be called from another, for a stream not being started.
type Stream struct {
	d       *Device
	queue   queue
	pool    uint64
	cb      commandBuffer
	fence   uint64
	pending bool
}

// Stream allocates a command buffer and a fence for one submission at a time.
func (d *Device) Stream() (*Stream, error) {
	s := &Stream{d: d, queue: d.queue, pool: d.cmdPool}
	if len(d.asyncQueues) > 0 {
		s.queue, s.pool = d.asyncQueues[d.nextAsync%len(d.asyncQueues)], d.asyncPool
		d.nextAsync++
	}
	cbai := commandBufferAllocateInfo{
		sType:              structCommandBufferAllocateInfo,
		commandPool:        s.pool,
		level:              0,
		commandBufferCount: 1,
	}
	if err := check("vkAllocateCommandBuffers", vkAllocateCommandBuffers(d.dev, &cbai, &s.cb)); err != nil {
		return nil, err
	}
	fci := fenceCreateInfo{sType: structFenceCreateInfo}
	if err := check("vkCreateFence", vkCreateFence(d.dev, &fci, 0, &s.fence)); err != nil {
		s.Close()
		return nil, err
	}
	return s, nil
}

// Start records the stream's command buffer and submits it without waiting.
// The previous submission must have been waited for: the buffer is reused.
//
// Nothing orders one stream's work after another's except barriers, and a
// barrier reaches back over everything submitted before it on the queue,
// other streams included. A stream that wants to run beside the others keeps
// to TransferBarrier between its copies and its dispatch.
func (s *Stream) Start(record func(*Recorder)) error {
	if s.pending {
		if err := s.Wait(); err != nil {
			return err
		}
	}
	if err := check("vkResetCommandBuffer", vkResetCommandBuffer(s.cb, 0)); err != nil {
		return err
	}
	bi := commandBufferBeginInfo{sType: structCommandBufferBeginInfo, flags: commandBufferOneTime}
	if err := check("vkBeginCommandBuffer", vkBeginCommandBuffer(s.cb, &bi)); err != nil {
		return err
	}
	record(&Recorder{cb: s.cb})
	if err := check("vkEndCommandBuffer", vkEndCommandBuffer(s.cb)); err != nil {
		return err
	}
	if err := check("vkResetFences", vkResetFences(s.d.dev, 1, &s.fence)); err != nil {
		return err
	}
	si := submitInfo{
		sType:              structSubmitInfo,
		commandBufferCount: 1,
		pCommandBuffers:    uintptr(unsafe.Pointer(&s.cb)),
	}
	if err := check("vkQueueSubmit", vkQueueSubmit(s.queue, 1, &si, s.fence)); err != nil {
		return err
	}
	s.pending = true
	return nil
}

// Wait blocks until the stream's last submission has run, and nothing else.
// It is harmless when nothing is pending.
func (s *Stream) Wait() error {
	if !s.pending {
		return nil
	}
	if err := check("vkWaitForFences", vkWaitForFences(s.d.dev, 1, &s.fence, 1, ^uint64(0))); err != nil {
		return err
	}
	s.pending = false
	return nil
}

// Done says whether the last submission has run, without blocking. A stream
// with nothing pending is done.
func (s *Stream) Done() bool {
	if !s.pending {
		return true
	}
	return vkGetFenceStatus(s.d.dev, s.fence) == success
}

// Close waits for the stream and releases it.
func (s *Stream) Close() {
	s.Wait()
	if s.fence != 0 {
		vkDestroyFence(s.d.dev, s.fence, 0)
		s.fence = 0
	}
	if s.cb != 0 {
		vkFreeCommandBuffers(s.d.dev, s.pool, 1, &s.cb)
		s.cb = 0
	}
}

// TransferBarrier is Barrier for a recording whose only writes before it are
// copies and fills: what comes after waits for the transfers earlier on the
// queue, and not for the dispatches. It is how a stream puts new data in front
// of its kernel without waiting for every other stream's kernel to finish.
func (r *Recorder) TransferBarrier() {
	b := memoryBarrier{
		sType:         structMemoryBarrier,
		srcAccessMask: accessTransferWrite,
		dstAccessMask: accessShaderRead | accessShaderWrite | accessTransferRead | accessTransferWrite,
	}
	vkCmdPipelineBarrier(r.cb, stageTransfer, stageComputeShader|stageTransfer, 0, 1, &b, 0, 0, 0, 0)
}
