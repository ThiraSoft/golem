package tha3

import "fmt"

// What the face morpher paints over the eyes, it paints over everything that
// happens to be in front of them: a lock of hair that falls across an eye is
// gone as soon as the eye shuts, and back when it opens. The networks know
// nothing of depth.
//
// A front mask says which pixels of the picture are in front. They are put
// back over the morphed face, before the rotator, so that they still turn
// with the head and breathe with the body. The mask is drawn by hand, on the
// picture, and used as it was drawn: working it out from the colours under a
// rough stroke was tried and dropped, since on a character whose hair is
// near its skin it keeps the whole socket, and on one whose hair is as dark
// as its lashes it keeps the open eye.

// SetFront sets the mask of what the picture keeps in front of the face: one
// channel, 512×512, in the frame's coordinates, one where the picture wins
// and zero where the morpher does, anything between for an edge. An empty
// tensor clears it.
//
// On the card the mask is an input written with the picture, and the
// operation that reads it is always in the graph: setting or clearing a mask
// costs a write, never a rebuild.
func (p *Poser) SetFront(mask Tensor) error {
	if p.closed {
		return errClosed
	}
	if mask.Data != nil && (mask.C != 1 || mask.H != Size || mask.W != Size) {
		return fmt.Errorf("tha3: front mask is %dx%dx%d, want 1x%dx%d", mask.C, mask.H, mask.W, Size, Size)
	}
	if mask.Data != nil {
		mask = mask.Clone()
	}
	p.front = mask
	p.haveLast = false
	p.cutFront()
	if p.gpu != nil {
		return p.gpu.setFront(p)
	}
	return nil
}

// Front is the mask SetFront was given.
func (p *Poser) Front() Tensor { return p.front }

// cutFront cuts the mask to the window the face morpher works in, at the
// scale of the picture and at the scale of the larger one. Blown up, a mask
// drawn at 512 leaves an edge a few pixels wide where the picture and the
// morphed face are mixed, which is what an edge drawn by hand deserves.
func (p *Poser) cutFront() {
	p.frontFace, p.hiFront = Tensor{}, Tensor{}
	if p.front.Data == nil || p.image.Data == nil {
		return
	}
	mask := p.front.Crop(32, 160, 192, 192)
	if !anyOf(mask) {
		return
	}
	p.frontFace = mask
	if p.high.Data != nil {
		k := p.high.H / Size
		p.hiFront = resize(mask, 192*k, 192*k)
	}
}

// anyOf reports whether a mask holds anything at all.
func anyOf(mask Tensor) bool {
	for _, v := range mask.Plane(0) {
		if v > 0.02 {
			return true
		}
	}
	return false
}

// composeFront puts the picture back over the morphed face where the mask
// says the picture is in front. It is applyColorChange, the blend the
// morpher's own fields use, with the mask as its alpha.
func composeFront(mask, picture, morphed Tensor) Tensor {
	if mask.Data == nil {
		return morphed
	}
	return applyColorChange(mask, picture, morphed)
}
