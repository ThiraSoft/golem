package tha3

import (
	"fmt"
	"sort"
)

// The face morpher paints the closed eyelid in a skin it has learned, not in
// the character's: on every picture tried it comes out warmer than the face
// around it, short of blue by a tenth and more. At 512 the whole frame is the
// networks' and the patch passes; on the larger picture it is a beige lid on
// a face the picture drew, and it reads as a mask. The gain below measures
// that drift once, on the picture itself, and the morpher's colour is
// multiplied by it from then on.

// toneLow and toneHigh are how far the gain may pull a channel. The drift measured on the
// characters at hand is within a sixth either way; further than this means
// the measurement found something that is not skin, and the colour is better
// left alone than dragged.
const toneLow, toneHigh = 0.75, 1.35

// noTone is the gain that changes nothing.
var noTone = [3]float32{1, 1, 1}

// eyeTone is the gain that brings the skin the morpher paints over a closed
// eye onto the skin around it, channel by channel, in light. The fields are
// what the morpher decided for a pose that shuts both eyes, and under is the
// picture it painted them on, warped and with the mouth done, so that what is
// compared is what the frame would have shown.
//
// The lid is the mask's inside, the skin around it the ring of untouched
// picture just outside; each is read at its bright end, so that the lashes in
// one and the hair in the other weigh nothing. A picture where either is too
// small to measure keeps its colour.
func eyeTone(f faceFields, under Tensor, amount float32) [3]float32 {
	if amount <= 0 {
		return noTone
	}
	lid, ring := maskRegions(f.eyeAlpha)
	if len(lid) < toneFloor || len(ring) < toneFloor {
		return noTone
	}
	painted, skin := brightMean(f.eyeColor, lid), brightMean(under, ring)
	var g [3]float32
	for c := range 3 {
		if painted[c] <= 0 {
			return noTone
		}
		g[c] = min(max(skin[c]/painted[c], toneLow), toneHigh)
		g[c] = 1 + amount*(g[c]-1)
	}
	return g
}

// toneFloor is how many pixels of the 192×192 fields a region needs before
// its mean says anything. A shut eye paints some seven hundred.
const toneFloor = 64

// toneRing is how far around the painted lid the skin is read, in pixels of
// the fields. Eight is wide enough to hold skin on every side of the eye and
// short enough to stay on the face.
const toneRing = 8

// maskRegions is the inside of a mask and the ring of untouched picture
// around it: the pixels the morpher paints outright, and the pixels it leaves
// alone within toneRing of one of them.
func maskRegions(mask Tensor) (inside, ring []int) {
	a := mask.Plane(0)
	w, h := mask.W, mask.H
	near := make([]bool, len(a))
	for i, v := range a {
		if v < 0.6 {
			continue
		}
		inside = append(inside, i)
		y0, x0 := i/w, i%w
		for y := max(y0-toneRing, 0); y <= min(y0+toneRing, h-1); y++ {
			for x := max(x0-toneRing, 0); x <= min(x0+toneRing, w-1); x++ {
				near[y*w+x] = true
			}
		}
	}
	for i, v := range a {
		if near[i] && v <= 0.02 {
			ring = append(ring, i)
		}
	}
	return inside, ring
}

// toneBright is the share of a region read as its skin: the brightest four
// tenths. A closed eye is skin and lashes, the ring around it skin and hair,
// and in both the skin is what is bright.
const toneBright = 0.6

// brightMean is the mean colour, in light, of the brightest pixels of t at
// the indices given.
func brightMean(t Tensor, at []int) [3]float32 {
	light := func(c, i int) float32 { return (t.Plane(c)[i] + 1) / 2 }
	byLight := make([]int, len(at))
	copy(byLight, at)
	sort.Slice(byLight, func(a, b int) bool {
		var la, lb float32
		for c := range 3 {
			la += lumaWeights[c] * light(c, byLight[a])
			lb += lumaWeights[c] * light(c, byLight[b])
		}
		return la < lb
	})
	bright := byLight[int(float32(len(byLight))*toneBright):]
	var sum [3]float32
	for _, i := range bright {
		for c := range 3 {
			sum[c] += light(c, i)
		}
	}
	for c := range 3 {
		sum[c] /= float32(len(bright))
	}
	return sum
}

// toned is the colour the morpher paints over the eyes, multiplied channel by
// channel in light by the gain and clamped to the range. The alpha channel is
// left alone: the gain moves the colour, not where it goes.
func toned(color Tensor, g [3]float32) Tensor {
	if g == noTone {
		return color
	}
	out := color.Clone()
	for c := range 3 {
		p := out.Plane(c)
		for i, v := range p {
			p[i] = min(max(g[c]*(v+1)/2, 0), 1)*2 - 1
		}
	}
	return out
}

// shutEyes is the pose the gain is measured on: both eyes closed and nothing
// else, which is what paints the whole lid.
func shutEyes() []float32 {
	var pose [NumParams]float32
	pose[EyeWinkLeft], pose[EyeWinkRight] = 1, 1
	return pose[eyebrowParams:faceParamsEnd]
}

// SetEyeTone sets how far the colour the face morpher paints over the eyes is
// brought onto the picture's own skin. Zero, the default, is the networks'
// colour as it comes; one moves it the whole way, which is what a character
// drawn in any palette but the networks' wants; above one it is dragged
// further, up to the range the gain is held in. It costs one run of the face
// morpher on the processor for each picture set, and nothing per frame.
//
// It works at every scale, but it is at a scale above one that it matters:
// there the lid is painted next to skin the picture drew, and a drift of a
// tenth reads as a patch.
func (p *Poser) SetEyeTone(amount float32) error {
	if p.closed {
		return errClosed
	}
	if amount < 0 {
		return fmt.Errorf("tha3: an eye tone of %v is below zero", amount)
	}
	if amount == p.toneAmount {
		return nil
	}
	p.toneAmount = amount
	p.haveLast = false
	if p.image.Data == nil {
		p.eyeTone = noTone
		return nil
	}
	p.eyeTone = p.measureEyeTone(p.image)
	return nil
}

// EyeTone is the gain SetEyeTone measured on the picture, channel by channel.
func (p *Poser) EyeTone() [3]float32 { return p.eyeTone }

// measureEyeTone runs the face morpher once, on img with both eyes shut, and
// reads the skin it paints against the skin around it.
//
// The morpher is fed the face as the picture has it, without the brows the
// decomposer and the combiner would have redrawn over it. At rest those two
// put back very nearly what they were given, and what is measured here is the
// skin beside the eyes, which they do not touch at all.
func (p *Poser) measureEyeTone(img Tensor) [3]float32 {
	if p.toneAmount <= 0 {
		return noTone
	}
	crop := img.Crop(32, 160, 192, 192)
	_, f := p.face.forward(crop, shutEyes(), noTone, nil)
	under := applyColorChange(f.mouthAlpha, f.mouthColor, applyGridChange(f.grid, crop))
	return eyeTone(f, under, p.toneAmount)
}
