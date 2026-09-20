package tha3

import (
	"errors"
	"fmt"
	"image"
	"os"
	"path/filepath"
	"time"
)

// The five networks, by the name of their file and of their waypoints.
const (
	netEyebrowDecomposer = "eyebrow_decomposer"
	netEyebrowCombiner   = "eyebrow_morphing_combiner"
	netFaceMorpher       = "face_morpher"
	netRotator           = "two_algo_face_body_rotator"
	netEditor            = "editor"
)

// Dir is where the converted weights are: $GOLEM_THA3 when set, else
// ~/.cache/golem/tha3/separable_float, where ref/tha3/README.md puts them.
func Dir() string {
	if d := os.Getenv("GOLEM_THA3"); d != "" {
		return d
	}
	home, _ := os.UserHomeDir()
	return filepath.Join(home, ".cache", "golem", "tha3", "separable_float")
}

// Poser holds the five networks and the picture they animate. Pose is not safe
// for concurrent use: it writes the timings map, and on the card it rewrites the
// pose buffer.
type Poser struct {
	decomposer *eyebrowDecomposer
	combiner   *eyebrowCombiner
	face       *faceMorpher
	rotator    *rotator
	editor     *editor
	files      []*weights
	gpu        *gpuPoser // set by UseVulkan
	gpuTrace   bool      // the card keeps every waypoint, for the tests
	view       image.Rectangle
	closed     bool

	// The last pose rendered and its frame, reused while neither changes.
	// The processor also keeps what the first two stages made, for a pose
	// that changes only what comes after; the card keeps its own.
	lastPose  [NumParams]float32
	lastFrame Tensor
	haveLast  bool
	renders   int   // poses actually computed, for the tests
	started   stage // where the last computed pose started, for the tests
	eyebrows  Tensor
	morphed   Tensor

	// What each network decided on the processor, applied again to the
	// larger picture when the scale is above one, and what that made.
	decomposed decomposerFields
	combined   combinerFields
	faced      faceFields
	edited     editorFields
	scale      int
	// front is the mask of what the picture keeps in front of the morphed
	// face, as SetFront was given it, and frontFace and hiFront are that
	// mask refined on the face's window at each resolution.
	front, frontFace, hiFront Tensor
	sharpen                   float32 // the unsharp mask on what the face morpher paints
	toneAmount                float32 // how far the eyes' colour is brought onto the picture's skin
	eyeTone                   [3]float32
	high                      Tensor
	hiEyebrow, hiBackground   Tensor
	hiEyebrows, hiMorphed     Tensor

	// heldBody reuses the rotator's and the editor's last decisions when
	// only the face and the brows move.
	heldBody bool

	// After SetBackground, frames come out finished, zoomed by zoom.
	finish bool
	bg     [3]uint8
	zoom   Zoom

	image, eyebrow, background Tensor
	timings                    Timings

	trace tracer // tests only
}

// whole is the frame the networks make, and the view until SetView.
var whole = image.Rect(0, 0, Size, Size)

// Timings is how long each network took on the last call, by network name.
type Timings map[string]time.Duration

// errClosed is what a poser answers once Close has run: its weights and its
// card are gone, and answering from what is left would be a silent fallback.
var errClosed = errors.New("tha3: poser is closed")

// Open loads the five networks from dir, which Dir usually names.
func Open(dir string) (*Poser, error) {
	p := &Poser{timings: Timings{}, view: whole, scale: 1, eyeTone: noTone}
	byName := map[string]*weights{}
	for _, n := range []string{netEyebrowDecomposer, netEyebrowCombiner, netFaceMorpher, netRotator, netEditor} {
		w, err := openWeights(dir, n)
		if err != nil {
			p.Close()
			return nil, err
		}
		p.files = append(p.files, w)
		byName[n] = w
	}
	p.decomposer = newEyebrowDecomposer(byName[netEyebrowDecomposer])
	p.combiner = newEyebrowCombiner(byName[netEyebrowCombiner])
	p.face = newFaceMorpher(byName[netFaceMorpher])
	p.rotator = newRotator(byName[netRotator])
	p.editor = newEditor(byName[netEditor])
	for _, w := range p.files {
		if err := w.err(); err != nil {
			p.Close()
			return nil, err
		}
	}
	return p, nil
}

// Close releases the weight files and, after UseVulkan, the card. The poser
// cannot be used afterwards: SetImage, Pose and UseVulkan return an error.
func (p *Poser) Close() error {
	p.closed = true
	if p.gpu != nil {
		p.gpu.close()
		p.gpu = nil
	}
	var first error
	for _, w := range p.files {
		if err := w.close(); err != nil && first == nil {
			first = err
		}
	}
	p.files = nil
	return first
}

// SetImage sets the picture to animate and runs the eyebrow decomposer on it,
// which depends on nothing else and is kept for every pose that follows.
// With a scale above one, the larger picture is the picture resized: see
// SetImageHigh for one that has more to show.
func (p *Poser) SetImage(img Tensor) error { return p.SetImageHigh(img, Tensor{}) }

// SetImageHigh is SetImage with the larger picture Pose draws from when the
// scale is above one: the same picture, 512 times the scale on a side, so
// that what the networks only move keeps its detail. An empty high is img
// resized; at a scale of one, high is not used.
func (p *Poser) SetImageHigh(img, high Tensor) error {
	if p.closed {
		return errClosed
	}
	if img.C != 4 || img.H != Size || img.W != Size {
		return fmt.Errorf("tha3: picture is %dx%dx%d, want 4x%dx%d", img.C, img.H, img.W, Size, Size)
	}
	n := Size * p.scale
	switch {
	case p.scale == 1:
		high = Tensor{}
	case high.Data == nil:
		high = ResizeBilinear(img, n, n)
	case high.C != 4 || high.H != n || high.W != n:
		return fmt.Errorf("tha3: larger picture is %dx%dx%d, want 4x%dx%d", high.C, high.H, high.W, n, n)
	default:
		high = high.Clone()
	}
	// Clone the image so callers cannot mutate the stored picture. It is
	// stored only once the card, if there is one, holds it too.
	pic := img.Clone()
	p.haveLast = false
	p.eyeTone = p.measureEyeTone(pic)
	if p.gpu != nil {
		if err := p.gpu.setImage(p, pic, high); err != nil {
			return err
		}
		p.image, p.high = pic, high
		p.cutFront()
		return p.gpu.setFront(p)
	}
	p.image, p.high = pic, high
	p.cutFront()
	crop := img.Crop(64, 192, 128, 128)
	p.trace.emit(netEyebrowDecomposer+".in.0", crop)
	start := time.Now()
	p.eyebrow, p.background, p.decomposed = p.decomposer.forward(crop, p.trace.sub(netEyebrowDecomposer))
	if p.high.Data != nil {
		k := p.high.H / Size
		p.hiEyebrow, p.hiBackground = p.decomposed.resized(128*k, 128*k).apply(p.high.Crop(64*k, 192*k, 128*k, 128*k))
	}
	p.timings[netEyebrowDecomposer] = time.Since(start)
	return nil
}

// stage is where a pose is computed from. Each stage reads its own share of
// the pose and what the stage before it made, so a pose that changes only
// the later shares starts later.
type stage int

const (
	stageEyebrows stage = iota // the eyebrow combiner, on pose[:eyebrowParams]
	stageFace                  // the face morpher, on pose[eyebrowParams:faceParamsEnd]
	stageBody                  // the rotator and the editor, on the rest
)

// stageNetworks is what each stage runs, for the timings of those skipped.
var stageNetworks = [...][]string{
	stageEyebrows: {netEyebrowCombiner},
	stageFace:     {netFaceMorpher},
	stageBody:     {netRotator, netEditor},
}

// firstChange is the stage the pose must be computed from, given the last
// one, or false when nothing changed at all.
func firstChange(last, pose [NumParams]float32) (stage, bool) {
	switch {
	case [eyebrowParams]float32(last[:eyebrowParams]) != [eyebrowParams]float32(pose[:eyebrowParams]):
		return stageEyebrows, true
	case [faceParamsEnd - eyebrowParams]float32(last[eyebrowParams:faceParamsEnd]) !=
		[faceParamsEnd - eyebrowParams]float32(pose[eyebrowParams:faceParamsEnd]):
		return stageFace, true
	case last != pose:
		return stageBody, true
	}
	return 0, false
}

// sameBody reports whether two poses turn the head and the body the same
// way, which is what the rotator and the editor read.
func sameBody(a, b [NumParams]float32) bool {
	return [NumParams - faceParamsEnd]float32(a[faceParamsEnd:]) == [NumParams - faceParamsEnd]float32(b[faceParamsEnd:])
}

// poseCPU runs the five networks on the processor, following FiveStepPoserComputationProtocol in the demo's separable_float.py,
// from stage from: what the stages before it made is kept from the last pose.
func (p *Poser) poseCPU(pose [NumParams]float32, from stage, short bool) (Tensor, error) {
	eyebrowPose := pose[:eyebrowParams]
	facePose := pose[eyebrowParams:faceParamsEnd]
	rotationPose := pose[faceParamsEnd:]
	timed := func(network string, run func()) {
		start := time.Now()
		run()
		p.timings[network] = time.Since(start)
	}

	if from <= stageEyebrows {
		p.trace.emit(netEyebrowCombiner+".in.0", p.background)
		p.trace.emit(netEyebrowCombiner+".in.1", p.eyebrow)
		timed(netEyebrowCombiner, func() {
			p.eyebrows, p.combined = p.combiner.forward(p.background, p.eyebrow, eyebrowPose, p.trace.sub(netEyebrowCombiner))
			if p.high.Data != nil {
				k := p.high.H / Size
				p.hiEyebrows = p.combined.resized(128*k, 128*k).apply(p.hiBackground, p.hiEyebrow)
			}
		})
	}

	if from <= stageFace {
		faceIn := p.image.Crop(32, 160, 192, 192)
		faceIn.Paste(p.eyebrows, 32, 32)
		p.trace.emit(netFaceMorpher+".in.0", faceIn)
		timed(netFaceMorpher, func() {
			p.morphed, p.faced = p.face.forward(faceIn, facePose, p.eyeTone, p.trace.sub(netFaceMorpher))
			p.morphed = composeFront(p.frontFace, faceIn, p.morphed)
			if p.high.Data != nil {
				k := p.high.H / Size
				hiIn := p.high.Crop(32*k, 160*k, 192*k, 192*k)
				hiIn.Paste(p.hiEyebrows, 32*k, 32*k)
				p.hiMorphed = p.faced.resized(192*k, 192*k).applySharp(hiIn, p.sharpen)
				p.hiMorphed = composeFront(p.hiFront, hiIn, p.hiMorphed)
			}
		})
	}

	if short {
		// The rotator and the editor keep what they decided for the
		// last pose: the face alone moved under their warp.
		p.timings[netRotator], p.timings[netEditor] = 0, 0
		return p.compose(), nil
	}

	full := p.image.Clone()
	full.Paste(p.morphed, 32, 160)
	half := ResizeBilinear(full, Size/2, Size/2)
	p.trace.emit(netRotator+".in.0", half)
	var warped, grid Tensor
	timed(netRotator, func() {
		warped, grid = p.rotator.forward(half, rotationPose, p.trace.sub(netRotator))
	})

	warped = ResizeBilinear(warped, Size, Size)
	grid = ResizeBilinear(grid, Size, Size)
	p.trace.emit(netEditor+".in.0", full)
	p.trace.emit(netEditor+".in.1", warped)
	p.trace.emit(netEditor+".in.2", grid)
	var frame Tensor
	timed(netEditor, func() {
		frame, p.edited = p.editor.forward(full, warped, grid, rotationPose, p.trace.sub(netEditor))
		if p.finish || p.high.Data != nil {
			frame = p.compose()
		}
	})
	if p.finish || p.high.Data != nil {
		return frame, nil
	}
	if v := p.view; v != whole {
		frame = frame.Crop(v.Min.Y, v.Min.X, v.Dy(), v.Dx())
	}
	return frame, nil
}

// compose is the last step on the processor, from what the editor decided:
// the larger picture warped and repainted, finished for the screen or not,
// cut to the view.
func (p *Poser) compose() Tensor {
	high, face := p.high, p.hiMorphed
	if high.Data == nil {
		high, face = p.image, p.morphed
	}
	if p.finish {
		return p.finishCPU(high, face)
	}
	k, v := high.H/Size, p.view
	full := high.Clone()
	full.Paste(face, 32*k, 160*k)
	frame := p.edited.resized(high.H, high.W).apply(full)
	return frame.Crop(v.Min.Y*k, v.Min.X*k, v.Dy()*k, v.Dx()*k)
}

// Pose renders the picture with the given pose. A pose equal to the last one,
// on the same picture, returns a copy of the last frame without computing
// anything, so a caller that asks for the same pose again costs nothing. A
// pose that keeps the eyebrows of the last one skips the eyebrow combiner,
// and one that also keeps the face skips the face morpher: breathing and
// turning the head run only the rotator and the editor.
func (p *Poser) Pose(pose [NumParams]float32) (Tensor, error) {
	if p.finish {
		return Tensor{}, fmt.Errorf("tha3: Pose after SetBackground: use PoseRGBA")
	}
	return p.render(pose, Zoom{})
}

func (p *Poser) render(pose [NumParams]float32, zoom Zoom) (Tensor, error) {
	if p.closed {
		return Tensor{}, errClosed
	}
	if p.image.Data == nil {
		return Tensor{}, fmt.Errorf("tha3: Pose before SetImage")
	}
	from := stageEyebrows
	if p.haveLast {
		changed, ok := firstChange(p.lastPose, pose)
		switch {
		case !ok && zoom == p.zoom:
			return p.lastFrame.Clone(), nil
		case !ok:
			// Only the zoom moved, which the last step alone reads.
			changed = stageBody
		}
		from = changed
	}
	p.zoom = zoom
	for _, skipped := range stageNetworks[:from] {
		for _, name := range skipped {
			p.timings[name] = 0
		}
	}
	// With the body held, a pose that leaves the rotation alone reuses
	// the warp and the repaint of the last one, and neither the rotator
	// nor the editor runs. The brows and the face may have moved; the
	// rotation, which those two read, may not.
	short := p.heldBody && p.haveLast && sameBody(p.lastPose, pose)
	var frame Tensor
	var err error
	if p.gpu != nil {
		frame, err = p.gpu.pose(p, pose, from, short)
	} else {
		frame, err = p.poseCPU(pose, from, short)
	}
	if err != nil {
		// What the stages kept may be half written: start over next time.
		p.haveLast = false
		return Tensor{}, err
	}
	p.renders++
	p.started = from
	p.lastPose, p.lastFrame, p.haveLast = pose, frame, true
	return frame.Clone(), nil
}

// SetView makes Pose return only the part r of the frame. The networks still
// compute the whole frame, which their norms need, but the card sends back
// only r: a 224×224 view is a fifth of the floats of the whole 512×512 frame.
// On the card a new view rebuilds the passes, so it is set once, early.
func (p *Poser) SetView(r image.Rectangle) error {
	if p.closed {
		return errClosed
	}
	if r.Empty() || !r.In(whole) {
		return fmt.Errorf("tha3: view %v is not inside the %dx%d frame", r, Size, Size)
	}
	if r == p.view {
		return nil
	}
	old := p.view
	p.view = r
	p.haveLast = false
	if p.gpu == nil {
		return nil
	}
	p.gpu.close()
	p.gpu = nil
	if err := p.useVulkan(p.gpuTrace); err != nil {
		// Back to the view the card had, on the processor: the caller
		// learns the card is gone rather than finding out by the speed.
		p.view = old
		return fmt.Errorf("tha3: the card for the new view: %w", err)
	}
	return nil
}

// Timings returns how long each network took on the last SetImage and Pose.
func (p *Poser) Timings() Timings {
	out := Timings{}
	for k, v := range p.timings {
		out[k] = v
	}
	return out
}
