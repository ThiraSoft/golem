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
	p := &Poser{timings: Timings{}, view: whole}
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
func (p *Poser) SetImage(img Tensor) error {
	if p.closed {
		return errClosed
	}
	if img.C != 4 || img.H != Size || img.W != Size {
		return fmt.Errorf("tha3: picture is %dx%dx%d, want 4x%dx%d", img.C, img.H, img.W, Size, Size)
	}
	// Clone the image so callers cannot mutate the stored picture. It is
	// stored only once the card, if there is one, holds it too.
	pic := img.Clone()
	p.haveLast = false
	if p.gpu != nil {
		if err := p.gpu.setImage(p, pic); err != nil {
			return err
		}
		p.image = pic
		return nil
	}
	p.image = pic
	crop := img.Crop(64, 192, 128, 128)
	p.trace.emit(netEyebrowDecomposer+".in.0", crop)
	start := time.Now()
	p.eyebrow, p.background = p.decomposer.forward(crop, p.trace.sub(netEyebrowDecomposer))
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

// poseCPU runs the five networks on the processor, following FiveStepPoserComputationProtocol in the demo's separable_float.py,
// from stage from: what the stages before it made is kept from the last pose.
func (p *Poser) poseCPU(pose [NumParams]float32, from stage) (Tensor, error) {
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
			p.eyebrows = p.combiner.forward(p.background, p.eyebrow, eyebrowPose, p.trace.sub(netEyebrowCombiner))
		})
	}

	if from <= stageFace {
		faceIn := p.image.Crop(32, 160, 192, 192)
		faceIn.Paste(p.eyebrows, 32, 32)
		p.trace.emit(netFaceMorpher+".in.0", faceIn)
		timed(netFaceMorpher, func() {
			p.morphed = p.face.forward(faceIn, facePose, p.trace.sub(netFaceMorpher))
		})
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
		frame = p.editor.forward(full, warped, grid, rotationPose, p.trace.sub(netEditor))
	})
	if v := p.view; v != whole {
		frame = frame.Crop(v.Min.Y, v.Min.X, v.Dy(), v.Dx())
	}
	return frame, nil
}

// Pose renders the picture with the given pose. A pose equal to the last one,
// on the same picture, returns a copy of the last frame without computing
// anything, so a caller that asks for the same pose again costs nothing. A
// pose that keeps the eyebrows of the last one skips the eyebrow combiner,
// and one that also keeps the face skips the face morpher: breathing and
// turning the head run only the rotator and the editor.
func (p *Poser) Pose(pose [NumParams]float32) (Tensor, error) {
	if p.closed {
		return Tensor{}, errClosed
	}
	if p.image.Data == nil {
		return Tensor{}, fmt.Errorf("tha3: Pose before SetImage")
	}
	from := stageEyebrows
	if p.haveLast {
		changed, ok := firstChange(p.lastPose, pose)
		if !ok {
			return p.lastFrame.Clone(), nil
		}
		from = changed
	}
	for _, skipped := range stageNetworks[:from] {
		for _, name := range skipped {
			p.timings[name] = 0
		}
	}
	var frame Tensor
	var err error
	if p.gpu != nil {
		frame, err = p.gpu.pose(p, pose, from)
	} else {
		frame, err = p.poseCPU(pose, from)
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
