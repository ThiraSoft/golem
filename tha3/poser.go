package tha3

import (
	"fmt"
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

	image, eyebrow, background Tensor
	timings                    Timings

	trace tracer // tests only
}

// Timings is how long each network took on the last call, by network name.
type Timings map[string]time.Duration

// Open loads the five networks from dir, which Dir usually names.
func Open(dir string) (*Poser, error) {
	p := &Poser{timings: Timings{}}
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

// Close releases the weight files. The networks hold copies of what they
// read, so this may be called as soon as Open returns.
func (p *Poser) Close() error {
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
	if img.C != 4 || img.H != Size || img.W != Size {
		return fmt.Errorf("tha3: picture is %dx%dx%d, want 4x%dx%d", img.C, img.H, img.W, Size, Size)
	}
	// Clone the image so callers cannot mutate the stored picture.
	p.image = img.Clone()
	if p.gpu != nil {
		return p.gpu.setImage(p, p.image)
	}
	crop := img.Crop(64, 192, 128, 128)
	p.trace.emit(netEyebrowDecomposer+".in.0", crop)
	start := time.Now()
	p.eyebrow, p.background = p.decomposer.forward(crop, p.trace.sub(netEyebrowDecomposer))
	p.timings[netEyebrowDecomposer] = time.Since(start)
	return nil
}

// Pose renders the picture with the given pose, following
// FiveStepPoserComputationProtocol in the demo's separable_float.py.
func (p *Poser) Pose(pose [NumParams]float32) (Tensor, error) {
	if p.image.Data == nil {
		return Tensor{}, fmt.Errorf("tha3: Pose before SetImage")
	}
	if p.gpu != nil {
		return p.gpu.pose(p, pose)
	}
	eyebrowPose := pose[:eyebrowParams]
	facePose := pose[eyebrowParams:faceParamsEnd]
	rotationPose := pose[faceParamsEnd:]
	timed := func(network string, run func()) {
		start := time.Now()
		run()
		p.timings[network] = time.Since(start)
	}

	var eyebrows Tensor
	p.trace.emit(netEyebrowCombiner+".in.0", p.background)
	p.trace.emit(netEyebrowCombiner+".in.1", p.eyebrow)
	timed(netEyebrowCombiner, func() {
		eyebrows = p.combiner.forward(p.background, p.eyebrow, eyebrowPose, p.trace.sub(netEyebrowCombiner))
	})

	faceIn := p.image.Crop(32, 160, 192, 192)
	faceIn.Paste(eyebrows, 32, 32)
	p.trace.emit(netFaceMorpher+".in.0", faceIn)
	var face Tensor
	timed(netFaceMorpher, func() {
		face = p.face.forward(faceIn, facePose, p.trace.sub(netFaceMorpher))
	})

	full := p.image.Clone()
	full.Paste(face, 32, 160)
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
	return frame, nil
}

// Timings returns how long each network took on the last SetImage and Pose.
func (p *Poser) Timings() Timings {
	out := Timings{}
	for k, v := range p.timings {
		out[k] = v
	}
	return out
}
