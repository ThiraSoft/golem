package vk

// The calibration of a .golem, on the card.
//
// Converting a checkpoint is two halves, and this is the one that runs the
// model. What it needs is not the model's answers but what its matrices are
// fed: the per-column power of the activations at four sites in every block,
// which is what cmd/golemquant turns into the salience scale each site's
// columns are quantized against.
//
// It ran on the processor because the tap is a line inside the block code and
// that code is the processor's. Eight thousand tokens of a twenty-seven
// billion parameter model is an hour and a half of eight cores; the same pass
// on the card is what a prompt of that length costs, which is seconds. So the
// tap moved here: one accumulator a site, a vector of the site's own width,
// added to where the activations already are.
//
// The four sites are nn/golem.go's, and the names are the ones
// cmd/golemquant files them under. A checkpoint the card can read is not the
// checkpoint being converted — this pipeline reads llama.cpp's types and the
// converter's input may be any of them — but the statistics are the same
// statistics: what a site is fed does not depend on which four-bit form the
// weights that fed it were stored in.

import (
	_ "embed"
	"fmt"
	"unsafe"
)

//go:generate glslc -O --target-env=vulkan1.1 -fshader-stage=compute shaders/accum_sq.comp -o shaders/accum_sq.spv

//go:embed shaders/accum_sq.spv
var accumSqSPIRV []byte

type accumSqPush struct {
	n       uint32
	columns uint32
}

// calibSite is one accumulator: the buffer it reads, its width, and where the
// sum lives.
type calibSite struct {
	key   string
	n     int
	acc   *Buffer
	set   *Set
	back  *Buffer
	total *Set // unused; kept nil, the readback is a copy
}

// qwenCalib is every accumulator of a pipeline in calibration.
type qwenCalib struct {
	pipe  *Pipeline
	sites []*calibSite
	byKey map[string]*calibSite
	rows  int
}

// StartCalibration allocates one accumulator a site and makes every recording
// carry the dispatches that fill them.
//
// It forgets the compiled passes: a recording laid down before this would run
// the model and count nothing, and would be indistinguishable from one that
// counted.
func (p *QwenPipeline) StartCalibration() error {
	if p.calib != nil {
		return nil
	}
	if p.usesGolem() {
		// The activations a .golem meets have already been through the site's
		// own scale and rotation, so their power is not what the salience is
		// measured from. A calibration reads a checkpoint of one of
		// llama.cpp's types, which is what it is measuring for.
		return fmt.Errorf("vk: a .golem checkpoint is the output of a calibration, not its input")
	}
	c := &qwenCalib{byKey: map[string]*calibSite{}}
	var err error
	if c.pipe, err = p.d.NewPipeline(accumSqSPIRV, 2, uint32(unsafe.Sizeof(accumSqPush{}))); err != nil {
		return err
	}
	s := p.shape
	// The head's site, which belongs to no block and so is filed under -1 —
	// the same key cmd/golemquant gives it. What it reads is what the final
	// norm made, which vk/qwen_pipeline.go writes to stage.
	if !p.noHead {
		head := &calibSite{key: "-1/head", n: s.Dim}
		if head.acc, err = p.d.Local(s.Dim*4, bufferUsageStorage|bufferUsageTransferSrc|bufferUsageTransferDst); err != nil {
			c.close()
			return err
		}
		if head.back, err = p.d.Readback(s.Dim*4, bufferUsageTransferDst); err != nil {
			c.close()
			return err
		}
		if head.set, err = c.pipe.NewSet([]*Buffer{p.stage, head.acc}); err != nil {
			c.close()
			return err
		}
		c.sites = append(c.sites, head)
		c.byKey[head.key] = head
	}

	blocks := min(len(p.isSSM), len(p.ffnBlocks))
	for i := 0; i < blocks; i++ {
		out, outWidth := p.attnOut, s.qDim()
		if p.isSSM[i] {
			out, outWidth = p.ySSM, s.Inner
		}
		for _, site := range []struct {
			name string
			buf  *Buffer
			n    int
		}{
			{"qkv", p.normed, s.Dim},
			{"o", out, outWidth},
			{"gateup", p.ffnNorm, s.Dim},
			{"down", p.actBuf, s.FFN},
		} {
			cs := &calibSite{key: fmt.Sprintf("%d/%s", p.blockBase+i, site.name), n: site.n}
			if cs.acc, err = p.d.Local(site.n*4, bufferUsageStorage|bufferUsageTransferSrc|bufferUsageTransferDst); err != nil {
				c.close()
				return err
			}
			if cs.back, err = p.d.Readback(site.n*4, bufferUsageTransferDst); err != nil {
				c.close()
				return err
			}
			if cs.set, err = c.pipe.NewSet([]*Buffer{site.buf, cs.acc}); err != nil {
				c.close()
				return err
			}
			c.sites = append(c.sites, cs)
			c.byKey[cs.key] = cs
		}
	}
	if err := p.d.Submit(func(r *Recorder) {
		for _, cs := range c.sites {
			r.Fill(cs.acc, 0)
		}
	}); err != nil {
		c.close()
		return err
	}
	p.calib = c
	p.forgetPasses()
	return nil
}

// accumulate is one site's dispatch, and nothing at all when no calibration is
// running.
func (p *QwenPipeline) accumulate(r *Recorder, block int, site string, columns int) {
	if p.calib == nil {
		return
	}
	if block >= 0 {
		block += p.blockBase
	}
	cs := p.calib.byKey[fmt.Sprintf("%d/%s", block, site)]
	if cs == nil {
		return
	}
	push := accumSqPush{n: uint32(cs.n), columns: uint32(columns)}
	r.Dispatch(cs.set, uint32((cs.n+255)/256), unsafe.Pointer(&push))
	r.Barrier()
}

// CountCalibration tells the accumulators how many rows they have seen, which
// nothing on the card can know: a pass writes the columns it was given and the
// host is what gave them.
func (p *QwenPipeline) CountCalibration(rows int) {
	if p.calib != nil {
		p.calib.rows += rows
	}
}

// CalibrationSums is the per-column power of every site, and how many rows it
// was taken over. The key is the block and the site — "7/qkv" — which is what
// cmd/golemquant files a vector under.
func (p *QwenPipeline) CalibrationSums() (map[string][]float32, int, error) {
	if p.calib == nil {
		return nil, 0, fmt.Errorf("vk: no calibration is running")
	}
	if err := p.d.Submit(func(r *Recorder) {
		for _, cs := range p.calib.sites {
			r.Copy(cs.back, 0, cs.acc, cs.n*4)
		}
	}); err != nil {
		return nil, 0, err
	}
	out := make(map[string][]float32, len(p.calib.sites))
	for _, cs := range p.calib.sites {
		v := make([]float32, cs.n)
		copy(v, cs.back.Floats()[:cs.n])
		out[cs.key] = v
	}
	return out, p.calib.rows, nil
}

// StopCalibration frees the accumulators and puts the recordings back.
func (p *QwenPipeline) StopCalibration() {
	if p.calib == nil {
		return
	}
	p.calib.close()
	p.calib = nil
	p.forgetPasses()
}

func (c *qwenCalib) close() {
	for _, cs := range c.sites {
		if cs.set != nil {
			cs.set.Close()
		}
		for _, b := range []*Buffer{cs.acc, cs.back} {
			if b != nil {
				b.Close()
			}
		}
	}
	c.sites = nil
	c.byKey = nil
	if c.pipe != nil {
		c.pipe.Close()
		c.pipe = nil
	}
}

// SetBlockWindow says which block of the model this pipeline's first block is,
// and whether this window holds the last of them. It has to be called before
// StartCalibration, which is what reads it.
//
// A pipeline that holds the whole trunk needs neither: the base is zero and the
// head is here.
func (p *QwenPipeline) SetBlockWindow(base int, last bool) {
	p.blockBase = base
	p.noHead = !last
}
