package krea2

// torch's normal noise, drawn on the processor. ComfyUI draws the starting
// latent with torch's CPU generator and the noise ER-SDE adds at every step
// with a generator on the card; on the machine the mobile front runs on the
// card is ROCm, so that one is rocrand's Philox4x32-10, spread over threads the
// way torch launches its kernel. Both are reproduced here to the bit before the
// floating point, which is what makes a seed give ComfyUI's picture.

import "math"

// TorchCPURandn is torch.randn(n) from a generator seeded with seed on the
// CPU: mt19937, a 24-bit uniform per draw, then torch's Box-Muller over blocks
// of sixteen (the first eight radii, the last eight angles). A length that is
// not a multiple of sixteen redraws the last sixteen, as torch does.
func TorchCPURandn(seed uint64, n int) []float32 {
	m := newMT19937(uint32(seed))
	d := make([]float32, n)
	if n < 16 {
		// torch takes a scalar path under sixteen, one Box-Muller pair at a
		// time in double precision. Nothing here draws that little noise.
		panic("krea2: TorchCPURandn wants at least sixteen values")
	}
	for i := range d {
		d[i] = m.uniform()
	}
	for i := 0; i+16 <= n; i += 16 {
		boxMuller16(d[i : i+16])
	}
	if n%16 != 0 {
		tail := d[n-16:]
		for i := range tail {
			tail[i] = m.uniform()
		}
		boxMuller16(tail)
	}
	return d
}

func boxMuller16(d []float32) {
	for j := 0; j < 8; j++ {
		u1 := 1 - d[j]
		u2 := d[j+8]
		radius := float32(math.Sqrt(-2 * math.Log(float64(u1))))
		theta := float32(2*math.Pi) * u2
		sin, cos := math.Sincos(float64(theta))
		d[j] = radius * float32(cos)
		d[j+8] = radius * float32(sin)
	}
}

type mt19937 struct {
	state [624]uint32
	next  int
}

func newMT19937(seed uint32) *mt19937 {
	m := &mt19937{next: 624}
	m.state[0] = seed
	for i := 1; i < 624; i++ {
		prev := m.state[i-1]
		m.state[i] = 1812433253*(prev^(prev>>30)) + uint32(i)
	}
	return m
}

func (m *mt19937) draw() uint32 {
	if m.next >= 624 {
		for k := 0; k < 624; k++ {
			y := m.state[k]&0x80000000 | m.state[(k+1)%624]&0x7fffffff
			v := m.state[(k+397)%624] ^ y>>1
			if y&1 != 0 {
				v ^= 0x9908b0df
			}
			m.state[k] = v
		}
		m.next = 0
	}
	y := m.state[m.next]
	m.next++
	y ^= y >> 11
	y ^= y << 7 & 0x9d2c5680
	y ^= y << 15 & 0xefc60000
	y ^= y >> 18
	return y
}

// uniform is torch's uniform_real_distribution<float>: the low 24 bits of a
// draw, over 2^24.
func (m *mt19937) uniform() float32 {
	return float32(m.draw()&(1<<24-1)) * (1.0 / (1 << 24))
}

// The launch torch gives its normal kernel on the card ComfyUI ran on: an RX
// 9070 XT under ROCm 7.2, which reports 32 multiprocessors of 2048 threads.
// torch launches 256 threads a block and at most as many blocks as fill every
// multiprocessor, and each thread draws four values a round.
const (
	cudaBlock          = 256
	cudaMaxBlocks      = 32 * 2048 / cudaBlock
	cudaDrawsARound    = 4
	philoxM0, philoxM1 = 0xD2511F53, 0xCD9E8D57
	philoxW0, philoxW1 = 0x9E3779B9, 0xBB67AE85
)

// CUDARandn is a torch generator on the card, seeded once and asked for
// torch.randn again and again: each call advances its Philox offset the way
// torch advances it, so the second call does not repeat the first.
type CUDARandn struct {
	seed   uint64
	offset uint64
}

// NewCUDARandn is torch.Generator(device="cuda").manual_seed(seed).
func NewCUDARandn(seed uint64) *CUDARandn { return &CUDARandn{seed: seed} }

// Next is the generator's next torch.randn of n values, in float32.
func (g *CUDARandn) Next(n int) []float32 {
	blocks := (n + cudaBlock - 1) / cudaBlock
	if blocks > cudaMaxBlocks {
		blocks = cudaMaxBlocks
	}
	threads := cudaBlock * blocks
	round := threads * cudaDrawsARound
	rounds := (n-1)/round + 1
	out := make([]float32, n)
	for thread := 0; thread < threads; thread++ {
		p := newPhilox(g.seed, uint64(thread), g.offset)
		for at := thread; at < rounds*round; at += round {
			r := p.next4()
			a, b := rocrandBoxMuller(r[0], r[1])
			c, d := rocrandBoxMuller(r[2], r[3])
			for k, v := range [4]float32{a, b, c, d} {
				if i := at + threads*k; i < n {
					out[i] = v
				}
			}
		}
	}
	// torch moves the offset by four draws a round, whatever the draws were.
	g.offset += uint64(rounds * cudaDrawsARound)
	return out
}

// philox is rocrand's philox4x32_10 engine, at a whole counter: torch's
// offsets are always multiples of four, so the engine's sub-counter state is
// never anything but zero and is left out.
type philox struct {
	counter [4]uint32
	key     [2]uint32
	result  [4]uint32
}

func newPhilox(seed, subsequence, offset uint64) *philox {
	p := &philox{key: [2]uint32{uint32(seed), uint32(seed >> 32)}}
	lo, hi := uint32(subsequence), uint32(subsequence>>32)
	z := p.counter[2]
	p.counter[2] += lo
	p.counter[3] += hi
	if p.counter[2] < z {
		p.counter[3]++
	}
	if offset%4 != 0 {
		panic("krea2: a Philox offset torch would not give")
	}
	p.advance(offset / 4)
	p.result = philoxRounds(p.counter, p.key)
	return p
}

func (p *philox) advance(by uint64) {
	old := p.counter
	p.counter[0] += uint32(by)
	p.counter[1] += uint32(by >> 32)
	if p.counter[0] < old[0] {
		p.counter[1]++
	}
	if p.counter[1] < old[1] {
		p.counter[2]++
	}
	if p.counter[2] < old[2] {
		p.counter[3]++
	}
}

func (p *philox) next4() [4]uint32 {
	r := p.result
	p.advance(1)
	p.result = philoxRounds(p.counter, p.key)
	return r
}

func philoxRounds(c [4]uint32, k [2]uint32) [4]uint32 {
	for i := 0; i < 10; i++ {
		if i > 0 {
			k[0] += philoxW0
			k[1] += philoxW1
		}
		m0 := uint64(philoxM0) * uint64(c[0])
		m1 := uint64(philoxM1) * uint64(c[2])
		c = [4]uint32{uint32(m1>>32) ^ c[1] ^ k[0], uint32(m1), uint32(m0>>32) ^ c[3] ^ k[1], uint32(m0)}
	}
	return c
}

// rocrandBoxMuller is rocrand's box_muller for float: uniforms in (0, 1],
// the angle already scaled by 2π, sine first.
func rocrandBoxMuller(x, y uint32) (float32, float32) {
	const inv = float32(2.3283064e-10)     // ROCRAND_2POW32_INV
	const inv2pi = float32(1.46291807e-09) // ROCRAND_2POW32_INV_2PI
	u := inv + float32(x)*inv
	v := inv2pi + float32(y)*inv2pi
	s := float32(math.Sqrt(float64(-2 * float32(math.Log(float64(u))))))
	sin, cos := math.Sincos(float64(v))
	return float32(sin) * s, float32(cos) * s
}
