package yue2

import (
	"fmt"
	"math"
)

// mt19937 is the standard 32-bit Mersenne Twister (MT19937), matching C++ std::mt19937.
type mt19937 struct {
	mt  [624]uint32
	idx int
}

func newMT19937(seed uint32) *mt19937 {
	m := &mt19937{}
	m.mt[0] = seed
	for i := 1; i < 624; i++ {
		m.mt[i] = 1812433253*(m.mt[i-1]^(m.mt[i-1]>>30)) + uint32(i)
	}
	m.idx = 624
	return m
}

func (m *mt19937) next() uint32 {
	if m.idx >= 624 {
		for i := 0; i < 624; i++ {
			y := (m.mt[i] & 0x80000000) | (m.mt[(i+1)%624] & 0x7fffffff)
			m.mt[i] = m.mt[(i+397)%624] ^ (y >> 1)
			if y&1 != 0 {
				m.mt[i] ^= 0x9908b0df
			}
		}
		m.idx = 0
	}
	y := m.mt[m.idx]
	m.idx++
	y ^= (y >> 11)
	y ^= (y << 7) & 0x9d2c5680
	y ^= (y << 15) & 0xefc60000
	y ^= (y >> 18)
	return y
}

func torchCPUUniform(m *mt19937) float32 {
	return float32(m.next()&0x00ffffff) * (1.0 / 16777216.0)
}

// Single-precision trigonometric and log constants matching glibc libm FMA implementation.
var (
	invPiOver2 = math.Float64frombits(0x41645f306dc9c883)
	piOver2    = math.Float64frombits(0x3ff921fb54442d18)

	trigSigns = [4]float64{
		math.Float64frombits(0x3ff0000000000000), //  1.0
		math.Float64frombits(0xbff0000000000000), // -1.0
		math.Float64frombits(0xbff0000000000000), // -1.0
		math.Float64frombits(0x3ff0000000000000), //  1.0
	}

	polyC40 = math.Float64frombits(0xbfc555545995a603)
	polyC50 = math.Float64frombits(0x3f81107605230bc4)
	polyC60 = math.Float64frombits(0xbf2994eb3774cf24)

	polyC38 = math.Float64frombits(0xbfdffffffd0c621c)
	polyC48 = math.Float64frombits(0x3fa55553e1068f19)
	polyC58 = math.Float64frombits(0xbf56c087e89a359d)
	polyC68 = math.Float64frombits(0x3ef99343027bf8c3)

	logTabInvc = [16]float64{
		math.Float64frombits(0x3ff661ec79f8f3be),
		math.Float64frombits(0x3ff571ed4aaf883d),
		math.Float64frombits(0x3ff49539f0f010b0),
		math.Float64frombits(0x3ff3c995b0b80385),
		math.Float64frombits(0x3ff30d190c8864a5),
		math.Float64frombits(0x3ff25e227b0b8ea0),
		math.Float64frombits(0x3ff1bb4a4a1a343f),
		math.Float64frombits(0x3ff12358f08ae5ba),
		math.Float64frombits(0x3ff0953f419900a7),
		math.Float64frombits(0x3ff0000000000000),
		math.Float64frombits(0x3fee608cfd9a47ac),
		math.Float64frombits(0x3feca4b31f026aa0),
		math.Float64frombits(0x3feb2036576afce6),
		math.Float64frombits(0x3fe9c2d163a1aa2d),
		math.Float64frombits(0x3fe886e6037841ed),
		math.Float64frombits(0x3fe767dcf5534862),
	}

	logTabLogc = [16]float64{
		math.Float64frombits(0xbfd57bf7808caade),
		math.Float64frombits(0xbfd2bef0a7c06ddb),
		math.Float64frombits(0xbfd01eae7f513a67),
		math.Float64frombits(0xbfcb31d8a68224e9),
		math.Float64frombits(0xbfc6574f0ac07758),
		math.Float64frombits(0xbfc1aa2bc79c8100),
		math.Float64frombits(0xbfba4e76ce8c0e5e),
		math.Float64frombits(0xbfb1973c5a611ccc),
		math.Float64frombits(0xbfa252f438e10c1e),
		0.0,
		math.Float64frombits(0x3faaa5aa5df25984),
		math.Float64frombits(0x3fbc5e53aa362eb4),
		math.Float64frombits(0x3fc526e57720db08),
		math.Float64frombits(0x3fcbc2860d224770),
		math.Float64frombits(0x3fd1058bc8a07ee1),
		math.Float64frombits(0x3fd4043057b6ee09),
	}

	logLn2 = math.Float64frombits(0x3fe62e42fefa39ef)
	logA0  = math.Float64frombits(0xbfd00ea348b88334)
	logA2  = math.Float64frombits(0x3fd5575b0be00b6a)
	logA1  = math.Float64frombits(0xbfdffffef20a4123)
)

// glibcLogF computes logf(x) matching glibc libm FMA implementation on x86_64.
func glibcLogF(x float32) float32 {
	ix := math.Float32bits(x)
	if ix == 0x3f800000 {
		return 0.0
	}
	tmp := ix - 0x3f330000
	i := (tmp >> 19) & 0xf
	k := int32(tmp) >> 23
	iz := ix - (tmp & 0xff800000)
	z := float64(math.Float32frombits(iz))

	invc := logTabInvc[i]
	logc := logTabLogc[i]

	r := math.FMA(z, invc, -1.0)
	y0 := math.FMA(float64(k), logLn2, logc)
	xmm3 := math.FMA(logA2, r, logA1)
	r2 := r * r
	y := y0 + r
	xmm3 = math.FMA(logA0, r2, xmm3)
	res := math.FMA(xmm3, r2, y)
	return float32(res)
}

// glibcSinF computes sinf(x) with exact bit parity to glibc libm on x86_64.
func glibcSinF(x float32) float32 {
	xd := float64(x)
	negative := math.Float32bits(x)>>31 != 0
	axd := math.Abs(xd)

	ecx := int32(axd * invPiOver2)
	ecx += 0x800000
	ecx >>= 24
	k := float64(ecx)
	r := math.FMA(-k, piOver2, axd)
	r2 := r * r

	var res float64
	if (ecx & 1) == 0 {
		sign := trigSigns[ecx&3]
		xmm0 := math.FMA(polyC60, r2, polyC50)
		xmm1 := sign * r
		xmm3 := xmm1 * r2
		xmm2 := xmm3 * r2
		xmm3 = math.FMA(xmm3, polyC40, xmm1)
		xmm0 = math.FMA(xmm0, xmm2, xmm3)
		res = xmm0
	} else {
		sign := 1.0
		if (ecx & 2) != 0 {
			sign = -1.0
		}
		r4 := r2 * r2
		xmm0 := math.FMA(polyC68, r2, polyC58)
		r6 := r4 * r2
		xmm2 := math.FMA(r2, polyC38, 1.0)
		xmm1 := math.FMA(r4, polyC48, xmm2)
		xmm0 = math.FMA(xmm0, r6, xmm1)
		res = sign * xmm0
	}
	if negative {
		res = -res
	}
	return float32(res)
}

// glibcCosF computes cosf(x) with exact bit parity to glibc libm on x86_64.
func glibcCosF(x float32) float32 {
	xd := float64(x)
	axd := math.Abs(xd)

	ecx := int32(axd * invPiOver2)
	ecx += 0x800000
	ecx >>= 24
	k := float64(ecx)
	r := math.FMA(-k, piOver2, axd)
	r2 := r * r

	var res float64
	if (ecx & 1) == 0 {
		sign := 1.0
		if (ecx & 2) != 0 {
			sign = -1.0
		}
		r4 := r2 * r2
		xmm0 := math.FMA(polyC68, r2, polyC58)
		r6 := r4 * r2
		xmm2 := math.FMA(r2, polyC38, 1.0)
		xmm1 := math.FMA(r4, polyC48, xmm2)
		xmm0 = math.FMA(xmm0, r6, xmm1)
		res = sign * xmm0
	} else {
		sign := trigSigns[ecx&3]
		xmm0 := math.FMA(polyC60, r2, polyC50)
		xmm1 := sign * r
		xmm3 := xmm1 * r2
		xmm2 := xmm3 * r2
		xmm3 = math.FMA(xmm3, polyC40, xmm1)
		xmm0 = math.FMA(xmm0, xmm2, xmm3)
		res = xmm0
	}
	return float32(res)
}

func torchCPUNormalFill16(data []float32) {
	const twoPi = 2.0 * 3.14159265358979323846
	for j := 0; j < 8; j++ {
		u1 := 1.0 - data[j]
		u2 := data[j+8]
		l := glibcLogF(u1)
		radius := float32(math.Sqrt(float64(-2.0 * l)))
		theta := float32(twoPi * float64(u2))
		data[j] = radius * glibcCosF(theta)
		data[j+8] = radius * glibcSinF(theta)
	}
}

// TorchRandn fills out with standard normal float32 values using the exact algorithm
// of PyTorch CPU generator (std::mt19937 low 32 bits, 24-bit uniform, Box-Muller blocks
// of 16 pairing j and j+8, tail redraw).
func TorchRandn(seed uint64, out []float32) {
	n := len(out)
	if n < 16 {
		panic(fmt.Sprintf("yue2: TorchRandn length %d < 16", n))
	}
	engine := newMT19937(uint32(seed & 0xffffffff))
	for i := 0; i < n; i++ {
		out[i] = torchCPUUniform(engine)
	}
	for i := 0; i+16 <= n; i += 16 {
		torchCPUNormalFill16(out[i : i+16])
	}
	if n%16 != 0 {
		tail := out[n-16 : n]
		for i := 0; i < 16; i++ {
			tail[i] = torchCPUUniform(engine)
		}
		torchCPUNormalFill16(tail)
	}
}
