package nn

import "unsafe"

// GemmPanelsF32 computes c[j*cStride+r] = Σ_{k<n} a[k*aStride+r]·x[j*xStride+k]
// for r < rows and j < cols, on the caller's thread, rows a multiple of
// sixteen: sixteen outputs of a are contiguous for each k, and each column of
// x is contiguous along k. It is the shape of both halves of an attention
// once the keys are transposed: scores with the keys along a, then the
// answer with the values' dimensions along a.
//
// It is nn/gemm_f16.go's tile in float32, with its sums stored rather than
// added. Without the kernel it takes one sum at a time.
func GemmPanelsF32(a []float32, aStride int, x []float32, xStride, n, rows, cols int, c []float32, cStride int) {
	if rows%16 != 0 {
		panic("nn: GemmPanelsF32 takes rows sixteen at a time")
	}
	if !avx2 || n == 0 {
		for j := 0; j < cols; j++ {
			for r := 0; r < rows; r++ {
				var s float32
				for k := 0; k < n; k++ {
					s += a[k*aStride+r] * x[j*xStride+k]
				}
				c[j*cStride+r] = s
			}
		}
		return
	}
	var edgeC [6 * 16]float32
	var edgeX []float32
	for j := 0; j < cols; j += 6 {
		m := min(6, cols-j)
		xs, xst := unsafe.Pointer(&x[j*xStride]), xStride*4
		if m < 6 {
			// The last columns are copied into a tile of six, zeros after
			// them.
			edgeX = make([]float32, 6*n)
			for q := 0; q < m; q++ {
				copy(edgeX[q*n:(q+1)*n], x[(j+q)*xStride:(j+q)*xStride+n])
			}
			xs, xst = unsafe.Pointer(&edgeX[0]), n*4
		}
		for r := 0; r < rows; r += 16 {
			if m == 6 {
				gemmF32x16x6AVX2(&a[r], aStride*4, (*float32)(xs), xst, n, &c[j*cStride+r], cStride*4)
				continue
			}
			gemmF32x16x6AVX2(&a[r], aStride*4, (*float32)(xs), xst, n, &edgeC[0], 64)
			for q := 0; q < m; q++ {
				copy(c[(j+q)*cStride+r:(j+q)*cStride+r+16], edgeC[q*16:(q+1)*16])
			}
		}
	}
}
