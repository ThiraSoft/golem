package nn

// ResizeBilinearAntialias is ggml's bilinear resize under its antialias flag —
// GGML_SCALE_MODE_BILINEAR | GGML_SCALE_FLAG_ANTIALIAS, which is what
// F.interpolate(..., mode="bilinear", align_corners=False, antialias=True)
// does, and what clip.cpp resizes a learned position table with when the patch
// grid is not the one the table was trained at.
//
// Two details separate it from the bilinear anyone would write, and neither
// shows on a picture:
//
// The filter is a triangle whose support *widens as the picture shrinks* —
// max(1, 1/scale) — so every source that should contribute does. A fixed
// two-tap bilinear shrinking fourfold reads two pixels out of every sixteen
// and calls the rest absent.
//
// And the weights are divided by what was actually gathered rather than by
// their nominal sum. An output near an edge sees fewer sources; normalizing by
// the nominal sum would darken it, which on a position table is a systematic
// drift at the borders of the grid.
//
// src and the result are channel-major: value (c, y, x) at c*w*h + y*w + x.
func ResizeBilinearAntialias(src []float32, srcW, srcH, dstW, dstH, channels int) []float32 {
	// Half a pixel, which is align_corners=False: a destination pixel's centre
	// sits half a step in, not on the corner.
	const offset float32 = 0.5

	// Every coordinate here is float32, because every one of them is float32
	// in ggml — the scale factors, the support, the centres and the weights.
	// Computing them wider is not more correct, it is a different answer: on a
	// ratio like 23/48 the two disagree in the sixth decimal, which is enough
	// to shift a weight and land a millionth off the thing being matched.
	sfW := float32(dstW) / float32(srcW)
	sfH := float32(dstH) / float32(srcH)
	supportW := float32(1)
	if v := 1 / sfW; v > supportW {
		supportW = v
	}
	supportH := float32(1)
	if v := 1 / sfH; v > supportH {
		supportH = v
	}
	invW, invH := 1/supportW, 1/supportH

	out := make([]float32, channels*dstW*dstH)
	for c := 0; c < channels; c++ {
		plane := src[c*srcW*srcH : (c+1)*srcW*srcH]
		dstPlane := out[c*dstW*dstH : (c+1)*dstW*dstH]
		for j := 0; j < dstH; j++ {
			y := (float32(j) + offset) / sfH
			yMin := max(int(y-supportH+offset), 0)
			yMax := min(int(y+supportH+offset), srcH)
			for i := 0; i < dstW; i++ {
				x := (float32(i) + offset) / sfW
				xMin := max(int(x-supportW+offset), 0)
				xMax := min(int(x+supportW+offset), srcW)

				var val, total float32
				for sy := yMin; sy < yMax; sy++ {
					wy := triangleFilter((float32(sy) - y + offset) * invH)
					if wy <= 0 {
						continue
					}
					for sx := xMin; sx < xMax; sx++ {
						wx := triangleFilter((float32(sx) - x + offset) * invW)
						w := wx * wy
						if w <= 0 {
							continue
						}
						val += plane[sy*srcW+sx] * w
						total += w
					}
				}
				if total > 0 {
					val /= total
				}
				dstPlane[j*dstW+i] = val
			}
		}
	}
	return out
}

// triangleFilter is the tent ggml weights its samples with, in the width ggml
// evaluates it at.
func triangleFilter(x float32) float32 {
	if x < 0 {
		x = -x
	}
	if 1-x < 0 {
		return 0
	}
	return 1 - x
}
