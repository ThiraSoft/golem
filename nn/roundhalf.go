package nn

// RoundHalfRange rounds every value of v to the nearest fp16 and back, in
// place. It is RoundHalf over a slice, done eight at a time where the machine
// allows it.
func RoundHalfRange(v []float32) {
	if len(v) == 0 {
		return
	}
	if fastRoundHalf(&v[0], len(v)) {
		return
	}
	for i, x := range v {
		v[i] = RoundHalf(x)
	}
}
