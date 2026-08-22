package gemma

// The audio encoder.
//
// Two of them, behind one type. E2B's projector carries a twelve-block
// conformer; the 12B's carries a single projection and no encoder at all. What
// they share is their answer — one row per soft token, ProjDim wide — and the
// caller does not care which it got.

// AudioTower is a projector's audio half, bound and ready to run.
type AudioTower struct {
	Cfg *AudioConfig
	W   *AudioWeights
}

// NewAudioTower binds a configuration to its weights.
func NewAudioTower(cfg *AudioConfig, w *AudioWeights) *AudioTower {
	return &AudioTower{Cfg: cfg, W: w}
}
