package krea2

import "testing"

// The decode of the recorded run, from the latent ComfyUI handed its VAE to
// the picture. ComfyUI decodes in fp16.
func TestVAEMatchesComfyUI(t *testing.T) {
	f := loadFixtures(t)
	needFile(t, VAEPath())
	shape := f.shape(t, "vae/z")
	h, w := shape[3], shape[4]
	v, err := OpenVAE(device(t), VAEPath(), h*w)
	if err != nil {
		t.Fatal(err)
	}
	defer v.Close()
	rgb, err := v.Decode(f.read(t, "vae/z"), h, w)
	if err != nil {
		t.Fatal(err)
	}
	compareRMS(t, "image", rgb, f.read(t, "sample/image"), 1e-2)
	compare(t, "image", rgb, f.read(t, "sample/image"), 5e-2)
}
