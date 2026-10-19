package flux3

import (
	"os"
	"path/filepath"
)

// comfyDefault is where the machine this was written on keeps ComfyUI, whose
// models directory and tokenizer are what Open reads unless told otherwise.
const comfyDefault = "/mnt/data/dev/ComfyUI"

func comfyDir() string {
	if d := os.Getenv("GOLEM_FLUX3_COMFY"); d != "" {
		return d
	}
	if d := os.Getenv("GOLEM_KREA2_COMFY"); d != "" {
		return d
	}
	return comfyDefault
}

// DiTPath, EncoderPath, VAEPath, and TokenizerDir return the paths to the
// three FLUX 3 model files and the tokenizer in ComfyUI.
func DiTPath() string      { return ComfyUI(comfyDir()).DiT }
func EncoderPath() string  { return ComfyUI(comfyDir()).Encoder }
func VAEPath() string      { return ComfyUI(comfyDir()).VAE }
func TokenizerDir() string { return ComfyUI(comfyDir()).Tokenizer }

// Options says where the files are and what may stay on the card.
type Options struct {
	// The three checkpoints and the tokenizer; empty means ComfyUI's.
	Encoder, DiT, VAE, Tokenizer string
	// Keep leaves the DiT on the card between pictures.
	Keep bool
	// MaxPixels bounds width x height and sizes the VAE; 1024 x 1024 when 0.
	MaxPixels int
	// Precision specifies the weight storage precision in DiT (PrecisionFP8 or PrecisionFP16).
	Precision Precision
}

// ComfyUI returns Options that read the three files and the tokenizer out of
// a ComfyUI directory.
func ComfyUI(dir string) Options {
	return Options{
		DiT:       filepath.Join(dir, "models", "diffusion_models", "flux3-image-klein-bf16.safetensors"),
		Encoder:   filepath.Join(dir, "models", "text_encoders", "qwen3vl_4b_fp8_scaled.safetensors"),
		VAE:       filepath.Join(dir, "models", "vae", "flux2-vae.safetensors"),
		Tokenizer: filepath.Join(dir, "comfy", "text_encoders", "qwen25_tokenizer"),
		Precision: PrecisionFP8,
	}
}
