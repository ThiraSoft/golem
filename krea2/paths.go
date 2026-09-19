package krea2

import (
	"os"
	"path/filepath"
)

// comfyDefault is where the machine this was written on keeps ComfyUI, whose
// models directory and tokenizer are what Open reads unless told otherwise.
const comfyDefault = "/mnt/data/dev/ComfyUI"

func comfyDir() string {
	if d := os.Getenv("GOLEM_KREA2_COMFY"); d != "" {
		return d
	}
	return comfyDefault
}

// The three files the mobile workflow loads and the tokenizer, from the
// ComfyUI of GOLEM_KREA2_COMFY or this machine's.
func DiTPath() string      { return ComfyUI(comfyDir()).DiT }
func EncoderPath() string  { return ComfyUI(comfyDir()).Encoder }
func VAEPath() string      { return ComfyUI(comfyDir()).VAE }
func TokenizerDir() string { return ComfyUI(comfyDir()).Tokenizer }

// ComfyUI is the Options that read the three files and the tokenizer out of
// a ComfyUI directory. The tokenizer is ComfyUI's copy of Qwen2's, which is
// the one its Krea 2 text encoder is loaded with.
func ComfyUI(dir string) Options {
	return Options{
		DiT:       filepath.Join(dir, "models", "unet", "krea2_turbo_fp8.safetensors"),
		Encoder:   filepath.Join(dir, "models", "text_encoders", "qwen3vl_4b_fp8_scaled.safetensors"),
		VAE:       filepath.Join(dir, "models", "vae", "qwen_image_vae.safetensors"),
		Tokenizer: filepath.Join(dir, "comfy", "text_encoders", "qwen25_tokenizer"),
	}
}
