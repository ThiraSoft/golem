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

// The three files the mobile workflow loads, under ComfyUI's models directory.
func DiTPath() string {
	return filepath.Join(comfyDir(), "models", "unet", "krea2_turbo_fp8.safetensors")
}
func EncoderPath() string {
	return filepath.Join(comfyDir(), "models", "text_encoders", "qwen3vl_4b_fp8_scaled.safetensors")
}
func VAEPath() string {
	return filepath.Join(comfyDir(), "models", "vae", "qwen_image_vae.safetensors")
}

// TokenizerDir is ComfyUI's copy of the Qwen2 tokenizer, which is the one its
// Krea 2 text encoder is loaded with.
func TokenizerDir() string {
	return filepath.Join(comfyDir(), "comfy", "text_encoders", "qwen25_tokenizer")
}
