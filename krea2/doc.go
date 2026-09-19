// Package krea2 draws a picture from a sentence with Krea 2, the way ComfyUI
// draws it from the same three files: the Qwen3-VL-4B text encoder, the 12B
// single-stream DiT in fp8, and the Wan 2.1 VAE, sampled with ER-SDE over the
// flow schedule. The same prompt and seed give the same picture, because the
// noise ComfyUI draws is drawn here bit for bit; see rng.go.
//
// Each stage is transcribed from ComfyUI's Python and checked against it
// waypoint by waypoint; see ref/krea2.
package krea2
