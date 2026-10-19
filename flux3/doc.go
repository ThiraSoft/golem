// Package flux3 draws a picture from a sentence with FLUX 3 Image Klein, the way
// ComfyUI draws it from the same three files: the Qwen3-VL-4B text encoder,
// the 4B single-stream DiT in bf16, and the Flux2 VAE, sampled over the
// flow schedule.
//
// Each stage is transcribed from ComfyUI's Python and checked against it
// waypoint by waypoint; see ref/flux3.
package flux3
