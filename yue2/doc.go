// Package yue2 generates songs from style tags and lyrics using YuE2,
// following yue2.cpp's pipeline: an autoregressive transformer writing
// an ABC score and semantic tokens, a non-autoregressive transformer generating
// audio latents with flow matching, and an Oobleck VAE decoding them to 48 kHz stereo.
//
// Each stage is checked against recordings from yue2.cpp; see ref/yue2.
package yue2
