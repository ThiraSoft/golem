package main

import (
	"context"
	"encoding/binary"
	"fmt"
	"io"
	"os"
	"os/exec"
	"os/signal"
	"syscall"

	"github.com/ThiraSoft/golem/stt"
)

var recorders = []struct {
	name string
	args []string
}{
	{"pw-record", []string{"--rate", "24000", "--channels", "1", "--format", "s16", "-"}},
	{"arecord", []string{"-q", "-t", "raw", "-f", "S16_LE", "-r", "24000", "-c", "1"}},
	{"ffmpeg", []string{"-hide_banner", "-loglevel", "error", "-f", "alsa", "-i", "default", "-ar", "24000", "-ac", "1", "-f", "s16le", "-"}},
}

func findRecorderIn(look func(string) (string, error)) (string, error) {
	for _, r := range recorders {
		if _, err := look(r.name); err == nil {
			return r.name, nil
		}
	}
	return "", fmt.Errorf("no recorder found: looked for pw-record, arecord and ffmpeg")
}

func findRecorder() (string, error) { return findRecorderIn(exec.LookPath) }

func recorderCmd(ctx context.Context) (*exec.Cmd, error) {
	name, err := findRecorder()
	if err != nil {
		return nil, err
	}
	for _, r := range recorders {
		if r.name == name {
			return exec.CommandContext(ctx, name, r.args...), nil
		}
	}
	return nil, fmt.Errorf("unknown recorder %s", name)
}

func runListen(sttPath string) error {
	opts, err := stt.Locate(sttPath)
	if err != nil {
		return err
	}
	m, err := stt.Open(opts)
	if err != nil {
		return err
	}
	defer m.Close()

	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer cancel()

	cmd, err := recorderCmd(ctx)
	if err != nil {
		return err
	}

	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return err
	}
	if err := cmd.Start(); err != nil {
		return fmt.Errorf("failed to start recorder %s: %w", cmd.Path, err)
	}
	defer func() {
		_ = cmd.Process.Kill()
		_ = cmd.Wait()
	}()

	stream := m.Stream(ctx)
	defer stream.Close()

	done := make(chan struct{})
	go func() {
		defer close(done)
		for seg := range stream.Text() {
			fmt.Print(seg.Text)
			_ = os.Stdout.Sync()
		}
		fmt.Println()
	}()

	buf := make([]byte, 4096)
	samples := make([]float32, len(buf)/2)
	for {
		n, err := io.ReadFull(stdout, buf)
		if n > 0 {
			numSamples := n / 2
			for i := 0; i < numSamples; i++ {
				s := int16(binary.LittleEndian.Uint16(buf[i*2 : (i+1)*2]))
				samples[i] = float32(s) / 32768.0
			}
			stream.Write(samples[:numSamples])
		}
		if err != nil {
			break
		}
	}

	stream.Close()
	<-done
	return nil
}

func runTranscribe(sttPath, filePath string) error {
	opts, err := stt.Locate(sttPath)
	if err != nil {
		return err
	}
	m, err := stt.Open(opts)
	if err != nil {
		return err
	}
	defer m.Close()

	data, err := os.ReadFile(filePath)
	if err != nil {
		return err
	}
	text, err := m.Transcribe(data)
	if err != nil {
		return err
	}
	fmt.Println(text)
	return nil
}
