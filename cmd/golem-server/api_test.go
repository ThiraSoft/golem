package main

import (
	"encoding/base64"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestDecodePlainStringContentStillWorks(t *testing.T) {
	body := `{"model":"m","messages":[{"role":"user","content":"hello"}]}`
	var req completionRequest
	if err := json.Unmarshal([]byte(body), &req); err != nil {
		t.Fatal(err)
	}
	if len(req.Messages) != 1 || req.Messages[0].Content != "hello" {
		t.Fatalf("content came out %q", req.Messages[0].Content)
	}
	if req.Messages[0].Images != nil {
		t.Fatal("a plain string brought a picture")
	}
}

func TestDecodeContentParts(t *testing.T) {
	png := base64.StdEncoding.EncodeToString([]byte("\x89PNG pretend"))
	body := `{"model":"m","messages":[{"role":"user","content":[
		{"type":"text","text":"what is this"},
		{"type":"image_url","image_url":{"url":"data:image/png;base64,` + png + `"}},
		{"type":"text","text":"be brief"}
	]}]}`
	var req completionRequest
	if err := json.Unmarshal([]byte(body), &req); err != nil {
		t.Fatal(err)
	}
	m := req.Messages[0]
	if m.Content != "what is this\nbe brief" {
		t.Fatalf("text came out %q", m.Content)
	}
	if len(m.Images) != 1 || string(m.Images[0]) != "\x89PNG pretend" {
		t.Fatalf("%d images decoded: %q", len(m.Images), m.Images)
	}
}

func TestDecodeContentPartRefusesTheNetwork(t *testing.T) {
	body := `{"messages":[{"role":"user","content":[
		{"type":"image_url","image_url":{"url":"https://example.com/cat.png"}}]}]}`
	var req completionRequest
	err := json.Unmarshal([]byte(body), &req)
	if err == nil {
		t.Fatal("an http URL was accepted")
	}
	if !strings.Contains(err.Error(), "does not fetch") {
		t.Fatalf("the refusal reads %q", err)
	}
}

func TestDecodeContentPartReadsAFile(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "picture.png")
	if err := os.WriteFile(path, []byte("bytes"), 0o600); err != nil {
		t.Fatal(err)
	}
	body := `{"messages":[{"role":"user","content":[
		{"type":"image_url","image_url":{"url":"` + path + `"}}]}]}`
	var req completionRequest
	if err := json.Unmarshal([]byte(body), &req); err != nil {
		t.Fatal(err)
	}
	if len(req.Messages[0].Images) != 1 || string(req.Messages[0].Images[0]) != "bytes" {
		t.Fatalf("the file came out %q", req.Messages[0].Images)
	}
}

func TestDecodeContentPartRefusesAnUnknownType(t *testing.T) {
	body := `{"messages":[{"role":"user","content":[{"type":"video_url"}]}]}`
	var req completionRequest
	if err := json.Unmarshal([]byte(body), &req); err == nil {
		t.Fatal("a video part was accepted")
	}
}
