package main

import (
	"fmt"
	"testing"
)

func TestImageCacheLooksOnce(t *testing.T) {
	c := newImageCache(2)
	looks := 0
	encode := func(data []byte) ([][]float32, error) {
		looks++
		return [][]float32{{float32(len(data))}}, nil
	}
	for i := 0; i < 3; i++ {
		rows, err := c.Encode([]byte("a picture"), encode)
		if err != nil {
			t.Fatal(err)
		}
		if len(rows) != 1 || rows[0][0] != 9 {
			t.Fatalf("the rows came back %v", rows)
		}
	}
	if looks != 1 {
		t.Fatalf("the same picture was looked at %d times", looks)
	}
}

func TestImageCacheForgetsTheOldest(t *testing.T) {
	c := newImageCache(2)
	looks := 0
	encode := func([]byte) ([][]float32, error) {
		looks++
		return [][]float32{{1}}, nil
	}
	for i := 0; i < 3; i++ {
		if _, err := c.Encode([]byte(fmt.Sprint(i)), encode); err != nil {
			t.Fatal(err)
		}
	}
	// The first is gone, the last two are not.
	if _, err := c.Encode([]byte("0"), encode); err != nil {
		t.Fatal(err)
	}
	if looks != 4 {
		t.Fatalf("%d looks, expected the first picture to have been forgotten", looks)
	}
	if _, err := c.Encode([]byte("2"), encode); err != nil {
		t.Fatal(err)
	}
	if looks != 4 {
		t.Fatalf("%d looks, expected the newest to still be held", looks)
	}
}

func TestImageCacheRefusesWhatItCannotRead(t *testing.T) {
	c := newImageCache(2)
	if _, err := c.Encode([]byte("x"), func([]byte) ([][]float32, error) {
		return nil, fmt.Errorf("not a picture")
	}); err == nil {
		t.Fatal("a picture that would not decode was cached")
	}
}
