package main

// What the tower made of a picture, kept between requests.
//
// /v1/chat/completions is stateless, so a conversation about one picture sends
// that picture again with every turn. Looking at it again each time would make
// the tenth turn cost what the first did, and looking is not cheap: it is the
// one part of a multimodal prompt that a cached prefix cannot make free.
//
// So the rows are kept under the hash of the bytes they came from, and a
// bounded number of them: a server that remembered every picture it was ever
// sent would grow without a limit anybody chose.

import (
	"crypto/sha256"
	"sync"
)

type imageCache struct {
	mu    sync.Mutex
	rows  map[[32]byte][][]float32
	order [][32]byte
	limit int
}

func newImageCache(limit int) *imageCache {
	return &imageCache{rows: map[[32]byte][][]float32{}, limit: limit}
}

// Encode returns what the tower makes of these bytes, looking only if it has
// not looked at them before.
func (c *imageCache) Encode(data []byte, encode func([]byte) ([][]float32, error)) ([][]float32, error) {
	key := sha256.Sum256(data)

	c.mu.Lock()
	rows, ok := c.rows[key]
	c.mu.Unlock()
	if ok {
		return rows, nil
	}

	// Outside the lock: a second request for the same picture while the first
	// is still looking will look again, which costs a second and is simpler
	// than holding the model's queue behind a map.
	rows, err := encode(data)
	if err != nil {
		return nil, err
	}

	c.mu.Lock()
	defer c.mu.Unlock()
	if _, ok := c.rows[key]; !ok {
		c.rows[key] = rows
		c.order = append(c.order, key)
		for len(c.order) > c.limit {
			delete(c.rows, c.order[0])
			c.order = c.order[1:]
		}
	}
	return rows, nil
}
