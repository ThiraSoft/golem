package gemma

// Several conversations kept at once, over one set of weights.
//
// A slot is a cache and nothing else. The weights are mapped once and the
// scratch is one request's worth, because the server still drives one request
// at a time; what having several caches buys is that two clients stop evicting
// each other's prefix, which is the whole reason the cache exists.
//
// The context is split rather than multiplied, as llama.cpp's -parallel does:
// the memory a server was told it may use is what it uses, and each of the n
// conversations gets its share of the positions.

import "fmt"

// SetSlots rebuilds the cache as n independent ones, each holding
// MaxContext/n positions. It replaces whatever was cached, so it belongs at
// startup, before a conversation exists.
func (m *Model) SetSlots(n int) error {
	if n < 1 {
		return fmt.Errorf("gemma: %d slots", n)
	}
	per := m.Cfg.MaxContext / n
	if per < 1 {
		return fmt.Errorf("gemma: a context of %d cut into %d slots leaves no position each", m.Cfg.MaxContext, n)
	}
	cfg := *m.Cfg
	cfg.MaxContext = per
	m.caches = make([]*Cache, n)
	for i := range m.caches {
		m.caches[i] = NewCache(&cfg)
	}
	m.slot = 0
	m.cache = m.caches[0]
	m.slotContext = per
	return nil
}

// Slots is how many conversations the model can hold at once.
func (m *Model) Slots() int {
	if len(m.caches) == 0 {
		return 1
	}
	return len(m.caches)
}

// SlotContext is how many positions one of them holds.
func (m *Model) SlotContext() int {
	if m.slotContext == 0 {
		return m.Cfg.MaxContext
	}
	return m.slotContext
}

// Slot is one of the caches, for a caller building a batch out of several
// conversations at once.
func (m *Model) Slot(i int) *Cache {
	if len(m.caches) == 0 {
		return m.cache
	}
	return m.caches[i]
}

// UseSlot points the next forward pass at one of them. Out of range is a
// caller's mistake rather than a request's, so it panics.
func (m *Model) UseSlot(i int) {
	if len(m.caches) == 0 {
		if i != 0 {
			panic(fmt.Sprintf("gemma: slot %d of a model with one cache", i))
		}
		return
	}
	if i < 0 || i >= len(m.caches) {
		panic(fmt.Sprintf("gemma: slot %d of %d", i, len(m.caches)))
	}
	m.slot = i
	m.cache = m.caches[i]
}
