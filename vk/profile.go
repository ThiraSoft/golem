package vk

// Where a submission's time actually goes.
//
// Every other measurement here is wall clock around a whole token, which says
// what a change was worth and never says what to change. A Timeline writes the
// card's own clock into a query pool between the dispatches, so one token
// comes back as a list of spans: the attention, the experts, the six little
// norms nobody suspects.
//
// The clock's period is not read. The driver reports it in
// VkPhysicalDeviceLimits, which is a hundred and forty fields none of the rest
// of this package needs; the spans are scaled so that their sum is the wall
// clock the caller measured around the submission, which is the same answer
// for the only question anyone asks of them.

import (
	"fmt"
	"sort"
	"time"
	"unsafe"
)

// A Timeline is a query pool and the labels of the stamps written into it.
type Timeline struct {
	d      *Device
	pool   uint64
	labels []string
	n      int // how many stamps the recording may write
	at     int // how many it has written so far
}

// NewTimeline makes room for n stamps.
func (d *Device) NewTimeline(n int) (*Timeline, error) {
	t := &Timeline{d: d, n: n}
	qpci := queryPoolCreateInfo{
		sType:      structQueryPoolCreateInfo,
		queryType:  queryTypeTimestamp,
		queryCount: uint32(n),
	}
	if err := check("vkCreateQueryPool", vkCreateQueryPool(d.dev, &qpci, 0, &t.pool)); err != nil {
		return nil, err
	}
	return t, nil
}

func (t *Timeline) Close() {
	if t != nil && t.pool != 0 {
		vkDestroyQueryPool(t.d.dev, t.pool, 0)
		t.pool = 0
	}
}

// Reset clears the pool and the labels. It must be recorded before any stamp.
func (t *Timeline) Reset(r *Recorder) {
	vkCmdResetQueryPool(r.cb, t.pool, 0, uint32(t.n))
	t.labels = t.labels[:0]
	t.at = 0
}

// Stamp records the card's clock at the end of everything submitted so far,
// and names the span that ends here. The first stamp names nothing; it is the
// start.
func (t *Timeline) Stamp(r *Recorder, label string) {
	if t == nil || t.at >= t.n {
		return
	}
	vkCmdWriteTimestamp(r.cb, stageBottomOfPipe, t.pool, uint32(t.at))
	t.labels = append(t.labels, label)
	t.at++
}

// A Span is the time between two stamps, under the label of the second.
type Span struct {
	Label string
	Ticks uint64
}

// Spans reads the pool back. The submission must have finished.
func (t *Timeline) Spans() ([]Span, error) {
	if t.at < 2 {
		return nil, nil
	}
	out := make([]uint64, t.at)
	const wait = 0x1 // VK_QUERY_RESULT_64_BIT
	const blocking = 0x2
	if err := check("vkGetQueryPoolResults", vkGetQueryPoolResults(t.d.dev, t.pool, 0, uint32(t.at),
		uint64(len(out))*8, unsafe.Pointer(&out[0]), 8, wait|blocking)); err != nil {
		return nil, err
	}
	spans := make([]Span, 0, t.at-1)
	for i := 1; i < t.at; i++ {
		spans = append(spans, Span{Label: t.labels[i], Ticks: out[i] - out[i-1]})
	}
	return spans, nil
}

// Report groups the spans by label, scales them so that they add up to total,
// and writes them heaviest first. This is the thing worth reading.
func (t *Timeline) Report(total time.Duration) (string, error) {
	spans, err := t.Spans()
	if err != nil {
		return "", err
	}
	sum := uint64(0)
	byLabel := map[string]uint64{}
	count := map[string]int{}
	for _, s := range spans {
		byLabel[s.Label] += s.Ticks
		count[s.Label]++
		sum += s.Ticks
	}
	if sum == 0 {
		return "", fmt.Errorf("vk: the timeline came back empty")
	}
	labels := make([]string, 0, len(byLabel))
	for l := range byLabel {
		labels = append(labels, l)
	}
	sort.Slice(labels, func(i, j int) bool { return byLabel[labels[i]] > byLabel[labels[j]] })

	out := fmt.Sprintf("%v over %d stamps, %d ticks, %.2f ns a tick\n",
		total, len(spans), sum, float64(total.Nanoseconds())/float64(sum))
	for _, l := range labels {
		share := float64(byLabel[l]) / float64(sum)
		out += fmt.Sprintf("  %-16s %7.3f ms  %5.1f%%  over %d\n",
			l, share*float64(total)/float64(time.Millisecond), share*100, count[l])
	}
	return out, nil
}
