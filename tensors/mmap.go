package tensors

// The whole file is mapped rather than read. A GGUF of a language model is
// several gigabytes, and reading it would trade a twelve-millisecond startup
// for several seconds and a second copy in memory. The kernels work on the
// mapped bytes directly.

import (
	"fmt"
	"os"
	"syscall"
)

type mapping struct {
	data []byte
	file *os.File
}

func mapFile(path string) (*mapping, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	info, err := f.Stat()
	if err != nil {
		f.Close()
		return nil, err
	}
	if info.Size() == 0 {
		f.Close()
		return nil, fmt.Errorf("%s is empty", path)
	}
	data, err := syscall.Mmap(int(f.Fd()), 0, int(info.Size()),
		syscall.PROT_READ, syscall.MAP_SHARED)
	if err != nil {
		f.Close()
		return nil, fmt.Errorf("mmap %s: %w", path, err)
	}
	return &mapping{data: data, file: f}, nil
}

func (m *mapping) Close() error {
	if m == nil || m.data == nil {
		return nil
	}
	err := syscall.Munmap(m.data)
	m.data = nil
	if cerr := m.file.Close(); err == nil {
		err = cerr
	}
	return err
}
