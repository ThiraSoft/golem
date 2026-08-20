package architecture

// The shared layer must not know that engines exist, and no engine may reach
// into another. This is the rule the whole repository is arranged around, and
// it is cheap enough to check that there is no reason to trust it instead.

import (
	"go/build"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

const module = "github.com/ThiraSoft/golem"

// sharedLayer holds primitives with no opinion about what is being run.
var sharedLayer = []string{"tensors", "nn", "token", "audio", "sample"}

// engines are the self-contained models.
var engines = []string{"pockettts", "gemma"}

func TestImportBoundaries(t *testing.T) {
	root := repoRoot(t)

	err := filepath.Walk(root, func(path string, info os.FileInfo, err error) error {
		if err != nil || !info.IsDir() {
			return err
		}
		name := info.Name()
		if name == ".git" || name == "testdata" || name == "ref" || name == "docs" {
			return filepath.SkipDir
		}

		pkg, err := build.ImportDir(path, 0)
		if err != nil {
			return nil // not a Go package
		}
		relative, err := filepath.Rel(root, path)
		if err != nil {
			return err
		}
		top := strings.Split(filepath.ToSlash(relative), "/")[0]

		imports := append(append([]string{}, pkg.Imports...), pkg.TestImports...)
		for _, imported := range imports {
			if !strings.HasPrefix(imported, module+"/") {
				continue
			}
			importedTop := strings.Split(strings.TrimPrefix(imported, module+"/"), "/")[0]

			if contains(sharedLayer, top) && contains(engines, importedTop) {
				t.Errorf("%s is in the shared layer but imports the %s engine (%s)",
					relative, importedTop, imported)
			}
			if contains(engines, top) && contains(engines, importedTop) && top != importedTop {
				t.Errorf("%s imports %s: the engines must not know each other", relative, imported)
			}
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
}

func contains(list []string, s string) bool {
	for _, item := range list {
		if item == s {
			return true
		}
	}
	return false
}

func repoRoot(t *testing.T) string {
	t.Helper()
	dir, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	for {
		if _, err := os.Stat(filepath.Join(dir, "go.mod")); err == nil {
			return dir
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			t.Fatal("go.mod not found above the current directory")
		}
		dir = parent
	}
}
