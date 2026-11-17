package cmd

import (
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/engine"
)

// tierEngine is an engine that runs a secondary tier: it records the storage it was given.
type tierEngine struct {
	engine.Engine
	files func(rootDir string) tierFiles
}

func (e *tierEngine) ShareTierStorage(f func(rootDir string) tierFiles) { e.files = f }

// plainEngine runs no tier and takes no storage.
type plainEngine struct{ engine.Engine }

// Engines of one model that name one root_dir get one file system, so a block file one publishes is
// found by the other; another model, or another root_dir, gets its own; an engine with no tier is
// left alone.
func TestEnginesOfOneModelShareTheFileSystemUnderOneRootDir(t *testing.T) {
	s := newTierStorage()
	a, b, c := &tierEngine{}, &tierEngine{}, &tierEngine{}
	s.share(a, "model-x")
	s.share(b, "model-x")
	s.share(c, "model-y")
	s.share(plainEngine{}, "model-x")

	a.files("/kv").Publish("/kv/h1")
	if !b.files("/kv").Exists("/kv/h1") {
		t.Error("an engine of the same model and root_dir does not see the block file the first published")
	}
	if b.files("/other").Exists("/kv/h1") || b.files("/other").Exists("/other/h1") {
		t.Error("another root_dir shares the file system")
	}
	if c.files("/kv").Exists("/kv/h1") {
		t.Error("another model shares the file system under one root_dir")
	}
}
