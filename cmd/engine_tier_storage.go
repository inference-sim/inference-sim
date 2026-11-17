package cmd

import (
	"github.com/inference-sim/blis-substrate/contracts/engine"
)

// Shared tier storage (#1698): vLLM instances that name one root_dir for a secondary tier (the fs
// tier) read and write one file system, so a block one instance wrote is a hit for another that
// serves the same model. An engine that runs such a tier takes its files through ShareTierStorage;
// the run gives every engine the same store for a (model, root_dir) pair and a different one for a
// different pair, since the block files of two models are not interchangeable.

// tierFiles is the shape of a tier's block-file store as the engine declares it
// (blis-vllm core/vllm TierFiles), written as the interface literal.
type tierFiles = interface {
	Exists(path string) bool
	Publish(path string)
}

// memoryFiles is a block-file store held in memory: the simulated file system under one root_dir.
type memoryFiles map[string]bool

func (m memoryFiles) Exists(path string) bool { return m[path] }
func (m memoryFiles) Publish(path string)     { m[path] = true }

// tierStorage is a run's shared file systems, one per (model, root_dir).
type tierStorage struct {
	stores map[[2]string]memoryFiles
}

func newTierStorage() *tierStorage { return &tierStorage{stores: map[[2]string]memoryFiles{}} }

// of is the storage func an engine serving model takes: each root_dir it names is the run's one
// store for the pair.
func (s *tierStorage) of(model string) func(rootDir string) tierFiles {
	return func(rootDir string) tierFiles {
		k := [2]string{model, rootDir}
		if s.stores[k] == nil {
			s.stores[k] = memoryFiles{}
		}
		return s.stores[k]
	}
}

// share hands eng the run's storage for model if eng runs a tier that takes files; an engine that
// does not is left alone.
func (s *tierStorage) share(eng engine.Engine, model string) {
	if sh, ok := eng.(interface {
		ShareTierStorage(func(rootDir string) tierFiles)
	}); ok {
		sh.ShareTierStorage(s.of(model))
	}
}
