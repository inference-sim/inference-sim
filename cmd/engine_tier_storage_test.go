package cmd

import (
	"testing"

	"github.com/inference-sim/inference-sim/substrate/contracts/engine"
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

// objEngine is an engine that runs an obj tier: it records the buckets it was given.
type objEngine struct {
	engine.Engine
	buckets func(bucket string) tierBucket
}

func (e *objEngine) ShareTierBuckets(f func(bucket string) tierBucket) { e.buckets = f }

// Engines of one model that name one bucket share its objects: a transfer one writes is found by
// the other, and a READ of a set with a missing object ends in error; another model, or another
// bucket, gets its own.
func TestEnginesOfOneModelShareTheObjectsOfOneBucket(t *testing.T) {
	s := newTierStorage()
	a, b, c := &objEngine{}, &objEngine{}, &objEngine{}
	s.share(a, "model-x")
	s.share(b, "model-x")
	s.share(c, "model-y")
	s.share(plainEngine{}, "model-x")

	if !a.buckets("s3").Write([]string{"k1", "k2"}) {
		t.Fatal("a write to an empty bucket ended in error")
	}
	if !b.buckets("s3").Read([]string{"k1", "k2"}) || !b.buckets("s3").Exists("k1") {
		t.Error("an engine of the same model and bucket does not see the objects the first wrote")
	}
	if b.buckets("s3").Read([]string{"k1", "missing"}) {
		t.Error("a READ with a missing object succeeded")
	}
	if b.buckets("other").Exists("k1") || c.buckets("s3").Exists("k1") {
		t.Error("another bucket or another model shares the objects")
	}
}
