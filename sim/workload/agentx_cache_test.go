package workload

import (
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"sync"
	"testing"
)

const agxSlicePath = "testdata/agentx/weka_v7_slice.jsonl"

// TestAgentXCorpusCacheRoundTrip: a miss parses and writes a snapshot; a hit reads back a
// corpus identical to a fresh parse, and a replay of it sends exactly the same requests
// (INV-6 with and without the cache).
func TestAgentXCorpusCacheRoundTrip(t *testing.T) {
	fresh, err := LoadAgentXCorpus(agxSlicePath, "", 0, true)
	if err != nil {
		t.Fatal(err)
	}
	dir := t.TempDir()
	miss, r1, err := LoadAgentXCorpusCached(agxSlicePath, "", 0, true, dir, "build-a")
	if err != nil || r1.Hit || r1.Ignored != "" || r1.WriteErr != nil {
		t.Fatalf("first load: hit=%v ignored=%q write=%v err=%v", r1.Hit, r1.Ignored, r1.WriteErr, err)
	}
	hit, r2, err := LoadAgentXCorpusCached(agxSlicePath, "", 0, true, dir, "build-a")
	if err != nil || !r2.Hit || r2.Path != r1.Path {
		t.Fatalf("second load: hit=%v ignored=%q err=%v", r2.Hit, r2.Ignored, err)
	}
	if !reflect.DeepEqual(fresh, miss) || !reflect.DeepEqual(fresh, hit) {
		t.Fatal("cached corpus differs from a fresh parse")
	}
	cfg := AgentXReplayConfig{Concurrency: 4, Seed: 42, DurationUs: 900e6, GraceUs: 30e6, IdleCapUs: 10e6, MinRatio: 0, MaxRatio: 1}
	a := runAgentX(t, fresh, cfg, agxConformanceService)
	b := runAgentX(t, hit, cfg, agxConformanceService)
	if !reflect.DeepEqual(issueLog(a), issueLog(b)) || len(a.issued) == 0 {
		t.Fatal("a replay of the cached corpus differs from a replay of the parsed one")
	}
}

// TestAgentXCorpusCacheKeyAndDamage: a snapshot written under another filter, split
// setting or build is not used, and a damaged, truncated or foreign file is ignored with a
// reason, re-parsed, and replaced by a good snapshot.
func TestAgentXCorpusCacheKeyAndDamage(t *testing.T) {
	dir := t.TempDir()
	_, r, err := LoadAgentXCorpusCached(agxSlicePath, "", 0, true, dir, "build-a")
	if err != nil {
		t.Fatal(err)
	}
	for _, other := range []struct {
		limit int
		split bool
		build string
	}{{60000, true, "build-a"}, {0, false, "build-a"}, {0, true, "build-b"}} {
		_, ro, err := LoadAgentXCorpusCached(agxSlicePath, "", other.limit, other.split, dir, other.build)
		if err != nil || ro.Hit || ro.Path == r.Path {
			t.Errorf("%+v reused the snapshot of another key (hit=%v path=%s)", other, ro.Hit, ro.Path)
		}
	}
	good, err := os.ReadFile(r.Path)
	if err != nil {
		t.Fatal(err)
	}
	fresh, _ := LoadAgentXCorpus(agxSlicePath, "", 0, true)
	flip := append([]byte(nil), good...)
	flip[len(flip)/2] ^= 0xff
	wrongKey := []byte(strings.Replace(string(good), "build=build-a", "build=build-z", 1))
	for name, bad := range map[string][]byte{
		"flipped byte":  flip,
		"truncated":     good[:len(good)/3],
		"empty":         {},
		"foreign":       []byte("not a snapshot at all"),
		"key rewritten": wrongKey,
	} {
		if err := os.WriteFile(r.Path, bad, 0o644); err != nil {
			t.Fatal(err)
		}
		c, rb, err := LoadAgentXCorpusCached(agxSlicePath, "", 0, true, dir, "build-a")
		if err != nil || rb.Hit || rb.Ignored == "" {
			t.Errorf("%s: hit=%v ignored=%q err=%v", name, rb.Hit, rb.Ignored, err)
		}
		if !reflect.DeepEqual(c, fresh) {
			t.Errorf("%s: fallback corpus differs from a fresh parse", name)
		}
		if b, _ := os.ReadFile(r.Path); string(b) != string(good) {
			t.Errorf("%s: the damaged snapshot was not replaced", name)
		}
	}
	if _, _, err := LoadAgentXCorpusCached(agxSlicePath, strings.Repeat("0", 64), 0, true, dir, "build-a"); err == nil {
		t.Error("a cached load accepted a wrong corpus digest")
	}
}

// TestAgentXCorpusCacheConcurrentProcesses: many loaders racing on one cache directory all
// get the same corpus, and no temporary file is left behind.
func TestAgentXCorpusCacheConcurrentProcesses(t *testing.T) {
	dir := t.TempDir()
	fresh, _ := LoadAgentXCorpus(agxSlicePath, "", 0, true)
	var wg sync.WaitGroup
	errs := make(chan string, 16)
	for i := 0; i < 16; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			c, r, err := LoadAgentXCorpusCached(agxSlicePath, "", 0, true, dir, "build-a")
			switch {
			case err != nil:
				errs <- err.Error()
			case r.Ignored != "":
				errs <- "saw a partial snapshot: " + r.Ignored
			case !reflect.DeepEqual(c, fresh):
				errs <- "corpus differs"
			}
		}()
	}
	wg.Wait()
	close(errs)
	for e := range errs {
		t.Error(e)
	}
	left, _ := filepath.Glob(filepath.Join(dir, ".agentx-corpus-*.tmp"))
	all, _ := filepath.Glob(filepath.Join(dir, "*"))
	if len(left) != 0 || len(all) != 1 {
		t.Errorf("cache dir holds %v", all)
	}
}
