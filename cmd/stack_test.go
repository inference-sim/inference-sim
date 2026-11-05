package cmd

import (
	"os"
	"path/filepath"
	"reflect"
	"regexp"
	"testing"

	"github.com/inference-sim/inference-sim/internal/stack"
)

// stack.json lists the engine releases this binary's build site links, one per kind (LIF-6).
func TestStackJSONNamesTheEnginesThisBinaryRuns(t *testing.T) {
	s, err := stack.Read(filepath.Join("..", "stack.json"))
	if err != nil {
		t.Fatal(err)
	}
	if got, want := s.Engines, EngineReleases(); !reflect.DeepEqual(got, want) {
		t.Errorf("stack.json engines %+v; the build site runs %+v (go run ./internal/stack/cmd/stackgen)", got, want)
	}
}

// The llm-d guides the router tests read are extracted at the commit guides.lock pins, which
// stack.json records (SIM-3, SIM-5).
func TestTheGuideFixturesAreAtTheLockedCommit(t *testing.T) {
	commit, err := stack.ReadGuidesLock(filepath.Join("..", "guides.lock"))
	if err != nil {
		t.Fatal(err)
	}
	files, err := filepath.Glob(filepath.Join("..", "sim", "llmd", "testdata", "llm-d-guides", "*.yaml"))
	if err != nil || len(files) == 0 {
		t.Fatalf("no guide fixtures: %v", err)
	}
	from := regexp.MustCompile(`(?m)^# EXTRACTED VERBATIM from llm-d/llm-d@([0-9a-f]{40}):`)
	for _, f := range files {
		b, err := os.ReadFile(f)
		if err != nil {
			t.Fatal(err)
		}
		m := from.FindSubmatch(b)
		switch {
		case m == nil:
			t.Errorf("%s names no llm-d/llm-d commit it was extracted at", f)
		case string(m[1]) != commit:
			t.Errorf("%s was extracted at %s; guides.lock pins %s", f, m[1], commit)
		}
	}
}
