// Original to blis-vllm.

package blisvllm

import (
	"bytes"
	"os"
	"strings"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/spec/fields"
)

// fields.json is the field list VLM-3 asks for: every configuration field and environment
// variable vLLM v0.32.0 reads, each with a status, its evidence and an estimation rule. The
// names come from vLLM's source (fields/gen_fields.py, checked against the pinned commit in
// CI); this test holds the file to the document type blis-substrate defines for it.
//
//   - It loads through the substrate's strict loader (fields.Load), which refuses unknown
//     keys, a status without its evidence, an issue that is not <owner>/<repo>#<n>, an
//     estimation rule without its setting, and an unsorted or repeated entry.
//   - It is the engine and release this build models (Upstream).
//   - It is in canonical form: the substrate's Marshal writes the same bytes.
//   - Every trace a modelled field cites is a file in this repository.
func TestTheFieldListLoadsAndNamesThisRelease(t *testing.T) {
	l, err := fields.Load("fields.json")
	if err != nil {
		t.Fatal(err)
	}
	if l.Engine.Kind != Kind || l.Engine.Version != Upstream.Version {
		t.Errorf("fields.json is for %s %s; this build models %s %s", l.Engine.Kind, l.Engine.Version, Kind, Upstream.Version)
	}
	raw, err := os.ReadFile("fields.json")
	if err != nil {
		t.Fatal(err)
	}
	canonical, err := l.Marshal()
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(raw, canonical) {
		t.Error("fields.json is not in the substrate's canonical form; regenerate it with fields/gen_fields.py")
	}
	// A floor, so a generator that emits almost nothing cannot pass (X22).
	const floor = 970
	if len(l.Fields) < floor {
		t.Errorf("fields.json lists %d fields, want at least %d: vLLM v0.32.0 reads more than that", len(l.Fields), floor)
	}
	for _, f := range l.Fields {
		for _, tr := range f.Evidence.Traces {
			if strings.HasPrefix(tr, "/") || strings.Contains(tr, "..") {
				t.Errorf("%s: trace %q is not a path inside this repository", f.Name, tr)
				continue
			}
			if _, err := os.Stat(tr); err != nil {
				t.Errorf("%s: trace %q: %v", f.Name, tr, err)
			}
		}
	}
}

// FieldListJSON is the committed file, byte for byte: what a simulator prints is what CI checks.
func TestFieldListJSONIsTheCommittedFile(t *testing.T) {
	raw, err := os.ReadFile("fields.json")
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(FieldListJSON, raw) {
		t.Error("FieldListJSON differs from fields.json")
	}
}
