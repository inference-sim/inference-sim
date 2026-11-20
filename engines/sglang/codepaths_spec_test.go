// Original to blis-sglang.

package blissglang

import (
	"encoding/json"
	"os"
	"sort"
	"testing"

	"github.com/inference-sim/inference-sim/substrate/contracts/spec/codepaths"
)

// checkSpecMatchesTable fails unless the spec at specPath names, for each path, exactly the
// functions the table at tablePath lists, and the same release.
func checkSpecMatchesTable(t *testing.T, specPath, tablePath string) {
	t.Helper()
	raw, err := os.ReadFile(specPath)
	if err != nil {
		t.Fatal(err)
	}
	var spec struct {
		Engine codepaths.Engine `json:"engine"`
		Paths  map[string][]struct {
			ID string `json:"id"`
		} `json:"paths"`
	}
	if err := json.Unmarshal(raw, &spec); err != nil {
		t.Fatal(err)
	}
	tab, err := codepaths.Load(tablePath)
	if err != nil {
		t.Fatal(err)
	}
	if spec.Engine != tab.Engine {
		t.Errorf("the spec is for %+v, the table for %+v", spec.Engine, tab.Engine)
	}
	for _, p := range tab.Paths {
		var want, got []string
		for _, f := range spec.Paths[p.Path] {
			want = append(want, f.ID)
		}
		for _, f := range p.Functions {
			got = append(got, f.ID)
		}
		sort.Strings(want)
		if len(want) != len(got) {
			t.Errorf("path %s: the spec names %d functions, the table lists %d", p.Path, len(want), len(got))
			continue
		}
		for i := range want {
			if want[i] != got[i] {
				t.Errorf("path %s: the spec names %s where the table lists %s", p.Path, want[i], got[i])
			}
		}
	}
	if len(spec.Paths) != len(tab.Paths) {
		t.Errorf("the spec names %d paths, the table lists %d", len(spec.Paths), len(tab.Paths))
	}
}
