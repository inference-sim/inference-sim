package ledger_test

import (
	"encoding/json"
	"os"
	"os/exec"
	"strings"
	"testing"

	"github.com/inference-sim/blis-substrate/ledger"
	"github.com/inference-sim/blis-substrate/ledger/inferencex"
)

// The committed ledger loads and passes every check that needs no source data: the layout,
// D3's pinned revisions, the cut-points, each unit's unit_id and role key, one unit per
// measurement, and the legacy table against the unit list.
func TestCommittedLedger(t *testing.T) {
	l, err := ledger.Load("../catalog/ledger", map[string]ledger.RoleKeyFunc{
		inferencex.SourceID: inferencex.RoleKey,
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(l.Units[inferencex.SourceID]) == 0 {
		t.Fatal("the InferenceX unit list is empty (X22)")
	}
	if l.Legacy == nil || len(l.Legacy.Units) == 0 {
		t.Fatal("legacy-map.yaml is missing or empty")
	}
}

// No tracked file outside the published lists and the legacy table pairs a unit with a role
// (D4).
func TestNoCommittedRoles(t *testing.T) {
	out, err := exec.Command("git", "-C", "..", "ls-files").Output()
	if err != nil {
		t.Fatalf("git ls-files: %v", err)
	}
	var files []string
	for _, f := range strings.Split(strings.TrimSpace(string(out)), "\n") {
		for _, ext := range []string{".yaml", ".yml", ".json", ".jsonl", ".jsonl.gz", ".csv", ".tsv"} {
			if strings.HasSuffix(f, ext) && !strings.HasPrefix(f, ".github/") {
				files = append(files, f)
				break
			}
		}
	}
	if len(files) == 0 {
		t.Fatal("no files to scan (X22)")
	}
	if err := ledger.CheckNoRoles("..", files, nil); err != nil {
		t.Fatal(err)
	}
}

// D3: the AISimulate end-to-end artifact the extractor reads is pinned in
// the ledger (source aisimulate, artifact e2e-accuracy-web), and the pin file the extractor
// checks its input against says the same thing. The ledger entry decides; a pin file that
// differs would let the extractor accept an artifact the ledger does not name.
func TestExtractorPinEqualsTheLedgerEntry(t *testing.T) {
	src, err := ledger.LoadSources("../catalog/ledger")
	if err != nil {
		t.Fatal(err)
	}
	s := src.Lookup("aisimulate")
	if s == nil || len(s.Artifacts) != 1 || s.Artifacts[0].ID != "e2e-accuracy-web" {
		t.Fatalf("source aisimulate does not pin the artifact e2e-accuracy-web: %+v", s)
	}
	rev := s.Artifacts[0].Revision
	raw, err := os.ReadFile("../kernel/scripts/aisimulate_e2e.pin.json")
	if err != nil {
		t.Fatal(err)
	}
	var pin struct {
		Source        string `json:"source"`
		ArtifactID    string `json:"artifact_id"`
		RunID         string `json:"run_id"`
		RunAttempt    int    `json:"run_attempt"`
		Commit        string `json:"commit"`
		Artifact      string `json:"artifact"`
		ZipSHA256     string `json:"zip_sha256"`
		SummarySHA256 string `json:"summary_sha256"`
	}
	if err := json.Unmarshal(raw, &pin); err != nil {
		t.Fatal(err)
	}
	sums := map[string]string{}
	for _, a := range rev.Assets {
		sums[a.Name] = a.SHA256
	}
	for name, pair := range map[string][2]any{
		"source":         {pin.Source, s.ID},
		"artifact id":    {pin.ArtifactID, s.Artifacts[0].ID},
		"run id":         {pin.RunID, rev.Ref},
		"run attempt":    {pin.RunAttempt, rev.RunAttempt},
		"commit":         {pin.Commit, rev.Commit},
		"artifact":       {pin.Artifact, rev.Artifact},
		"zip sha256":     {pin.ZipSHA256, sums[rev.Artifact+".zip"]},
		"summary sha256": {pin.SummarySHA256, sums["summary.json"]},
	} {
		if pair[0] != pair[1] {
			t.Errorf("%s: the extractor's pin file has %v, sources.yaml has %v", name, pair[0], pair[1])
		}
	}
}
