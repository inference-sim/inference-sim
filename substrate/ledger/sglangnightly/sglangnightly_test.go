package sglangnightly_test

import (
	"archive/zip"
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/inference-sim/inference-sim/substrate/ledger"
	"github.com/inference-sim/inference-sim/substrate/ledger/sglangnightly"
)

func sum(b []byte) string {
	s := sha256.Sum256(b)
	return hex.EncodeToString(s[:])
}

func result(gpu string, partition int, variant, model, bench string) string {
	return fmt.Sprintf(`{"gpu_config":%q,"partition":%d,"model":%q,"variant":%q,"server_args":["--tp","8"],"benchmarks":[%s],"benchmarks_by_io_len":{}}`,
		gpu, partition, model, variant, bench)
}

func bs(throughput string) string {
	return `{"batch_size":1,"output_throughput":` + throughput + `}`
}

// pin builds the artifact a nightly run uploaded and the sources.yaml entry that pins it.
func pin(t *testing.T, run string, attempt int, commit string, results ...string) (ledger.Artifact, []byte) {
	t.Helper()
	name := "consolidated-metrics-" + run
	js := []byte(fmt.Sprintf(`{"run_id":%q,"run_date":"2026-11-01T00:00:00+00:00","commit_sha":%q,"branch":"main","results":[%s]}`,
		run, commit, strings.Join(results, ",")))
	return pinBytes(t, run, attempt, commit, name, js)
}

// pinBytes zips js as the artifact name and pins it.
func pinBytes(t *testing.T, run string, attempt int, commit, name string, js []byte) (ledger.Artifact, []byte) {
	t.Helper()
	var zb bytes.Buffer
	w := zip.NewWriter(&zb)
	f, _ := w.Create(name + ".json")
	_, _ = f.Write(js)
	if err := w.Close(); err != nil {
		t.Fatal(err)
	}
	return ledger.Artifact{ID: fmt.Sprintf("nightly-%s-%d", run, attempt), UsedBy: "test", Revision: ledger.Revision{
		Kind: "workflow_artifact", Ref: run, RunAttempt: attempt, Commit: commit, Artifact: name,
		Assets: []ledger.Asset{{Name: name + ".zip", SHA256: sum(zb.Bytes())}, {Name: name + ".json", SHA256: sum(js)}},
	}}, zb.Bytes()
}

func write(t *testing.T, as []ledger.Artifact, zips [][]byte) (*ledger.Source, string) {
	t.Helper()
	dir := t.TempDir()
	for i, a := range as {
		if err := os.WriteFile(filepath.Join(dir, a.ID+".zip"), zips[i], 0o644); err != nil {
			t.Fatal(err)
		}
	}
	return &ledger.Source{ID: sglangnightly.SourceID, Artifacts: as}, dir
}

var c1 = strings.Repeat("a", 40)

// A unit is a test case of a run; its role key is the commit, test name, model and hardware, so
// two partitions of one case share a role key, a re-run attempt that measured the same is a
// duplicate, and one that measured differently is another measurement of the same unit.
func TestIngestUnitsAndRoleKeys(t *testing.T) {
	a1, z1 := pin(t, "100", 1, c1,
		result("8-gpu-h200", 1, "TP8", "m", bs("1.5")),
		result("8-gpu-h200", 2, "TP8", "m", bs("2.5")),
		result("8-gpu-b200", 1, "TP8", "m", bs("3.5")),
		result("8-gpu-h200", 1, "MTP", "m", ""))
	a2, z2 := pin(t, "100", 2, c1,
		result("8-gpu-h200", 1, "TP8", "m", bs("1.5")),
		result("8-gpu-h200", 2, "TP8", "m", bs("2.75")),
		result("8-gpu-b200", 1, "TP8", "m", bs("3.5")),
		result("8-gpu-h200", 1, "MTP", "m", bs("4.5")))
	src, dir := write(t, []ledger.Artifact{a1, a2}, [][]byte{z1, z2})
	arts, input, err := sglangnightly.ReadArtifacts(src, dir)
	if err != nil {
		t.Fatal(err)
	}
	units, rep, err := sglangnightly.Ingest(arts, c1, input)
	if err != nil {
		t.Fatal(err)
	}
	if rep.Records != 8 || len(rep.LeftOut[sglangnightly.LeftOutNoBenchmarks]) != 1 {
		t.Fatalf("records %d, left out %v; want 8 and one case without a benchmark row", rep.Records, rep.LeftOut)
	}
	if rep.Units != 4 || rep.RoleKeys != 3 {
		t.Errorf("units %d role keys %d, want 4 3: the two H200 partitions share a role key", rep.Units, rep.RoleKeys)
	}
	if rep.Duplicates != 2 || rep.JoinedUnits != 1 {
		t.Errorf("duplicates %d joined units %d, want 2 and 1: partition 2 differs between attempts", rep.Duplicates, rep.JoinedUnits)
	}
	for _, u := range units {
		rk, err := sglangnightly.RoleKey(u.Identity)
		if err != nil {
			t.Fatal(err)
		}
		if _, ok := rk["partition"]; ok {
			t.Errorf("the role key holds the partition: %v", rk)
		}
		if id, _ := ledger.ID(rk); id != u.RoleKey {
			t.Errorf("unit %s has role key %s, want %s", u.UnitID, u.RoleKey, id)
		}
	}
	if rep.UnitsBy["8-gpu-h200"] != 3 || rep.UnitsBy["8-gpu-b200"] != 1 {
		t.Errorf("units by hardware: %v", rep.UnitsBy)
	}
}

// D3: an artifact whose zip, file, run or commit differs from its pin is refused, and so is a
// missing one.
func TestReadArtifactsRefusesWhatIsNotPinned(t *testing.T) {
	a, z := pin(t, "100", 1, c1, result("g", 1, "t", "m", bs("1")))
	other, oz := pin(t, "100", 1, c1, result("g", 1, "t", "m", bs("2")))
	for name, c := range map[string]struct {
		mut  func(a *ledger.Artifact)
		zip  []byte
		want string
	}{
		"another zip":     {func(*ledger.Artifact) {}, oz, "zip's SHA-256"},
		"another file":    {func(a *ledger.Artifact) { a.Revision.Assets[1].SHA256 = other.Revision.Assets[1].SHA256 }, z, "pins"},
		"another run":     {func(a *ledger.Artifact) { a.Revision.Ref = "101" }, z, "pins run 101"},
		"another commit":  {func(a *ledger.Artifact) { a.Revision.Commit = strings.Repeat("b", 40) }, z, "pins run 100"},
		"another file id": {func(a *ledger.Artifact) { a.Revision.Assets[1].Name = "x.json" }, z, "exactly the pinned file"},
	} {
		x := a
		x.Revision.Assets = append([]ledger.Asset(nil), a.Revision.Assets...)
		c.mut(&x)
		src, dir := write(t, []ledger.Artifact{x}, [][]byte{c.zip})
		if _, _, err := sglangnightly.ReadArtifacts(src, dir); err == nil || !strings.Contains(err.Error(), c.want) {
			t.Errorf("%s: %v, want an error containing %q", name, err, c.want)
		}
	}
	var eb bytes.Buffer
	zw := zip.NewWriter(&eb)
	for _, n := range []string{a.Revision.Artifact + ".json", "extra.json"} {
		f, _ := zw.Create(n)
		_, _ = f.Write([]byte("{}"))
	}
	if err := zw.Close(); err != nil {
		t.Fatal(err)
	}
	extra := a
	extra.Revision.Assets = []ledger.Asset{{Name: a.Revision.Artifact + ".zip", SHA256: sum(eb.Bytes())}, a.Revision.Assets[1]}
	src, dir := write(t, []ledger.Artifact{extra}, [][]byte{eb.Bytes()})
	if _, _, err := sglangnightly.ReadArtifacts(src, dir); err == nil || !strings.Contains(err.Error(), "exactly the pinned file") {
		t.Errorf("a zip with a second file: %v", err)
	}
	src, dir = write(t, []ledger.Artifact{a}, [][]byte{z})
	if err := os.Remove(filepath.Join(dir, a.ID+".zip")); err != nil {
		t.Fatal(err)
	}
	if _, _, err := sglangnightly.ReadArtifacts(src, dir); err == nil {
		t.Error("a missing artifact was accepted")
	}
}

func TestRoleKeyRefusesOtherIdentities(t *testing.T) {
	good := map[string]any{"source": sglangnightly.SourceID, "run_id": "1", "commit": "c", "test": "t", "model": "m",
		"hardware": "h", "partition": 1, "server_args_sha256": "s"}
	if _, err := sglangnightly.RoleKey(good); err != nil {
		t.Fatal(err)
	}
	for name, mut := range map[string]func(m map[string]any){
		"extra":  func(m map[string]any) { m["x"] = 1 },
		"lacks":  func(m map[string]any) { delete(m, "test"); m["y"] = 1 },
		"source": func(m map[string]any) { m["source"] = "other" },
	} {
		m := map[string]any{}
		for k, v := range good {
			m[k] = v
		}
		mut(m)
		if _, err := sglangnightly.RoleKey(m); err == nil {
			t.Errorf("%s: accepted", name)
		}
	}
}
