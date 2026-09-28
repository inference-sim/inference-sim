package cmd

import (
	"go/ast"
	"go/parser"
	"go/token"
	"io/fs"
	"os"
	"path/filepath"
	"reflect"
	"regexp"
	"strconv"
	"strings"
	"testing"
)

// R2H3 follow-up (#1838): the catalog's `networks/*.yaml` reusable fabric classes carry NO
// PDTransferBaseLatencyMs. blis-catalog#12 removed it — a fabric class has no inherent
// per-transfer base latency to state (the nominal figure was always 0, so the field only ever
// carried a placeholder), and because the fabric schema is CLOSED the catalog CI gate now
// rejects the key outright as an unknown field.
//
// Nothing is broken today: no code path reads PDTransferBaseLatencyMs from a catalog file. It is
// a cluster.DeploymentConfig field fed solely by --pd-transfer-base-latency (default 0.05 ms,
// consumed in sim/cluster/pd_events.go), and the `networks/` fabric reader has not landed here —
// cmd/catalog_load.go's namespace list still says so. The point of #1838, and of this file, is
// that the reader MUST be authored against a field-free fabric class *when* it lands, because
// the catalog↔loader compatibility is pinned by CATALOG_REVISION: the moment that pin advances
// past blis-catalog#12, a reader expecting the field reads a key the catalog no longer has (and
// a fixture still carrying it fails the catalog's own gate).
//
// The three rules #1838 states for that reader:
//
//	1. read the fabric's InterNodeBwGBps (nominal) as the PD-transfer bandwidth — there is no
//	   separate PD bandwidth figure (R2H2, blis-catalog#10);
//	2. do NOT read or require PDTransferBaseLatencyMs from the fabric class;
//	3. keep the PD-transfer base latency sourced from --pd-transfer-base-latency, whose eventual
//	   owner is blis-registry (blis-registry#10, `method: assumed`). Effective base latency = that
//	   value ALONE — there is no catalog 0 to compose with.
//
// Four contracts encode them:
//
//	BC-1 no production Go source lets a config file supply the field (static guard);
//	BC-2 no committed catalog file declares it, and a `networks/` fixture states its bandwidth
//	     as InterNodeBwGBps (rule 1);
//	BC-3 the number stays a CLI/registry input, identically on run and replay (rule 3, INV-13);
//	BC-4 a tripwire on the loader's namespace report, so the reader cannot land without this
//	     file's rules being read (rule 2).
//
// Rule 1 is deliberately NOT implemented here: PD-transfer bandwidth comes from
// --pd-transfer-bandwidth (default 25 GB/s) today, and sourcing it from a fabric class would
// change values, which R2 forbids (#1817 is value-preserving). It is recorded as a rule, and
// BC-2 asserts it the moment a `networks/` fixture exists.

// pdTransferBaseLatencyField is the Go field / config key at issue. Compared ASCII-folded
// throughout, so a case variant cannot slip past the guards.
const pdTransferBaseLatencyField = "PDTransferBaseLatencyMs"

// pdTransferBaseLatencyOwner is the ONE file allowed to declare a struct field with that name:
// cluster.DeploymentConfig's CLI-sourced field. Relative to the repository root, so the guard
// fails if the declaration moves to (or is copied into) a fabric-class struct.
const pdTransferBaseLatencyOwner = "sim/cluster/deployment.go"

// ---------------------------------------------------------------------------
// BC-1: no production Go source lets a config file supply the field
// ---------------------------------------------------------------------------

// TestNetworksFabric_NoConfigKeyBindsPDTransferBaseLatency is BC-1. It is a STATIC guard rather
// than a behavioral one because the behavior it protects does not exist yet: there is no fabric
// reader to feed a bad input to. What can be checked is that the repository contains no way for
// a catalog-authored file to supply the number —
//
//   - no struct field anywhere binds a YAML or JSON key whose folded name is the field (a fabric
//     struct could name its Go field anything and still decode the retired key via a tag), and
//   - no struct field is DECLARED with that name outside pdTransferBaseLatencyOwner, which is
//     where a fabric class would most naturally grow one.
//
// The CLI wiring (cmd/root.go, cmd/replay.go) and the consumer (sim/cluster/pd_events.go) are
// untouched by both checks: they reference the field, they do not declare it, and
// DeploymentConfig gives it no yaml/json tag.
func TestNetworksFabric_NoConfigKeyBindsPDTransferBaseLatency(t *testing.T) {
	folded := strings.ToLower(pdTransferBaseLatencyField)
	scanned, sawKnownTag := 0, false

	for _, rel := range productionGoSources(t) {
		scanned++
		fset := token.NewFileSet()
		parsed, err := parser.ParseFile(fset, filepath.Join("..", rel), nil, parser.SkipObjectResolution)
		if err != nil {
			t.Fatalf("parse %s: %v", rel, err)
		}
		ast.Inspect(parsed, func(n ast.Node) bool {
			st, ok := n.(*ast.StructType)
			if !ok || st.Fields == nil {
				return true
			}
			for _, field := range st.Fields.List {
				line := fset.Position(field.Pos()).Line
				for _, key := range configTagNames(t, rel, field) {
					// Non-vacuity anchor: a real, unrelated fabric key proves the tag
					// extraction below actually sees tag names.
					if key == "InterNodeBwGBps" {
						sawKnownTag = true
					}
					if strings.ToLower(key) == folded {
						t.Errorf("%s:%d: a config key %q is bound to a struct field — the catalog's "+
							"networks/ fabric classes carry no %s (blis-catalog#12 removed it, and the "+
							"closed fabric schema now rejects it), so no file may supply it; the "+
							"PD-transfer base latency comes from --pd-transfer-base-latency and is owned "+
							"by blis-registry#10 (#1838)",
							rel, line, key, pdTransferBaseLatencyField)
					}
				}
				if rel == pdTransferBaseLatencyOwner {
					continue
				}
				for _, name := range field.Names {
					if name.Name == pdTransferBaseLatencyField {
						t.Errorf("%s:%d: %s is declared outside %s — the only %s is "+
							"cluster.DeploymentConfig's CLI-sourced field; a fabric class must not "+
							"carry one (blis-catalog#12 removed it from networks/*.yaml, #1838)",
							rel, line, pdTransferBaseLatencyField, pdTransferBaseLatencyOwner,
							pdTransferBaseLatencyField)
					}
				}
			}
			return true
		})
	}

	if scanned == 0 {
		t.Fatal("non-vacuity: no production Go sources were scanned")
	}
	if !sawKnownTag {
		t.Errorf("non-vacuity: the scan of %d file(s) saw no InterNodeBwGBps config tag, so the "+
			"tag extraction is not reading tag names and the banned-key check proves nothing", scanned)
	}
}

// productionGoSources returns every non-test .go file in the repository, as paths relative to
// the repository root. testdata/ is excluded (fixtures, covered by BC-2); .git, the agent-local
// .worktrees/, and third-party trees are excluded because they are not this commit's sources.
func productionGoSources(t *testing.T) []string {
	t.Helper()
	var files []string
	err := filepath.WalkDir("..", func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		rel, relErr := filepath.Rel("..", path)
		if relErr != nil {
			return relErr
		}
		if d.IsDir() {
			switch d.Name() {
			case ".git", ".worktrees", "testdata", "vendor", "node_modules":
				return fs.SkipDir
			}
			return nil
		}
		if strings.HasSuffix(rel, ".go") && !strings.HasSuffix(rel, "_test.go") {
			files = append(files, filepath.ToSlash(rel))
		}
		return nil
	})
	if err != nil {
		t.Fatalf("walk repository for Go sources: %v", err)
	}
	return files
}

// configTagNames returns the yaml and json key names a struct field binds, if any. A `-` name
// (explicitly not part of the surface) is skipped, as is a tag with no name (`yaml:",inline"`).
func configTagNames(t *testing.T, rel string, field *ast.Field) []string {
	t.Helper()
	if field.Tag == nil {
		return nil
	}
	raw, err := strconv.Unquote(field.Tag.Value)
	if err != nil {
		t.Fatalf("%s: unquote struct tag %s: %v", rel, field.Tag.Value, err)
	}
	var names []string
	for _, key := range []string{"yaml", "json"} {
		value, ok := reflect.StructTag(raw).Lookup(key)
		if !ok {
			continue
		}
		name := strings.Split(value, ",")[0]
		if name == "" || name == "-" {
			continue
		}
		names = append(names, name)
	}
	return names
}

// ---------------------------------------------------------------------------
// BC-2: no committed catalog file declares the field
// ---------------------------------------------------------------------------

// pdTransferBaseLatencyKeyRe matches a DECLARATION of the retired key in a catalog-authored
// YAML or JSON file — line start, the key (optionally quoted, any letter case), a colon. Prose
// in a `#` comment that names the key on purpose is therefore not matched, the same way
// TestDefaultsBlockRemoved_StaticGuard matches declarations rather than mentions.
var pdTransferBaseLatencyKeyRe = regexp.MustCompile(`(?mi)^[\t ]*"?PDTransferBaseLatencyMs"?[\t ]*:`)

// TestNetworksFabric_FixtureCatalogDeclaresNoPDTransferBaseLatency is BC-2, over real committed
// data. Both halves are forward-looking on purpose: the fixture catalog has no networks/
// namespace today (the reader that would read one has not landed), so the second half is dormant
// until a fabric fixture is added — and it is that addition, at the same moment as the reader,
// that this contract exists to catch. The first half is live now over every committed catalog
// file, and the scan count is asserted so it cannot pass by walking nothing.
func TestNetworksFabric_FixtureCatalogDeclaresNoPDTransferBaseLatency(t *testing.T) {
	scanned := 0
	err := filepath.WalkDir(fixtureCatalog, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() {
			return nil
		}
		switch strings.ToLower(filepath.Ext(path)) {
		case ".yaml", ".yml", ".json":
		default:
			return nil
		}
		data, readErr := os.ReadFile(path)
		if readErr != nil {
			return readErr
		}
		scanned++
		if pdTransferBaseLatencyKeyRe.Match(data) {
			t.Errorf("%s declares %s: the catalog does not carry that key (blis-catalog#12 removed "+
				"it from the closed networks/ fabric schema, which now rejects it), and the PD-transfer "+
				"base latency comes from --pd-transfer-base-latency / blis-registry#10 (#1838)",
				path, pdTransferBaseLatencyField)
		}
		return nil
	})
	if err != nil {
		t.Fatalf("walk fixture catalog %s: %v", fixtureCatalog, err)
	}
	if scanned == 0 {
		t.Fatalf("non-vacuity: no catalog files were scanned under %s", fixtureCatalog)
	}

	// Rule 1: a fabric class states its bandwidth as InterNodeBwGBps (nominal), which is the
	// PD-transfer bandwidth figure — there is no separate PD one (R2H2, blis-catalog#10).
	networksDir := filepath.Join(fixtureCatalog, "networks")
	entries, err := os.ReadDir(networksDir)
	if os.IsNotExist(err) {
		return // dormant: no fabric fixture yet, exactly as cmd/catalog_load.go's namespace list says.
	}
	if err != nil {
		t.Fatalf("read %s: %v", networksDir, err)
	}
	for _, e := range entries {
		if e.IsDir() || !strings.HasSuffix(e.Name(), catalogYAMLExt) {
			continue
		}
		path := filepath.Join(networksDir, e.Name())
		data, readErr := os.ReadFile(path)
		if readErr != nil {
			t.Fatalf("read %s: %v", path, readErr)
		}
		if !strings.Contains(string(data), "InterNodeBwGBps") {
			t.Errorf("%s declares no InterNodeBwGBps: a fabric class states its nominal inter-node "+
				"bandwidth there, and that IS the PD-transfer bandwidth — there is no separate PD "+
				"figure (#1838, blis-catalog#10)", path)
		}
	}
}

// ---------------------------------------------------------------------------
// BC-3: the number stays a CLI/registry input
// ---------------------------------------------------------------------------

// TestNetworksFabric_PDTransferBaseLatencyStaysACLIInput is BC-3 and rule 3: the effective
// PD-transfer base latency is the --pd-transfer-base-latency value ALONE. Its 0.05 ms default is
// a modeling estimate owned by blis-registry (blis-registry#10, `method: assumed`), not a
// datasheet fact a fabric class could state — so there is no catalog 0 to compose with, and the
// flag must keep supplying it on BOTH commands with the same default (INV-13: a run that cannot
// be replayed with identical flags is not reproducible).
func TestNetworksFabric_PDTransferBaseLatencyStaysACLIInput(t *testing.T) {
	const flagName = "pd-transfer-base-latency"
	const wantDefault = "0.05"

	runFlag := runCmd.Flags().Lookup(flagName)
	if runFlag == nil {
		t.Fatalf("run must register --%s: it is the sole source of the PD-transfer base latency "+
			"(no catalog fabric class states one, #1838)", flagName)
	}
	replayFlag := replayCmd.Flags().Lookup(flagName)
	if replayFlag == nil {
		t.Fatalf("replay must register --%s, or a run using it cannot be replayed (INV-13)", flagName)
	}
	if runFlag.DefValue != wantDefault {
		t.Errorf("--%s default on run = %q, want %q (the blis-registry#10 placeholder, #1838)",
			flagName, runFlag.DefValue, wantDefault)
	}
	if replayFlag.DefValue != runFlag.DefValue {
		t.Errorf("--%s default differs between run (%q) and replay (%q): INV-13 requires identical "+
			"flags to reproduce a run", flagName, runFlag.DefValue, replayFlag.DefValue)
	}
}

// ---------------------------------------------------------------------------
// BC-4: the reader cannot land without rule 2 being read
// ---------------------------------------------------------------------------

// TestNetworksFabric_ReaderHasNotLandedYet is BC-4, a deliberate TRIPWIRE rather than a
// regression test. #1838 is a coordination note whose whole purpose is that the `networks/`
// fabric reader be authored field-free; BC-1 and BC-2 catch the wrong implementation, but only a
// tripwire on the loader's own namespace report guarantees the rules are in front of whoever
// writes it.
//
// It fails exactly once, when a networks/fabric namespace count joins catalogLoadReport — that
// is, when the reader lands. The right response is not to delete this test: keep BC-1/BC-2/BC-3
// (which then guard real code), point the new reader at InterNodeBwGBps for bandwidth, and drop
// this function with the reader's own PR.
func TestNetworksFabric_ReaderHasNotLandedYet(t *testing.T) {
	typ := reflect.TypeOf(catalogLoadReport{})
	if typ.NumField() == 0 {
		t.Fatal("non-vacuity: catalogLoadReport has no fields to inspect")
	}
	for i := 0; i < typ.NumField(); i++ {
		folded := strings.ToLower(typ.Field(i).Name)
		if !strings.Contains(folded, "network") && !strings.Contains(folded, "fabric") {
			continue
		}
		t.Errorf("catalogLoadReport.%s means the networks/ fabric reader has landed — re-read #1838 "+
			"before relying on it: (1) the fabric's InterNodeBwGBps (nominal) is the PD-transfer "+
			"bandwidth, there is no separate PD figure; (2) the fabric class carries NO %s "+
			"(blis-catalog#12 removed it and the closed schema rejects it); (3) the PD-transfer base "+
			"latency stays sourced from --pd-transfer-base-latency, owned by blis-registry#10 — the "+
			"effective value is that number alone. Then delete this tripwire in the reader's PR.",
			typ.Field(i).Name, pdTransferBaseLatencyField)
	}
}
