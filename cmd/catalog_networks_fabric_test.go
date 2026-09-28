package cmd

import (
	"bytes"
	"errors"
	"go/ast"
	"go/parser"
	"go/token"
	"io"
	"io/fs"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"strconv"
	"strings"
	"testing"

	"github.com/spf13/cobra"
	"gopkg.in/yaml.v3"
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
//	BC-2 no committed catalog file DECLARES it — established by parsing every catalog file and
//	     walking its keys, not by pattern-matching lines — and a `networks/` fixture states its
//	     bandwidth as InterNodeBwGBps with a usable positive number (rule 1);
//	BC-3 the number stays a CLI/registry input, identically on run and replay: both register the
//	     flag with the same default, the flag writes the variable, and that variable is what feeds
//	     DeploymentConfig.PDTransferBaseLatencyMs on both paths (rule 3, INV-13);
//	BC-4 a tripwire on the loader's namespace report, so no new catalog namespace reader can land
//	     without this file's rules being read (rule 2).
//
// Rule 1 is deliberately NOT implemented here: PD-transfer bandwidth comes from
// --pd-transfer-bandwidth (default 25 GB/s) today, and sourcing it from a fabric class would
// change values, which R2 forbids (#1817 is value-preserving). It is recorded as a rule, and
// BC-2 asserts it the moment a `networks/` fixture exists. What no guard in this file can do is
// prove that the future reader *assigns* the fabric's bandwidth to
// DeploymentConfig.PDTransferBandwidthGBps — that behavior has no code to observe until the
// reader exists, which is why BC-4 puts the rule in front of its author instead.

// pdTransferBaseLatencyField is the Go field / config key at issue. Compared ASCII-folded
// throughout, so a case variant cannot slip past the guards.
const pdTransferBaseLatencyField = "PDTransferBaseLatencyMs"

// pdTransferBaseLatencyOwner is the ONE file allowed to declare a struct field with that name:
// cluster.DeploymentConfig's CLI-sourced field. Relative to the repository root, so the guard
// fails if the declaration moves to (or is copied into) a fabric-class struct.
const pdTransferBaseLatencyOwner = "sim/cluster/deployment.go"

// pdTransferBaseLatencyFlagVar is the package variable --pd-transfer-base-latency writes
// (cmd/root.go), and therefore the ONLY expression DeploymentConfig.PDTransferBaseLatencyMs may
// be fed from on either command. BC-3 checks both halves of that path.
const pdTransferBaseLatencyFlagVar = "pdTransferBaseLatency"

// fabricBandwidthKey is the fabric class's nominal inter-node bandwidth, which IS the PD-transfer
// bandwidth figure — there is no separate PD one (rule 1, R2H2, blis-catalog#10).
const fabricBandwidthKey = "InterNodeBwGBps"

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
					if key == fabricBandwidthKey {
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
		t.Errorf("non-vacuity: the scan of %d file(s) saw no %s config tag, so the "+
			"tag extraction is not reading tag names and the banned-key check proves nothing",
			scanned, fabricBandwidthKey)
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

// catalogKeyDecl is one mapping key DECLARED somewhere in a catalog-authored file: the key as
// written, the line it sits on, and the scalar it binds (isScalar is false when the value is a
// nested mapping or a sequence).
type catalogKeyDecl struct {
	name     string
	line     int
	scalar   string
	isScalar bool
}

// catalogKeyDecls returns every mapping key declared anywhere in a YAML or JSON document stream:
// at any nesting depth, in block or flow style, quoted or bare, across every document of a
// multi-document file.
//
// It PARSES rather than pattern-matches, and that is the whole point. The anchored regex this
// replaced (`^[\t ]*"?Key"?[\t ]*:`) saw only keys at the start of a line, so a flow mapping
// (`{Key: 0}`), a nested flow element, or a single-quoted key (`'Key':`) declared the retired
// field without tripping the guard. Parsing also keeps the exclusions the regex bought by
// accident but now holds by construction: a `#` comment contributes no keys, and a key named
// inside a string VALUE is not a declaration.
func catalogKeyDecls(t *testing.T, label string, data []byte) []catalogKeyDecl {
	t.Helper()
	var decls []catalogKeyDecl
	dec := yaml.NewDecoder(bytes.NewReader(data))
	for {
		var doc yaml.Node
		err := dec.Decode(&doc)
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			// A catalog file that does not parse is a repository integrity problem, not a
			// reason to skip the guard: JSON is YAML, so every committed catalog file must
			// decode here or this contract is checking a subset of the catalog.
			t.Fatalf("%s: parse as YAML/JSON: %v", label, err)
		}
		collectCatalogKeyDecls(&doc, &decls)
	}
	return decls
}

// collectCatalogKeyDecls appends every key of every mapping reachable from n, depth-first.
func collectCatalogKeyDecls(n *yaml.Node, out *[]catalogKeyDecl) {
	if n.Kind == yaml.MappingNode {
		// Mapping content is a flat key, value, key, value, … list.
		for i := 0; i+1 < len(n.Content); i += 2 {
			key, value := n.Content[i], n.Content[i+1]
			*out = append(*out, catalogKeyDecl{
				name:     key.Value,
				line:     key.Line,
				scalar:   value.Value,
				isScalar: value.Kind == yaml.ScalarNode,
			})
			collectCatalogKeyDecls(value, out)
		}
		return
	}
	for _, child := range n.Content {
		collectCatalogKeyDecls(child, out)
	}
}

// catalogDeclsOf returns the declarations of one key (folded comparison) in a catalog file.
func catalogDeclsOf(t *testing.T, label, key string, data []byte) []catalogKeyDecl {
	t.Helper()
	var hits []catalogKeyDecl
	for _, decl := range catalogKeyDecls(t, label, data) {
		if strings.EqualFold(decl.name, key) {
			hits = append(hits, decl)
		}
	}
	return hits
}

// TestNetworksFabric_FixtureCatalogDeclaresNoPDTransferBaseLatency is BC-2, over real committed
// data, in three parts:
//
//   - the detector sees a declaration in every form a catalog file could write one, and does not
//     mistake a prose mention for one — the coverage half, and the reason the check parses;
//   - no committed catalog file declares the retired key (live now, over every catalog file);
//   - a `networks/` fabric fixture declares its bandwidth as InterNodeBwGBps with a usable
//     positive number (rule 1). That part stays dormant until such a fixture exists — the reader
//     that would read one has not landed — and it is that addition, at the same moment as the
//     reader, that this contract exists to catch.
func TestNetworksFabric_FixtureCatalogDeclaresNoPDTransferBaseLatency(t *testing.T) {
	// Coverage: every declaration form a catalog author could legally write must be detected,
	// and a mention that is not a declaration must not be.
	for _, tc := range []struct {
		name    string
		doc     string
		declare bool
	}{
		{"block style", "PDTransferBaseLatencyMs: 0\n", true},
		{"double-quoted key", "\"PDTransferBaseLatencyMs\": 0\n", true},
		{"single-quoted key", "'PDTransferBaseLatencyMs': 0\n", true},
		{"flow mapping", "{PDTransferBaseLatencyMs: 0}\n", true},
		{"nested in a flow sequence", "fabrics: [{Name: ib-400g, PDTransferBaseLatencyMs: 0}]\n", true},
		{"nested block mapping", "ib-400g:\n  PDTransferBaseLatencyMs: 0\n", true},
		{"case variant", "pdtransferbaselatencyms: 0\n", true},
		{"json object", "{\"PDTransferBaseLatencyMs\": 0}\n", true},
		{"second document of a stream", "InterNodeBwGBps: 400\n---\nPDTransferBaseLatencyMs: 0\n", true},
		{"comment mention", "# PDTransferBaseLatencyMs was removed by blis-catalog#12\nInterNodeBwGBps: 400\n", false},
		{"string value mention", "note: PDTransferBaseLatencyMs is gone\n", false},
	} {
		got := len(catalogDeclsOf(t, tc.name, pdTransferBaseLatencyField, []byte(tc.doc))) > 0
		if got != tc.declare {
			t.Errorf("detector on %s: declares %s = %v, want %v — a form the guard misses is a form "+
				"a catalog file could smuggle the retired key in (#1838)",
				tc.name, pdTransferBaseLatencyField, got, tc.declare)
		}
	}

	// Live: no committed catalog file declares the key. keys counts declarations across the
	// whole catalog so the walk cannot pass by parsing nothing.
	scanned, keys := 0, 0
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
		decls := catalogKeyDecls(t, path, data)
		keys += len(decls)
		for _, decl := range decls {
			if !strings.EqualFold(decl.name, pdTransferBaseLatencyField) {
				continue
			}
			t.Errorf("%s:%d declares %s: the catalog does not carry that key (blis-catalog#12 removed "+
				"it from the closed networks/ fabric schema, which now rejects it), and the PD-transfer "+
				"base latency comes from --pd-transfer-base-latency / blis-registry#10 (#1838)",
				path, decl.line, pdTransferBaseLatencyField)
		}
		return nil
	})
	if err != nil {
		t.Fatalf("walk fixture catalog %s: %v", fixtureCatalog, err)
	}
	if scanned == 0 {
		t.Fatalf("non-vacuity: no catalog files were scanned under %s", fixtureCatalog)
	}
	if keys == 0 {
		t.Fatalf("non-vacuity: %d catalog file(s) under %s yielded no key declarations at all, so the "+
			"parse is not reading keys and the banned-key check proves nothing", scanned, fixtureCatalog)
	}

	// Coverage: the rule-1 usability predicate, which is dormant until a `networks/` fixture
	// exists and so would otherwise ship unexercised. Every value the reader cannot divide by
	// must be rejected — NaN included, since it passes an ordering test and an IsInf test both.
	for _, tc := range []struct {
		scalar string
		usable bool
	}{
		{"400", true},
		{"400.5", true},
		{"4e2", true},
		{"0.5", true},
		{"", false},
		{"0", false},
		{"-400", false},
		{"fast", false},
		{"NaN", false},
		{"nan", false},
		{"Inf", false},
		{"-Inf", false},
	} {
		if got := fabricBandwidthUsable(tc.scalar); got != tc.usable {
			t.Errorf("fabricBandwidthUsable(%q) = %v, want %v — an unusable bandwidth the guard accepts "+
				"is a fixture the reader cannot take a PD-transfer bandwidth from (#1838)",
				tc.scalar, got, tc.usable)
		}
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
		assertFabricDeclaresUsableBandwidth(t, path, data)
	}
}

// assertFabricDeclaresUsableBandwidth checks rule 1's catalog-side half: the fabric class must
// DECLARE InterNodeBwGBps as a key bound to a positive finite number, because that number is the
// PD-transfer bandwidth. A key that is present but empty, non-numeric, zero, or negative is no
// more usable to the reader than an absent one, so a mere "the name appears in the file" check
// would let an unusable fixture through.
func assertFabricDeclaresUsableBandwidth(t *testing.T, path string, data []byte) {
	t.Helper()
	decls := catalogDeclsOf(t, path, fabricBandwidthKey, data)
	if len(decls) == 0 {
		t.Errorf("%s declares no %s: a fabric class states its nominal inter-node bandwidth there, "+
			"and that IS the PD-transfer bandwidth — there is no separate PD figure (#1838, "+
			"blis-catalog#10)", path, fabricBandwidthKey)
		return
	}
	for _, decl := range decls {
		if !decl.isScalar {
			t.Errorf("%s:%d: %s binds a mapping or sequence, want a single positive number — it is "+
				"the fabric's nominal inter-node bandwidth in GB/s (#1838)", path, decl.line, fabricBandwidthKey)
			continue
		}
		if !fabricBandwidthUsable(decl.scalar) {
			t.Errorf("%s:%d: %s = %q, want a positive finite number: the reader takes the PD-transfer "+
				"bandwidth from this figure, so a zero/empty/non-numeric one leaves it with nothing to "+
				"read (#1838, blis-catalog#10)", path, decl.line, fabricBandwidthKey, decl.scalar)
		}
	}
}

// fabricBandwidthUsable reports whether a scalar bound to InterNodeBwGBps is a figure the
// PD-transfer path could divide a payload by: a real, strictly positive, finite number.
//
// NaN is the case worth stating outright, because it slips through the obvious phrasing of this
// check: ParseFloat accepts "NaN", and every comparison against NaN is false, so a *rejecting*
// form written as `value <= 0 || math.IsInf(value, 0)` lets it pass (IsInf is false for NaN too).
// A NaN bandwidth is worse than an absent key — it reaches sim/cluster/pd_events.go and turns
// every PD-transfer duration NaN silently. The *requiring* form below excludes NaN through
// `value > 0` on its own; the explicit IsNaN keeps that true if the condition is ever inverted
// back, and the coverage table asserts the outcome either way.
func fabricBandwidthUsable(scalar string) bool {
	value, err := strconv.ParseFloat(scalar, 64)
	if err != nil {
		return false
	}
	return !math.IsNaN(value) && !math.IsInf(value, 0) && value > 0
}

// ---------------------------------------------------------------------------
// BC-3: the number stays a CLI/registry input
// ---------------------------------------------------------------------------

// TestNetworksFabric_PDTransferBaseLatencyStaysACLIInput is BC-3 and rule 3: the effective
// PD-transfer base latency is the --pd-transfer-base-latency value ALONE. Its 0.05 ms default is
// a modeling estimate owned by blis-registry (blis-registry#10, `method: assumed`), not a
// datasheet fact a fabric class could state — so there is no catalog 0 to compose with.
//
// "Stays a CLI input" is the whole path, not just a registered flag, so all three links are
// checked on BOTH commands (INV-13: a run that cannot be replayed with identical flags is not
// reproducible): the flag exists with the same default, setting it writes
// pdTransferBaseLatencyFlagVar, and that variable is what the DeploymentConfig the command builds
// takes the field from. Checking only registration would pass while the assignment was deleted or
// fed from something else, which is exactly how the number would stop being a CLI input.
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

	// Link 2: the operator's value lands in the variable. Restored afterwards so the shared
	// package state this suite reads elsewhere is left as found.
	restore := pdTransferBaseLatency
	t.Cleanup(func() { pdTransferBaseLatency = restore })
	for _, tc := range []struct {
		command *cobra.Command
		label   string
		set     string
		want    float64
	}{
		{runCmd, "run", "0.17", 0.17},
		{replayCmd, "replay", "0.31", 0.31},
	} {
		flag := tc.command.Flags().Lookup(flagName)
		if err := flag.Value.Set(tc.set); err != nil {
			t.Fatalf("%s --%s=%s: %v", tc.label, flagName, tc.set, err)
		}
		if pdTransferBaseLatency != tc.want {
			t.Errorf("%s --%s=%s left %s = %v, want %v: the flag must write the variable the "+
				"DeploymentConfig is built from, or the CLI is not the source of the number (#1838)",
				tc.label, flagName, tc.set, pdTransferBaseLatencyFlagVar, pdTransferBaseLatency, tc.want)
		}
	}

	// Link 3: that variable, and nothing else, feeds the deployment field on both paths.
	for _, rel := range []string{"cmd/root.go", "cmd/replay.go"} {
		sites := deploymentFieldSources(t, rel, pdTransferBaseLatencyField)
		if len(sites) == 0 {
			t.Errorf("%s no longer assigns %s: the PD-transfer base latency must reach the "+
				"DeploymentConfig from --%s on this path, or a run/replay silently uses the zero "+
				"value instead of the operator's (rule 3 of #1838, INV-13)",
				rel, pdTransferBaseLatencyField, flagName)
			continue
		}
		for _, site := range sites {
			if site.source != pdTransferBaseLatencyFlagVar {
				t.Errorf("%s:%d feeds %s from %s, want %s: the number is a CLI/registry input "+
					"(blis-registry#10), and a fabric class carries none to source it from (#1838)",
					rel, site.line, pdTransferBaseLatencyField, site.source, pdTransferBaseLatencyFlagVar)
			}
		}
	}
}

// fieldSource is one site where a DeploymentConfig field is given a value, and the expression it
// is given: the identifier's name, or a placeholder when the value is not a bare identifier
// (which is itself a finding for a field that must come straight from a flag variable).
type fieldSource struct {
	line   int
	source string
}

// deploymentFieldSources returns every site in one repository file that sets the named struct
// field, whether as a composite-literal key (`Field: x`) or an assignment (`cfg.Field = x`).
// Both forms are recognized so the guard survives a refactor of how the config is built.
func deploymentFieldSources(t *testing.T, rel, field string) []fieldSource {
	t.Helper()
	fset := token.NewFileSet()
	parsed, err := parser.ParseFile(fset, filepath.Join("..", rel), nil, parser.SkipObjectResolution)
	if err != nil {
		t.Fatalf("parse %s: %v", rel, err)
	}
	var sites []fieldSource
	record := func(pos token.Pos, value ast.Expr) {
		source := "an expression that is not a bare identifier"
		if ident, ok := value.(*ast.Ident); ok {
			source = ident.Name
		}
		sites = append(sites, fieldSource{line: fset.Position(pos).Line, source: source})
	}
	ast.Inspect(parsed, func(n ast.Node) bool {
		switch node := n.(type) {
		case *ast.KeyValueExpr:
			if key, ok := node.Key.(*ast.Ident); ok && key.Name == field {
				record(node.Pos(), node.Value)
			}
		case *ast.AssignStmt:
			for i, lhs := range node.Lhs {
				sel, ok := lhs.(*ast.SelectorExpr)
				if !ok || sel.Sel.Name != field || i >= len(node.Rhs) {
					continue
				}
				record(lhs.Pos(), node.Rhs[i])
			}
		}
		return true
	})
	return sites
}

// ---------------------------------------------------------------------------
// BC-4: the reader cannot land without rule 2 being read
// ---------------------------------------------------------------------------

// catalogLoadReportKnownFields is the loader's namespace report as it stands while the `networks/`
// fabric reader is absent: the four namespace counts cmd/catalog_load.go's hardcoded list loads,
// plus the accumulated problems. BC-4 compares the report against this set EXACTLY, rather than
// looking for a field whose name says "network" — a reader reported as `Interconnects`, or folded
// into an embedded struct, would evade any name test, and a namespace count that grew into a
// nested struct would evade a name-and-count one.
//
// A slice, not a map, so the diagnostics come out in one order whatever the run (INV-6's habit).
var catalogLoadReportKnownFields = []struct {
	name string
	kind reflect.Kind
}{
	{"Models", reflect.Int},
	{"Hardware", reflect.Int},
	{"Presets", reflect.Int},
	{"DeviceClasses", reflect.Int},
	{"Problems", reflect.Slice},
}

// TestNetworksFabric_ReaderHasNotLandedYet is BC-4, a deliberate TRIPWIRE rather than a
// regression test. #1838 is a coordination note whose whole purpose is that the `networks/`
// fabric reader be authored field-free; BC-1 and BC-2 catch the wrong implementation, but only a
// tripwire on the loader's own namespace report guarantees the rules are in front of whoever
// writes it.
//
// It fails exactly once, when catalogLoadReport grows past the shape above — that is, when a new
// namespace reader lands. If that reader is the `networks/` one, the response is to obey the
// three rules in the failure message, keep BC-1/BC-2/BC-3 (which then guard real code), and drop
// this function in the reader's own PR. If it is an unrelated namespace (R2 adds `clusters` too,
// #1817), the response is to add its field to catalogLoadReportKnownFields, having read the rules
// on the way past — which is the cost of the tripwire and the reason it is one test, not a habit.
func TestNetworksFabric_ReaderHasNotLandedYet(t *testing.T) {
	typ := reflect.TypeOf(catalogLoadReport{})
	if typ.NumField() == 0 {
		t.Fatal("non-vacuity: catalogLoadReport has no fields to inspect")
	}
	known := make(map[string]reflect.Kind, len(catalogLoadReportKnownFields))
	for _, f := range catalogLoadReportKnownFields {
		known[f.name] = f.kind
	}
	seen := make(map[string]bool, typ.NumField())
	for i := 0; i < typ.NumField(); i++ {
		field := typ.Field(i)
		seen[field.Name] = true
		want, isKnown := known[field.Name]
		if !isKnown {
			t.Errorf("catalogLoadReport.%s is a namespace the loader did not read when #1838 was "+
				"written. If it is the networks/ fabric reader (under any name), re-read #1838 before "+
				"relying on it: (1) the fabric's %s (nominal) is the PD-transfer bandwidth, there is no "+
				"separate PD figure; (2) the fabric class carries NO %s (blis-catalog#12 removed it and "+
				"the closed schema rejects it); (3) the PD-transfer base latency stays sourced from "+
				"--pd-transfer-base-latency, owned by blis-registry#10 — the effective value is that "+
				"number alone. Then delete this tripwire in the reader's PR. If it is an unrelated "+
				"namespace, add %q to catalogLoadReportKnownFields and keep going.",
				field.Name, fabricBandwidthKey, pdTransferBaseLatencyField, field.Name)
			continue
		}
		if field.Type.Kind() != want {
			t.Errorf("catalogLoadReport.%s is now a %s, not a %s: a namespace count that grew a shape "+
				"is a reader gaining structure, so re-read #1838's three rules (a fabric class carries "+
				"no %s) before relying on it, then update catalogLoadReportKnownFields.",
				field.Name, field.Type.Kind(), want, pdTransferBaseLatencyField)
		}
	}
	for _, f := range catalogLoadReportKnownFields {
		if !seen[f.name] {
			t.Errorf("catalogLoadReport no longer has %s: this tripwire's known-field set is stale, so "+
				"it can no longer tell a new namespace reader from an old one — update "+
				"catalogLoadReportKnownFields (#1838)", f.name)
		}
	}
}
